#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from py.Activation import *
import numpy as np
import pyopencl as pyCL

from operator import mul

LOCAL_SIZE = np.int32(64)
def globalSize(size, local_size = LOCAL_SIZE):
	return int((size+local_size-1)/local_size)*local_size

class Operator(object):
	def __init__(self, proto, deep_func):
		self.name = proto.name
		self.type = proto.type
		self.input = proto.bottom
		self.output = proto.top
		self.deep_func = deep_func
		self.prg = self.deep_func.buf_region.buf_mgr.ctx_mgr.prg
		self.ctx = self.deep_func.buf_region.buf_mgr.ctx_mgr.ctx
		
		self.act_temp = []
		self.blob_buf = []
		#self.blob_host = []
		#self.blob_dev = []

	def alloc_act(self):
		pass
	def forward(self, q):
		raise("Operator %s does not implement forward operation"%(self.name))

# caffe의 default convolution operation을 모방
# caffe와는 다르게 top/bottom이 한 쌍만 존재한다 가정
# 현재로서는 group은 지원 안함 (AlexNet 외에 사용 안함)
class Convolution(Operator):
	def __init__(self, proto, blob, deep_func):
		super(Convolution, self).__init__(proto, deep_func)

		# initialize internal parameters
		cp = proto.convolution_param
		def out_shape(self, idx, in_shape):
			if idx == 0: 		# batch
				return in_shape[0]
			elif idx == 1:	# channel
				return cp.num_output
			else: # idx == 2 or idx == 3, height and width
				return int( np.floor((in_shape[idx]+2*self.pad-self.kernel_size)/self.stride) ) +1
		
		self.pad = 0 if cp.pad == [] else cp.pad[0]
		self.stride = 1 if cp.stride == [] else cp.stride[0]
		self.dilation = 1 if cp.dilation == [] else cp.dilation[0]
		self.bias_term = cp.bias_term
		self.kernel_size = cp.kernel_size[0]
		self.group = cp.group

		# store real weight
		if blob is None:
			raise("Convolution layer %s needs trained parameter"%(proto.name))		
		
		with deep_func.buf_region.getRegion(self.name) as br:
			weight_buf = br.getBuffer("weight", tuple(blob[0].shape.dim),
				pyCL.mem_flags.READ_ONLY, reuse = True,
				init = np.array(blob[0].data).astype(np.float32))
			self.blob_buf.append(weight_buf)

			if self.bias_term:
				bias_buf = br.getBuffer("bias", tuple(blob[1].shape.dim),
					pyCL.mem_flags.READ_ONLY, reuse = True, 
					init = np.array(blob[1].data).astype(np.float32))
				self.blob_buf.append(bias_buf)

		input_act = deep_func.m_act[self.input[0]]
		input_act.ref_count += 1
		input_shape = input_act.shape
		output_shape = tuple(
					out_shape(self, idx, input_shape) for idx in xrange(len(input_shape)) )
				
		if self.output[0] in deep_func.m_act:
			raise("Output %s is pre allocated??"%(self.output[0]))
		deep_func.m_act[self.output[0]] = Activation(self.output[0], output_shape)

		#im2col operator for matrix multiplication
		# M = output channel
		# K = input channel * kernel height * kernel width
		# N = output height * output width
		self.M = output_shape[1]
		self.K = input_shape[1] * self.kernel_size * self.kernel_size
		self.N = output_shape[2] * output_shape[3]

		act_temp = Activation(self.name + "_i2c", (self.K, self.N))
		act_temp.ref_count += 1
		self.act_temp.append(act_temp)
		deep_func.m_act[self.name + "_i2c"] = act_temp

	def forward(self, q):		
		input = self.deep_func.m_act[self.input[0]]
		output = self.deep_func.m_act[self.output[0]]
		temp = self.deep_func.m_act[self.name + "_i2c"]

		weight = self.blob_buf[0]		
		if self.bias_term:
			bias = self.blob_buf[1]

		im2col_num = input.shape[1]*output.shape[2]*output.shape[3]
		self.prg.im2col_kernel(q, 
			(globalSize(im2col_num),),(LOCAL_SIZE,), 
			np.int32(im2col_num), input.buf.getDevBuf(), 
			np.int32(input.shape[2]),np.int32(input.shape[3]),
			np.int32(weight.shape[2]), np.int32(weight.shape[3]),
			np.int32(self.pad), np.int32(self.pad),
			np.int32(self.stride), np.int32(self.stride),
			np.int32(self.dilation), np.int32(self.dilation),
			np.int32(output.shape[2]), np.int32(output.shape[3]),
			temp.buf.getDevBuf()
		)

		self.prg.poor_matmul(q, 
			(globalSize(self.N, 8),globalSize(self.M, 8)), (8,8),
			np.int32(self.M), np.int32(self.N), np.int32(self.K),
			np.float32(1.), weight.getDevBuf(), temp.buf.getDevBuf(),
			np.float32(0.), output.buf.getDevBuf()
		)

		if self.bias_term:
			self.prg.add_bias(q, (globalSize(output.size()),), (LOCAL_SIZE,),
				np.int32(output.size()), np.int32(output.shape[1]), 
				np.int32(output.shape[2]*output.shape[3]),
				bias.getDevBuf(), output.buf.getDevBuf()
			)

class ReLU(Operator):
	def __init__(self, proto, blob, deep_func):
		super(ReLU, self).__init__(proto, deep_func)
		
		self.in_place = (self.input[0] == self.output[0])

		if self.in_place == False:
			input_act = deep_func.m_act[self.input[0]]
			input_act.ref_count += 1
			
			if self.output[0] in deep_func.m_act:
				raise("Output %s is pre allocated??"%(self.output[0]))
			deep_func.m_act[self.output[0]] = \
				Activation(self.output[0], input_act.shape)
	def forward(self, q):
		input = self.deep_func.m_act[self.input[0]]
		output = self.deep_func.m_act[self.output[0]]

		self.prg.relu(q, (globalSize(output.size()),), (LOCAL_SIZE,),
			np.int32(output.size()), 
			input.buf.getDevBuf(), output.buf.getDevBuf() 
		 	)



class Pooling(Operator):
	def __init__(self, proto, blob, deep_func):
		super(Pooling, self).__init__(proto, deep_func)

		pp = proto.pooling_param
		def out_shape(self, idx, in_shape):
			if idx == 0:
				return in_shape[0]
			elif idx == 1:
				return in_shape[1]
			else: # idx == 2 or idx == 3
				return int( np.ceil((in_shape[idx]+2*self.pad-self.kernel_size)/self.stride) ) +1
	
		self.pad = pp.pad
		self.stride = pp.stride        
		self.kernel_size = pp.kernel_size
		self.method = pp.pool   # MAX: 0, AVE:1, STOCHASTIC: 2

		input_act = deep_func.m_act[self.input[0]]
		input_act.ref_count += 1
		input_shape = input_act.shape
		output_shape = tuple(
				[out_shape(self, idx, input_shape) for idx in xrange(len(input_shape))] )
		
		if self.output[0] in deep_func.m_act:
			raise("Output %s is pre allocated??"%(self.output[0]))		
		deep_func.m_act[self.output[0]] = Activation(self.output[0], output_shape)


	def forward(self, q):
		input = self.deep_func.m_act[self.input[0]]
		output = self.deep_func.m_act[self.output[0]]

		if self.method == 0:
			self.prg.max_pool_kernel(q, (globalSize(output.size()),), (LOCAL_SIZE,),
				np.int32(output.size()), input.buf.getDevBuf(), 
				np.int32(input.shape[0]), np.int32(input.shape[1]),
				np.int32(input.shape[2]), np.int32(input.shape[3]),
				np.int32(output.shape[2]), np.int32(output.shape[3]),
				np.int32(self.kernel_size), np.int32(self.kernel_size),
				np.int32(self.stride), np.int32(self.stride),
				np.int32(self.pad), np.int32(self.pad),
				output.buf.getDevBuf() 
				)
		else:
			raise("Other pooling methods are not implemented")


class InnerProduct(Operator):
	def __init__(self, proto, blob, deep_func):
		super(InnerProduct, self).__init__(proto, deep_func)

		# initialize internal parameters
		ip = proto.inner_product_param
		self.bias_term = ip.bias_term

		# store real weight
		if blob is None:
			raise("Inner product layer %s needs trained parameter"%(proto.name))		
		
		with deep_func.buf_region.getRegion(self.name) as br:
			weight_data = np.array(blob[0].data).astype(np.float32)
			weight_data = weight_data.reshape(tuple(blob[0].shape.dim))
			weight_data = np.transpose(weight_data).copy()
			#print(blob[0].shape.dim)

			weight_buf = br.getBuffer("weight", weight_data.shape,
				pyCL.mem_flags.READ_ONLY, reuse = True,
				init = weight_data)
			self.blob_buf.append(weight_buf)

			if self.bias_term:
				bias_buf = br.getBuffer("bias", tuple(blob[1].shape.dim),
					pyCL.mem_flags.READ_ONLY, reuse = True, 
					init = np.array(blob[1].data).astype(np.float32))
				self.blob_buf.append(bias_buf)

		input_act = deep_func.m_act[self.input[0]]
		input_act.ref_count += 1
		input_shape = input_act.shape
		#print(input_shape)
		output_shape = (input_shape[0], ip.num_output)
				
		if self.output[0] in deep_func.m_act:
			raise("Output %s is pre allocated??"%(self.output[0]))
		deep_func.m_act[self.output[0]] = Activation(self.output[0], output_shape)

		# matrix multiplication
		# M = input num
		# K = input remains
		# N = num_output
		self.M = input_shape[0]
		self.K = reduce(mul, input_shape[1:])
		self.N = output_shape[1]

		#print((self.M,self.K,self.N))

	def forward(self, q):		
		input = self.deep_func.m_act[self.input[0]]
		output = self.deep_func.m_act[self.output[0]]
		

		weight = self.blob_buf[0]		
		if self.bias_term:
			bias = self.blob_buf[1]

		if self.M == 1:
			self.prg.poor_matmul2(q, 
				(globalSize(self.N, ),), (LOCAL_SIZE,),
				np.int32(self.M), np.int32(self.N), np.int32(self.K),
				np.float32(1.), input.buf.getDevBuf(), weight.getDevBuf(), 
				np.float32(0.), output.buf.getDevBuf()
			)
		else:
			self.prg.poor_matmul(q, 
				(globalSize(self.N, 8),globalSize(self.M, 8)), (8,8),
				np.int32(self.M), np.int32(self.N), np.int32(self.K),
				np.float32(1.), input.buf.getDevBuf(), weight.getDevBuf(),
				np.float32(0.), output.buf.getDevBuf()
			)

		if self.bias_term:
			self.prg.add_bias(q, (globalSize(output.size()),), (LOCAL_SIZE,),
				np.int32(output.size()), np.int32(output.shape[1]), 
				np.int32(1),
				bias.getDevBuf(), output.buf.getDevBuf()
			)

class LRN(Operator):
	def __init__(self, proto, blob, deep_func):
		super(LRN, self).__init__(proto, deep_func)

		lp = proto.lrn_param

		self.local_size = lp.local_size
		self.alpha = lp.alpha        
		self.beta = lp.beta
		self.norm_region = lp.norm_region   # across ch: 0, inter ch: 1

		input_act = deep_func.m_act[self.input[0]]
		input_act.ref_count += 1
		input_shape = input_act.shape
		output_shape = input_act.shape

		if self.output[0] in deep_func.m_act:
			raise("Output %s is pre allocated??"%(self.output[0]))		
		deep_func.m_act[self.output[0]] = Activation(self.output[0], output_shape)

		act_temp = Activation(self.name + "_temp", input_shape)
		act_temp.ref_count += 1
		self.act_temp.append(act_temp)
		deep_func.m_act[self.name + "_temp"] = act_temp


	def forward(self, q):
		input = self.deep_func.m_act[self.input[0]]
		output = self.deep_func.m_act[self.output[0]]
		temp = self.deep_func.m_act[self.name + "_temp"]

		if self.norm_region == 1:
			self.prg.square_kernel(q, 
				(globalSize(output.size()),), (LOCAL_SIZE,),
				np.int32(globalSize(output.size())),
				input.buf.getDevBuf(), temp.buf.getDevBuf()
				)

			self.prg.lrn_inter_kernel(q, 
				(globalSize(output.size()),), (LOCAL_SIZE,),
				np.int32(output.size()), input.buf.getDevBuf(), 
				np.int32(input.shape[0]), np.int32(input.shape[1]),
				np.int32(input.shape[2]), np.int32(input.shape[3]),
				np.int32(self.local_size), 
				np.float32(self.alpha), np.float32(self.beta),
				temp.buf.getDevBuf(),	output.buf.getDevBuf() 
				)
		else:
			raise("Other normalization methods are not implemented")

class Reshape(Operator):
	def __init__(self, proto, blob, deep_func):
		super(Reshape, self).__init__(proto, deep_func)
		rp = proto.reshape_param
		self.shape = tuple(rp.shape.dim)

		input_act = deep_func.m_act[self.input[0]]
		input_act.ref_count += 1
		input_shape = input_act.shape

		input_num = reduce(mul, input_shape)
		output_shape = list(input_shape)

		for idx in xrange(len(self.shape)):
			if self.shape[idx] == 0:
				continue
			elif self.shape[idx] == -1:
				reg_idx = idx
				output_shape[idx] = -1
			else:
				output_shape[idx] = self.shape[idx]
		output_num = reduce(mul, output_shape)
		output_shape[reg_idx] = input_num // (output_num * -1)
		output_shape = tuple(output_shape)
		#print(output_shape)
		
		if self.output[0] in deep_func.m_act:
			raise("Output %s is pre allocated??"%(self.output[0]))		
		out_act =  Activation(self.output[0], output_shape)
		out_act.shared =  self.input[0]
		deep_func.m_act[self.output[0]] = out_act
	def forward(self, q):
		input = self.deep_func.m_act[self.input[0]]
		output = self.deep_func.m_act[self.output[0]]
		output.buf = input.buf
		pass

class Softmax(Operator):
	def __init__(self, proto, blob, deep_func):
		super(Softmax, self).__init__(proto, deep_func)	
		sp = proto.softmax_param
		self.axis = sp.axis
		if self.axis != 1:
			raise("This code only supports channel-wise softmax")

		input_act = deep_func.m_act[self.input[0]]
		input_act.ref_count += 1
		input_shape = input_act.shape
		output_shape = input_act.shape

		if self.output[0] in deep_func.m_act:
			raise("Output %s is pre allocated??"%(self.output[0]))		
		deep_func.m_act[self.output[0]] = Activation(self.output[0], output_shape)

		self.temp_buf = np.zeros(input_shape, dtype=np.float32)

	def forward(self, q):
		input = self.deep_func.m_act[self.input[0]]
		output = self.deep_func.m_act[self.output[0]]

		pyCL.enqueue_copy(q, self.temp_buf, input.buf.getDevBuf())

		if len(self.temp_buf.shape) == 4:
			for n in xrange(self.temp_buf.shape[0]):
				self.temp_buf[n,:,:,:] -= np.max(self.temp_buf[n,:,:,:], 0)
				
				self.temp_buf[n,:,:,:] = np.exp(self.temp_buf[n,:,:,:])
				self.temp_buf[n,:,:,:] /= np.sum(self.temp_buf[n,:,:,:], 0)
		else:
			for n in xrange(self.temp_buf.shape[0]):
				self.temp_buf[n,:] -= np.max(self.temp_buf[n,:])
				self.temp_buf[n,:] = np.exp(self.temp_buf[n,:])
				self.temp_buf[n,:] /= np.sum(self.temp_buf[n,:])
		pyCL.enqueue_copy(q, output.buf.getDevBuf(), self.temp_buf, is_blocking = False)


class Dropout(Operator):
	def __init__(self, proto, blob, deep_func):
		super(Dropout, self).__init__(proto, deep_func)	
		dp = proto.dropout_param
		self.threshold = dp.dropout_ratio
		self.scale = 1 / (1 - self.threshold)
		self.scale_train = dp.scale_train

		self.in_place = (self.input[0] == self.output[0])

		if self.in_place == False:
			input_act = deep_func.m_act[self.input[0]]
			input_act.ref_count += 1
			
			if self.output[0] in deep_func.m_act:
				raise("Output %s is pre allocated??"%(self.output[0]))
			deep_func.m_act[self.output[0]] = \
				Activation(self.output[0], input_act.shape)

	def forward(self, q):
		input = self.deep_func.m_act[self.input[0]]
		output = self.deep_func.m_act[self.output[0]]

		if self.in_place == False:
			pyCL.enqueue_copy(q, output.buf.getDevBuf(), input.buf.getDevBuf())

		if self.scale_train == False:
			self.prg.scale_kernel(q, 
				(globalSize(output.size()),), (LOCAL_SIZE,),
				np.int32(globalSize(output.size())),
				np.float32( 1 / self.scale),
				output.buf.getDevBuf()
				)

