#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
		self.blob_buf = []

	def reshape(self):
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
		self.pad = 0 if cp.pad == [] else cp.pad[0]
		self.stride = 1 if cp.stride == [] else cp.stride[0]
		self.dilation = 1 if cp.dilation == [] else cp.dilation[0]
		self.bias_term = cp.bias_term
		self.kernel_size = cp.kernel_size[0]
		self.group = cp.group
		self.num_output = cp.num_output

		# store real weight
		if blob is None:
			raise("Convolution layer %s needs trained parameter"%(proto.name))		
		
		with deep_func.buf_region.getRegion(self.name) as br:
			weight_buf = br.getBuffer("weight", blob[0].shape,
				pyCL.mem_flags.READ_ONLY, init = blob[0])
			self.blob_buf.append(weight_buf)

			if self.bias_term:
				bias_buf = br.getBuffer("bias", blob[1].shape,
					pyCL.mem_flags.READ_ONLY,	init = blob[1])
				self.blob_buf.append(bias_buf)

		self.input_act = deep_func.getDevTempTensor(self.input[0])
		self.input_act.ref()

		self.output_act = deep_func.getDevTempTensor(self.output[0], True)
		self.output_act.ref()

		self.temp_act = deep_func.getDevTempTensor(self.name + "_i2c", True)
		self.temp_act.ref()

	def out_shape(self, idx, in_shape):
			if idx == 0: 		# batch
				return in_shape[0]
			elif idx == 1:	# channel
				return self.num_output
			else: # idx == 2 or idx == 3, height and width
				return int( np.floor((in_shape[idx]+2*self.pad-self.kernel_size)/self.stride) ) +1

	def reshape(self):		
		input_shape = self.input_act.shape
		output_shape = tuple(self.out_shape(idx, input_shape) for idx in xrange(len(input_shape)))
		self.output_act.setShape(output_shape)

		#im2col operator for matrix multiplication
		# M = output channel
		# K = input channel * kernel height * kernel width
		# N = output height * output width
		self.M = output_shape[1]
		self.K = input_shape[1] * self.kernel_size * self.kernel_size
		self.N = output_shape[2] * output_shape[3]

		self.temp_act.setShape((self.K, self.N))

	def forward(self, q):		
		input = self.input_act.getDevBuf()
		output = self.output_act.getDevBuf()
		temp = self.temp_act.getDevBuf()

		input_shape = self.input_act.shape
		output_shape = self.output_act.shape
		output_size = self.output_act.size

		weight = self.blob_buf[0]		
		if self.bias_term:
			bias = self.blob_buf[1]

		im2col_num = input_shape[1]*output_shape[2]*output_shape[3]

		self.prg.im2col_kernel(q, 
			(globalSize(im2col_num),),(LOCAL_SIZE,), 
			np.int32(im2col_num), input.toDev(), 
			np.int32(input_shape[2]),np.int32(input_shape[3]),
			np.int32(weight.shape[2]), np.int32(weight.shape[3]),
			np.int32(self.pad), np.int32(self.pad),
			np.int32(self.stride), np.int32(self.stride),
			np.int32(self.dilation), np.int32(self.dilation),
			np.int32(output_shape[2]), np.int32(output_shape[3]),
			temp.toDev()
		)

		self.prg.poor_matmul(q, 
			(globalSize(self.N, 8),globalSize(self.M, 8)), (8,8),
			np.int32(self.M), np.int32(self.N), np.int32(self.K),
			np.float32(1.), weight.toDev(), temp.toDev(),
			np.float32(0.), output.toDev()
		)

		if self.bias_term:
			self.prg.add_bias(q, (globalSize(output_size),), (LOCAL_SIZE,),
				np.int32(output_size), np.int32(output_shape[1]), 
				np.int32(output_shape[2]*output_shape[3]),
				bias.toDev(), output.toDev()
			)

class ReLU(Operator):
	def __init__(self, proto, blob, deep_func):
		super(ReLU, self).__init__(proto, deep_func)
	
		self.in_place = (self.input[0] == self.output[0])

		self.input_act = deep_func.getDevTempTensor(self.input[0])
		self.input_act.ref()

		if self.in_place:
			self.output_act = self.input_act
		else:
			self.output_act = deep_func.getDevTempTensor(self.output[0], True)
		self.output_act.ref()

	def reshape(self):
		if self.in_place == False:
			self.output_act.setShape(self.input_act.shape)

	def forward(self, q):
		input = self.input_act.getDevBuf()
		output = self.output_act.getDevBuf()

		output_size = self.output_act.size

		self.prg.relu(q, (globalSize(output_size),), (LOCAL_SIZE,),
			np.int32(output_size), 
			input.toDev(), output.toDev()
		 	)

class Pooling(Operator):
	def __init__(self, proto, blob, deep_func):
		super(Pooling, self).__init__(proto, deep_func)

		pp = proto.pooling_param
			
		self.pad = pp.pad
		self.stride = pp.stride        
		self.kernel_size = pp.kernel_size
		self.method = pp.pool   # MAX: 0, AVE:1, STOCHASTIC: 2

		self.input_act = deep_func.getDevTempTensor(self.input[0])
		self.input_act.ref()

		self.output_act = deep_func.getDevTempTensor(self.output[0], True)
		self.output_act.ref()

	def out_shape(self, idx, in_shape):
			if idx == 0:
				return in_shape[0]
			elif idx == 1:
				return in_shape[1]
			else: # idx == 2 or idx == 3
				return int( np.ceil((in_shape[idx]+2*self.pad-self.kernel_size)/self.stride) ) +1
		
	def reshape(self):
		input_shape = self.input_act.shape
		output_shape =  \
			tuple([self.out_shape(idx, input_shape) for idx in xrange(len(input_shape))])
		self.output_act.setShape(output_shape)
		
	def forward(self, q):
		input = self.input_act.getDevBuf()
		output = self.output_act.getDevBuf()

		input_shape = self.input_act.shape
		output_shape = self.output_act.shape
		output_size = self.output_act.size

		if self.method == 0:
			self.prg.max_pool_kernel(q, (globalSize(output_size),), (LOCAL_SIZE,),
				np.int32(output_size), input.toDev(), 
				np.int32(input_shape[0]), np.int32(input_shape[1]),
				np.int32(input_shape[2]), np.int32(input_shape[3]),
				np.int32(output_shape[2]), np.int32(output_shape[3]),
				np.int32(self.kernel_size), np.int32(self.kernel_size),
				np.int32(self.stride), np.int32(self.stride),
				np.int32(self.pad), np.int32(self.pad),
				output.toDev() 
				)
		else:
			raise("Other pooling methods are not implemented")

class InnerProduct(Operator):
	def __init__(self, proto, blob, deep_func):
		super(InnerProduct, self).__init__(proto, deep_func)

		# initialize internal parameters
		ip = proto.inner_product_param
		self.bias_term = ip.bias_term
		self.num_output = ip.num_output

		# store real weight
		if blob is None:
			raise("Inner product layer %s needs trained parameter"%(proto.name))		
		
		with deep_func.buf_region.getRegion(self.name) as br:
			weight_data = np.transpose(blob[0]).copy()
			#print(blob[0].shape.dim)

			weight_buf = br.getBuffer("weight", weight_data.shape,
				pyCL.mem_flags.READ_ONLY, init = weight_data)
			self.blob_buf.append(weight_buf)

			if self.bias_term:
				bias_buf = br.getBuffer("bias", blob[1].shape,
					pyCL.mem_flags.READ_ONLY, init = blob[1])
				self.blob_buf.append(bias_buf)

		self.input_act = deep_func.getDevTempTensor(self.input[0])
		self.input_act.ref()

		self.output_act = deep_func.getDevTempTensor(self.output[0], True)
		self.output_act.ref()

	def reshape(self):
		input_shape = self.input_act.shape
		output_shape = (input_shape[0], self.num_output)
		self.output_act.setShape(output_shape)
				
		# matrix multiplication
		# M = input num
		# K = input remains
		# N = num_output
		self.M = input_shape[0]
		self.K = reduce(mul, input_shape[1:])
		self.N = output_shape[1]

		#print((self.M,self.K,self.N))

	def forward(self, q):		
		input = self.input_act.getDevBuf()
		output = self.output_act.getDevBuf()		

		weight = self.blob_buf[0]		
		if self.bias_term:
			bias = self.blob_buf[1]
		
		output_shape = self.output_act.shape
		output_size = self.output_act.size

		if self.M == 1:
			self.prg.poor_matmul2(q, 
				(globalSize(self.N, ),), (LOCAL_SIZE,),
				np.int32(self.M), np.int32(self.N), np.int32(self.K),
				np.float32(1.), input.toDev(), weight.toDev(), 
				np.float32(0.), output.toDev()
			)
		else:
			self.prg.poor_matmul(q, 
				(globalSize(self.N, 8),globalSize(self.M, 8)), (8,8),
				np.int32(self.M), np.int32(self.N), np.int32(self.K),
				np.float32(1.), input.toDev(), weight.toDev(),
				np.float32(0.), output.toDev()
			)

		if self.bias_term:
			self.prg.add_bias(q, (globalSize(output_size),), (LOCAL_SIZE,),
				np.int32(output_size), np.int32(output_shape[1]), 
				np.int32(1),
				bias.toDev(), output.toDev()
			)

class LRN(Operator):
	def __init__(self, proto, blob, deep_func):
		super(LRN, self).__init__(proto, deep_func)

		lp = proto.lrn_param

		self.local_size = lp.local_size
		self.alpha = lp.alpha        
		self.beta = lp.beta
		self.norm_region = lp.norm_region   # across ch: 0, inter ch: 1

		self.input_act = deep_func.getDevTempTensor(self.input[0])
		self.input_act.ref()

		self.output_act = deep_func.getDevTempTensor(self.output[0], True)
		self.output_act.ref()

		self.temp_act = deep_func.getDevTempTensor(self.name + "_temp", True)
		self.temp_act.ref()

	def reshape(self):
		input_shape = self.input_act.shape
		self.output_act.setShape(input_shape)
		self.temp_act.setShape(input_shape)

	def forward(self, q):
		input = self.input_act.getDevBuf()
		output = self.output_act.getDevBuf()	
		temp = self.temp_act.getDevBuf()

		input_shape = self.input_act.shape
		output_size = self.output_act.size

		if self.norm_region == 1:
			self.prg.square_kernel(q, 
				(globalSize(output_size),), (LOCAL_SIZE,),
				np.int32(globalSize(output_size)),
				input.toDev(), temp.toDev()
				)

			self.prg.lrn_inter_kernel(q, 
				(globalSize(output_size),), (LOCAL_SIZE,),
				np.int32(output_size), input.toDev(), 
				np.int32(input_shape[0]), np.int32(input_shape[1]),
				np.int32(input_shape[2]), np.int32(input_shape[3]),
				np.int32(self.local_size), 
				np.float32(self.alpha), np.float32(self.beta),
				temp.toDev(),	output.toDev() 
				)
		else:
			raise("Other normalization methods are not implemented")

class Reshape(Operator):
	def __init__(self, proto, blob, deep_func):
		super(Reshape, self).__init__(proto, deep_func)
		rp = proto.reshape_param
		self.shape = tuple(rp.shape.dim)

		self.input_act = deep_func.getDevTempTensor(self.input[0])
		self.input_act.ref()

		self.output_act = deep_func.getDevTempTensor(self.output[0], True)
		self.output_act.ref()

	def reshape(self):

		input_shape = self.input_act.shape

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

		self.output_act.setShape(output_shape)

	def forward(self, q):
		input = self.input_act.getDevBuf()
		output = self.output_act.getDevBuf()	
		output.devTT.buf = input.toDev()
		pass

class Softmax(Operator):
	def __init__(self, proto, blob, deep_func):
		super(Softmax, self).__init__(proto, deep_func)	
		sp = proto.softmax_param
		self.axis = sp.axis
		if self.axis != 1:
			raise("This code only supports channel-wise softmax")

		self.input_act = deep_func.getDevTempTensor(self.input[0])
		self.input_act.ref()

		self.output_act = deep_func.getDevTempTensor(self.output[0], True)
		self.output_act.ref()

	def reshape(self):
		input_shape = self.input_act.shape
		self.output_act.setShape(input_shape)

		#self.temp_buf = np.zeros(input_shape, dtype=np.float32)

	def forward(self, q):
		input = self.input_act.getDevBuf()
		output = self.output_act.getDevBuf()

		input_host = input.toHost()

		#pyCL.enqueue_copy(q, self.temp_buf, input.buf.getDevBuf())

		if len(input_host.shape) == 4:
			for n in xrange(input_host.shape[0]):
				input_host[n,:,:,:] -= np.max(input_host[n,:,:,:], 0)
				
				input_host[n,:,:,:] = np.exp(input_host[n,:,:,:])
				input_host[n,:,:,:] /= np.sum(input_host[n,:,:,:], 0)
		else:
			for n in xrange(input_host.shape[0]):
				input_host[n,:] -= np.max(input_host[n,:])
				input_host[n,:] = np.exp(input_host[n,:])
				input_host[n,:] /= np.sum(input_host[n,:])

		pyCL.enqueue_copy(q, output.toDev(), input_host, is_blocking = False)

class Dropout(Operator):
	def __init__(self, proto, blob, deep_func):
		super(Dropout, self).__init__(proto, deep_func)	
		dp = proto.dropout_param
		self.threshold = dp.dropout_ratio
		self.scale = 1 / (1 - self.threshold)
		self.scale_train = dp.scale_train

		self.in_place = (self.input[0] == self.output[0])

		self.input_act = deep_func.getDevTempTensor(self.input[0])
		self.input_act.ref()

		if self.in_place:
			self.output_act = self.input_act
		else:
			self.output_act = deep_func.getDevTempTensor(self.output[0], True)
		self.output_act.ref()

	def reshape(self):
		if self.in_place == False:
			self.output_act.setShape(self.input_act.shape)

	def forward(self, q):
		input = self.input_act.getDevBuf()
		output = self.output_act.getDevBuf()
		output_size = self.output_act.size

		if self.in_place == False:
			pyCL.enqueue_copy(q, output.toDev(), input.toDev())

		if self.scale_train == False:
			self.prg.scale_kernel(q, 
				(globalSize(output_size),), (LOCAL_SIZE,),
				np.int32(globalSize(output_size)),
				np.float32( 1 / self.scale),
				output.toDev()
				)

