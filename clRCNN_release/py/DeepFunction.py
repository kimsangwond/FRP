#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Operator import *
from RCNNOperator import *
from Activation import *


import numpy as np
import caffe_pb2 as caffe_pb2
from google.protobuf import text_format
import pyopencl as pyCL
import sys

def str2class(s):
	return getattr(sys.modules[__name__], s)

class DeepFunction:
	def __init__(self, proto, model, buf_region):
		# activation mapping: name to Buffer
		self.buf_region = buf_region
		self.m_act = {} 

		# read network structure, trained weight
		print("Deep function creation:\tread proto and weight...")
		net = caffe_pb2.NetParameter()
		with open(proto, "r") as f:
			text_format.Merge(str(f.read()), net)
        
		weight = caffe_pb2.NetParameter()
		with open(model, "rb") as f:
			weight.ParseFromString(f.read())
		print("Deep function creation:\tcreate operators...")
		
		# input  
		for i in xrange(len(net.input)):
			self.m_act[net.input[i]] = \
				Activation(net.input[i], tuple(net.input_shape[i].dim))

		self._input = {net.input[i]: (net.input[i], tuple(net.input_shape[i].dim))
			for i in xrange(len(net.input))}

		# operators
		self._ops = []

		weight_map = {}
		for l in weight.layer:
			weight_map[l.name] = l

		for l in net.layer:
			l_blob = weight_map[l.name].blobs if l.name in weight_map else None

			if l.type == "Python" and l.python_param.layer == "ProposalLayer":
				self._ops.append(Proposal(l, l_blob, self))
			else:
				self._ops.append(str2class(l.type)(l, l_blob, self))
			


		
		# output
		#print(self.m_act)	
		self._output = {}
		for act in self.m_act.values():
			if act.ref_count == 0:
				self._output[act.name] = act
		#print(self._output)

		# activation reservation
		# TODO: minimize activation memory resouce by layer-wise buffer sharing
		# each operator should implement alloc_act function 
		# current version simply assigns all activation seperatly
		for act in self.m_act.values():
			if act.shared is None:
				act.buf = buf_region.getBuffer(act.name, act.shape, 
					pyCL.mem_flags.READ_WRITE)
			else:
				# TODO: implement smart buffer sharing method
				#act.buf = self.m_act[act.shared].buf
				pass 


	def run(self, q, input_map):
		for key, value in input_map.items():
			pyCL.enqueue_copy(q, self.m_act[key].buf.getDevBuf(),
				value.copy(), is_blocking = False)
		
		for op in self._ops:
			op.forward(q)
		
		rtn_map = {}
		for key, value in self._output.items():
			if value.host:
				rtn_map[value.name] = value.host_buf
			else:
				buf = np.empty(value.shape, dtype=np.float32)
				pyCL.enqueue_copy(q, buf, value.buf.getDevBuf(), is_blocking = False )
				rtn_map[value.name] = buf
		
		q.flush()
		q.finish()

		return rtn_map
	def getActivation(self, q, name):
		act = self.m_act[name]
		host_buf = np.empty(act.shape, dtype=np.float32)
		pyCL.enqueue_copy(q, host_buf, act.buf.getDevBuf())
		return host_buf
                