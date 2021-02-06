#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from Operator import *
from RCNNOperator import *

import numpy as np
import pyopencl as pyCL
import pyopencl.array as pyArr

class DevBuffer:
	def __init__(self, devTT):
		self.devTT = devTT

	def toHost(self, is_blocking = True):
		if self.devTT.pos != "h":
			self.devTT.pos = "h"
			self.devTT.host_buf = np.empty(self.devTT.shape, dtype = np.float32)
			pyCL.enqueue_copy(self.devTT.deep_func.q, self.devTT.host_buf, self.devTT.buf, is_blocking = is_blocking)
		return self.devTT.host_buf

	def toDev(self):
		if self.devTT.pos != "d":
			self.devTT.pos = "d"
			pyCL.enqueue_copy(self.devTT.deep_func.q, self.devTT.buf, self.devTT.host_buf, is_blocking = False)
		return self.devTT.buf
	
	def __del__(self):
		if self.devTT.ref_count <= self.devTT.used_count+1:
			self.devTT.used_count = 0
			#del self.devTT.buf
			self.devTT.buf = None
			self.devTT.pos = "d"
			self.devTT.host_buf = None
		else:
			self.devTT.used_count += 1

class DevTempTensor:
	def __init__(self, name, deep_func):
		self.name = name
		self.deep_func = deep_func

		self.shape = None
		self.size = None
		self.dtype = None
		self.pos = "d"

		self.ref_count = 0 
		self.used_count = 0
		self.buf = None
		self.host_buf = None

		self.pre_buf = None
		self.pre_size = 0

	def ref(self):
		self.ref_count += 1

	def setShape(self, shape, dtype = np.float32):
		if(self.shape != shape):
			self.pos = "d"
			self.host_buf = None
			self.buf = None
			self.shape = shape
			self.size = reduce(mul, self.shape)
			self.dtype = dtype		

	def getDevBuf(self):
		# TODO pyopencl memory pool 사용시 메모리 누수 현상 존재. 수정 필요
		#if self.buf is None:
		#	buf_size = int(self.size * np.dtype(self.dtype).itemsize * 1.1)
		#	self.buf = self.deep_func.pool.allocate(buf_size )
		#return DevBuffer(self)

		buf_size = self.size * np.dtype(self.dtype).itemsize

		if self.buf is None:
			if self.pre_buf is None or self.pre_size < buf_size:
				self.pre_size = int(buf_size* 1.1) #넉넉하게 여유분 확보
				self.pre_buf = pyCL.Buffer(self.deep_func.buf_region.buf_mgr.ctx_mgr.ctx, pyCL.mem_flags.READ_ONLY, self.pre_size)
				
			self.buf = self.pre_buf
		return DevBuffer(self)


class DeepFunction(object):
	def __init__(self, buf_region, q):
		# activation mapping: name to Buffer
		self.buf_region = buf_region
		self.q = q
		self.pool = pyCL.tools.MemoryPool(pyCL.tools.ImmediateAllocator(self.q))
		self.m_act = {} 
		self.input = {}
		self.output = {}
		self.ops = []

	def getDevTempTensor(self, name, alloc = None):
		if alloc == None:
			if name in self.m_act:
				return self.m_act[name]
			else:
				t = DevTempTensor(name, self)
				self.m_act[name] = t
				return t
		elif alloc == True:
			if name in self.m_act:
				assert(0)
			t = DevTempTensor(name, self)
			self.m_act[name] = t
			return t
		else: #alloc == False:
			return self.m_act[name]						

	def run(self, q, input_map, output_in = []):
		for key, value in input_map.items():
			self.m_act[key].setShape(value.shape)
			pyCL.enqueue_copy(q, self.m_act[key].getDevBuf().toDev(), value.copy(), is_blocking = False)
		
		for output in output_in:
			self.m_act[output].ref_count += 1

		for op in self.ops:
			op.reshape()
			op.forward(q)
		
		rtn_map = {}
		output_lst = []

		for key, value in self.output.items():
			output_lst.append(value.getDevBuf())
			rtn_map[value.name] = output_lst[-1].toHost()

		for output in output_in:
			output_lst.append(self.m_act[output].getDevBuf())
			rtn_map[output] = output_lst[-1].toHost()
			self.m_act[output].ref_count -= 1
		
		q.flush()
		q.finish()

		return rtn_map
                