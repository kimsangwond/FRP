#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyopencl as pyCL
import numpy as np
#import logging
from operator import mul

# context에 할당될 gpu buffer를 관리한다.
# ctx_mgr: ContextManager 
class BufferManager:    
	def __init__(self, ctx_mgr, region = "g"):
		self.ctx_mgr = ctx_mgr
		self.region = region

		self.buf_map = {} # {name: [Buffer1, Buffer2, ...]}
 	  #	self.init_candidate = []

	def getRegion(self, region):
		return BufferRegion(self, region)

  # 모든 메모리 할당이 끝난 후 불러줘야 한다.
	def initialize(self, q):
		for name, lst in self.buf_map.items():
			shape = lst[0].shape
			mem_flag = lst[0].mem_flag
			init = lst[0].init
			dtype = lst[0].dtype
			reuse = lst[0].reuse
			max_size = reduce(mul, shape) * np.dtype(dtype).itemsize

			for l in lst[1:]:
				if shape != l.shape:
					shape = None
					max_size = max(max_size, reduce(mul, l.shape)*np.dtype(dtype).itemsize)
				if mem_flag != l.mem_flag:
					raise("Shared buffer %s has multiple memory flags" %(name))
				if np.array_equal(init, l.init) == False:
					if init is None:
						init = l.init
					else:
						raise("Shared buffer %s has multiple initialization sources"%(name))
				if dtype != l.dtype:
					max_size = max(max_size, reduce(mul, l.shape)*np.dtype(dtype).itemsize)
					print("Shared buffer %s is shared among the different datatype buffers"%(name))
				reuse |= l.reuse
			if shape is None:
				print("Shared buffer %s is shared among the different size buffers"%(name))
			dev_buf = pyCL.Buffer(self.ctx_mgr.ctx, mem_flag, max_size)

			if init is not None:
				pyCL.enqueue_copy(q, dev_buf, init, is_blocking = False)
			else:
				# nan에 의해 발생하는 버그 방지를 위하여, 0 initialization
				temp = np.zeros((max_size), dtype = np.int8)
				pyCL.enqueue_copy(q, dev_buf, temp, is_blocking = False)
			
			for l in lst:
				l.dev_buf = dev_buf
		q.flush()
		q.finish()
		#print(self.buf_map)
	
	def readBuf(self, name, q):
		buf = self.buf_map[name][0]
		rtn = np.empty(buf.shape, dtype = buf.dtype)
		pyCL.enqueue_copy(q, rtn, buf.getDevBuf())
		return rtn

class BufferRegion:
	def __init__(self, buf_mgr, region):
		self.buf_mgr = buf_mgr
		self.region = region

	def __enter__(self):
		self.prev_region = self.buf_mgr.region
		self.buf_mgr.region += "/" + self.region
		return self

	def __exit__(self, type, value, traceback):
		self.buf_mgr.region = self.prev_region
		pass
    
	def getRegion(self, region):
		return BufferRegion(self.buf_mgr, region)

	def getBuffer(self, name, shape, mem_flag, init = None, 
			dtype = np.float32, reuse = False, region_overide = None):
		if region_overide is not None:
			buf_name = region_overide + "." + name
		else:
			buf_name = self.buf_mgr.region + "." + name

		buf = Buffer(self.buf_mgr, buf_name, 
			shape,	mem_flag, init, dtype, reuse)
					
		if buf_name in self.buf_mgr.buf_map:
			self.buf_mgr.buf_map[buf_name].append(buf)
		else:
			self.buf_mgr.buf_map[buf_name] = [buf]

		return buf

class Buffer:
	def __init__(self, buf_mgr, name, shape, mem_flag, init = None, 
			dtype = np.float32, reuse = False):
		#self.buf_mgr = buf_mgr
		self.name = name
		self.shape = shape
		self.mem_flag = mem_flag
		self.init = init
		self.dtype = dtype
		self.reuse = reuse		
		self.dev_buf = None
		self.related = []

	def getDevBuf(self):
		return self.dev_buf

    