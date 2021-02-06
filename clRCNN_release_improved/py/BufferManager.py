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
	def __init__(self, ctx_mgr, pool_size = 100*1024*1024, region = "g"):
		self.ctx_mgr = ctx_mgr
		self.region = region
		self.pool_size = pool_size
		

		self.buf_map = {} # {name: [Buffer1, Buffer2, ...]}
		self.init_candidate = []

	def getRegion(self, region):
		return BufferRegion(self, region)

  # 모든 메모리 할당이 끝난 후 불러줘야 한다.
	def initialize(self, q):
		for buf in self.init_candidate:
			buf.dev_buf = pyCL.Buffer(self.ctx_mgr.ctx, buf.mem_flag, buf.size * np.dtype(buf.dtype).itemsize )

			if buf.init is not None:
				pyCL.enqueue_copy(q, buf.dev_buf, buf.init, is_blocking = False)				

		q.flush()
		q.finish()

		# 메모리를 아끼기 위해 초기화 후 host buffer(init) 삭제
		for buf in self.init_candidate:
			del buf.init
			buf.init = None
		self.init_candidate = []
		#print(self.buf_map)
	
	def readBuf(self, name, q):
		buf = self.buf_map[name][0]
		rtn = np.empty(buf.shape, dtype = buf.dtype)
		pyCL.enqueue_copy(q, rtn, buf.getDevBuf())
		return rtn

# BufferRegion을 통해 할당되는 메모리는 모두 재사용되는 메모리로 간주하고 항구적으로 할당한다.
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
			dtype = np.float32, share = False, region_overide = None):
		if region_overide is not None:
			buf_name = region_overide + "." + name
		else:
			buf_name = self.buf_mgr.region + "." + name
		buf = None

		if buf_name not in self.buf_mgr.buf_map:
			buf = Buffer(self.buf_mgr, buf_name, shape,	mem_flag, init, dtype)
			self.buf_mgr.buf_map[buf_name] = buf
			self.buf_mgr.init_candidate.append(buf)
		else:
			if(share == False):
				print("un-shared buffer is collision: " + buf_name)
				assert(0)
			else:
				origin_buf = self.buf_mgr.buf_map[buf_name]

				if origin_buf.size != reduce(mul, shape):
					print("shared buffer does not have same size: " + buf_name)
					assert(0)
				if origin_buf.mem_flag != mem_flag:
					print("shared buffer does not have same mem flag: " + buf_name)
					assert(0)
				if origin_buf.dtype != dtype:
					print("shared buffer does not have same data type: " + buf_name)
					assert(0)				
				buf = origin_buf				
		return buf

class Buffer:
	def __init__(self, buf_mgr, name, shape, mem_flag, init = None, 
			dtype = np.float32):
		#self.buf_mgr = buf_mgr
		self.name = name
		self.shape = shape
		self.size = reduce(mul, shape)
		self.mem_flag = mem_flag
		self.init = init
		self.dtype = dtype
		self.dev_buf = None

	def toDev(self):
		return self.dev_buf

    