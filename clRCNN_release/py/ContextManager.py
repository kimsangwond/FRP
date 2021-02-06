#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pyopencl as pyCL

# mainly store single context 
# handle context, platform, device and queue 
class ContextManager:
	def __init__(self, plf_id, dev_ids, prog_path, cache_path = None):
		print("Available platform:")
		for idx, plf in enumerate(pyCL.get_platforms()):
				print('\t%d: %s'%(idx,plf.name))
		self.plf = pyCL.get_platforms()[plf_id]
		print("\tselected platform: %d"%(plf_id))

		print("Available devices:")
		for idx, dev in enumerate(self.plf.get_devices()):
				print('\t%d: %s'%(idx,dev.name))
		self.m_dev = {dev_id: self.plf.get_devices()[dev_id] for dev_id in dev_ids}
		print("\tselected devices: %s"%(tuple(dev_ids).__str__()))

		self.ctx = pyCL.Context(self.m_dev.values(), cache_dir = cache_path)

		with open("./cl/kernel.cl", "r") as f:
			code = "".join(f.readlines())

		self.prg = pyCL.Program(self.ctx, code).build()

	def getQ(self, dev_id):
		return pyCL.CommandQueue(self.ctx, self.m_dev[dev_id])
