import sys
sys.path.append('./py')
sys.path.append('./faster_rcnn_codes')

import numpy as np
import pyopencl as pyCL

from ContextManager import ContextManager 
from BufferManager import BufferManager
from DeepFunction import DeepFunction

ctx_mgr = ContextManager(0, [0], "./cl/kernel.cl", './cache')
buf_mgr = BufferManager(ctx_mgr)

a_np = np.random.rand(50000).astype(np.float32)
b_np = np.random.rand(50000).astype(np.float32)
res_np = np.empty_like(a_np)

with buf_mgr.getRegion('test') as br:
	a_g = br.getBuffer('a', a_np.shape, pyCL.mem_flags.READ_ONLY, a_np)
	b_g = br.getBuffer('b', b_np.shape, pyCL.mem_flags.READ_ONLY, b_np)
	res_g = br.getBuffer('res', res_np.shape, pyCL.mem_flags.WRITE_ONLY)

q = ctx_mgr.getQ(0)
buf_mgr.initialize(q)

ctx_mgr.prg.sum(q, a_np.shape, None, a_g.toDev(), b_g.toDev(), res_g.toDev())
pyCL.enqueue_copy(q, res_np, res_g.toDev())

print(res_np - (a_np + b_np))
print(np.linalg.norm(res_np - (a_np + b_np)))

