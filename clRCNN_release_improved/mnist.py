#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
sys.path.append('./py')
sys.path.append('./faster_rcnn_codes')
sys.path.append('./mnist')

import numpy as np
from ContextManager import ContextManager 
from BufferManager import BufferManager
from DeepFunction import DeepFunction
from CaffeFunction import CaffeFunction
from load_mnist import load_mnist

ctx_mgr = ContextManager(0, [0], "./cl/kernel.cl", './cache')
buf_mgr = BufferManager(ctx_mgr)

q = ctx_mgr.getQ(0)
with buf_mgr.getRegion('net') as bg:
	df = CaffeFunction('./mnist/lenet_mod.prototxt', './mnist/lenet_mod.caffemodel', bg, q)

buf_mgr.initialize(q)

images, labels = load_mnist("testing", path="./mnist")
images = images.astype(np.float32)/255

idx = 10
rtn = df.run(q, {'data': images[idx,:,:].reshape(1,1,28,28)})
print(rtn['prob'])
print(labels[idx,:])
print(np.argmax(rtn['prob']))
