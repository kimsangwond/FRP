#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('./build/lib.linux-x86_64-2.7')
sys.path.append('./build/lib.linux-armv7l-2.7')

import proto_reader
import caffe_pb2 as caffe_pb2
from google.protobuf import text_format

from DeepFunction import *

def str2class(s):
	return getattr(sys.modules[__name__], s)

class CaffeFunction(DeepFunction):
  def __init__(self, proto, model, buf_region, q):
    super(CaffeFunction, self).__init__(buf_region, q)

    # read network structure, trained weight
    net = caffe_pb2.NetParameter()
    with open(proto, "r") as f:
			text_format.Merge(str(f.read()), net)
        
    # C++에서 관리하는 메모리 객체는 계속 들고있어야 한다. 
    # TODO: Dynamically alloc/dealloc weight data on python
    self.weight = proto_reader.ProtoReader(model)    
    print("Reading caffe proto and model is finisehd.")

    # input creation/allocation
    for i in xrange(len(net.input)):     
      self.input[net.input[i]] = self.getDevTempTensor(net.input[i], True)

    # create operators
    for l in net.layer:
      l_blob = []

      for idx in xrange(self.weight.num_blobs(str(l.name))):
        l_blob.append(self.weight.get_blob(str(l.name), idx))
      
      if l.type == "Python" and l.python_param.layer == "ProposalLayer":
        self.ops.append(Proposal(l, l_blob, self))
      else:
        self.ops.append(str2class(l.type)(l, l_blob, self))
    print("Device allcation and operator creation are finished")

    

    for act in self.m_act.values():
		  if act.ref_count == 1:
			  self.output[act.name] = act
    	  # activation reservation

	  # TODO: minimize activation memory resouce by layer-wise buffer sharing
	  # each operator should implement alloc_act function 
	  # current version simply assigns all activation seperatly
    #for act in self.m_act.values():
    #  if act.shared is None:
    #    act.buf = buf_region.getBuffer(act.name, act.shape, 
    #      pyCL.mem_flags.READ_WRITE)
    #  else:
		#	  # TODO: implement smart buffer sharing method
		#	  #act.buf = self.m_act[act.shared].buf
    #   pass 








