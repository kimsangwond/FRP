from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from operator import mul
class Activation():
	def __init__(self, name, shape):
		self.name = name
		self.shape = shape
		self.ref_count = 0
		self.shared = None
		self.buf = None
		self.host = False

		
	def __str__(self):
		return self.name + self.shape.__str__()
	def __repr__(self):
		return self.__str__()

	def size(self):
		return reduce(mul, self.shape)



		

		