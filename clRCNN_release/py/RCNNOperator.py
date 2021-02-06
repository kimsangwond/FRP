#-*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from py.Activation import *
from py.Operator import *
import numpy as np
import pyopencl as pyCL

import yaml
from rpn.generate_anchors import generate_anchors
from fast_rcnn.config import cfg
from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
#from fast_rcnn.nms_wrapper import nms

def _filter_boxes(boxes, min_size):
	"""Remove all boxes with any side smaller than min_size."""
	ws = boxes[:, 2] - boxes[:, 0] + 1
	hs = boxes[:, 3] - boxes[:, 1] + 1
	keep = np.where((ws >= min_size) & (hs >= min_size))[0]
	return keep

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def div_up(m,n):
	return m//n + (1 if m%n else 0)
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
threadsPerBlock = np.dtype(np.uint64).itemsize * 8

def py_gpu_nms(dets, thresh, q, ctx, prg):
	boxes_num = dets.shape[0]
	boxes_dim = dets.shape[1]
	keep = np.zeros(boxes_num, dtype = np.int32)
	scores = dets[:,4]
	order = scores.argsort()[::-1]
	sorted_dets = dets[order, :]

	col_blocks = div_up(boxes_num, threadsPerBlock)

	boxes_dev = pyCL.Buffer(ctx, pyCL.mem_flags.READ_WRITE, 
		boxes_num * boxes_dim * np.dtype(np.float32).itemsize)
	mask_dev = pyCL.Buffer(ctx,pyCL.mem_flags.READ_WRITE,
		 boxes_num*col_blocks* np.dtype(np.uint64).itemsize)
	pyCL.enqueue_copy(q, boxes_dev, sorted_dets, is_blocking = False)

	prg.nms_kernel(q, 
		(div_up(boxes_num,threadsPerBlock) * threadsPerBlock,
			 div_up(boxes_num,threadsPerBlock)),
		(threadsPerBlock,1),
		np.int32(boxes_num), np.float32(thresh),
		boxes_dev, mask_dev
	)
	mask_host = np.zeros(boxes_num * col_blocks, dtype = np.uint64)
	pyCL.enqueue_copy(q, mask_host, mask_dev)

	remv = np.zeros(col_blocks, dtype = np.uint64)

	num_out = 0

	for i in xrange(boxes_num):
		nblock = i // threadsPerBlock
		inblock = i % threadsPerBlock

		if np.bitwise_and(remv[nblock] ,  np.uint64(1<<inblock)) == 0:
			keep[num_out] = i
			num_out += 1

			for j in xrange(nblock, col_blocks):
				remv[j] |= mask_host[i*col_blocks + j]

	keep = keep[:num_out]
	return list(order[keep])

class Proposal(Operator):
	def __init__(self, proto, blob, deep_func):
		super(Proposal, self).__init__(proto, deep_func)

		layer_params = yaml.load(proto.python_param.param_str)
		self._feat_stride = layer_params['feat_stride']
		anchor_scales = layer_params.get('scales', (8, 16, 32))
		self._anchors = generate_anchors(scales=np.array(anchor_scales))
		self._num_anchors = self._anchors.shape[0]

		input_act_0 = deep_func.m_act[self.input[0]]
		input_act_0.ref_count += 1
		input_act_1 = deep_func.m_act[self.input[1]]
		input_act_1.ref_count += 1
		input_act_2 = deep_func.m_act[self.input[2]]
		input_act_2.ref_count += 1
		

		output_shape = (1, 5)
				
		if self.output[0] in deep_func.m_act:
			raise("Output %s is pre allocated??"%(self.output[0]))
		deep_func.m_act[self.output[0]] = Activation(self.output[0], output_shape)

		self.temp_buf = []
		self.temp_buf.append(np.zeros(input_act_0.shape, dtype=np.float32))
		self.temp_buf.append(np.zeros(input_act_1.shape, dtype=np.float32))
		self.temp_buf.append(np.zeros(input_act_2.shape, dtype=np.float32))


	def forward(self,q):
		# Algorithm:
		#
		# for each (H, W) location i
		#   generate A anchor boxes centered on cell i
		#   apply predicted bbox deltas at cell i to each of the A anchors
		# clip predicted boxes to image
		# remove predicted boxes with either height or width < threshold
		# sort all (proposal, score) pairs by score from highest to lowest
		# take top pre_nms_topN proposals before NMS
		# apply NMS with threshold 0.7 to remaining proposals
		# take after_nms_topN proposals after NMS
		# return the top proposals (-> RoIs top, scores top)
		input_0 = self.deep_func.m_act[self.input[0]]
		input_1 = self.deep_func.m_act[self.input[1]]
		input_2 = self.deep_func.m_act[self.input[2]]
		output = self.deep_func.m_act[self.output[0]]

		if input_0.shape[0] != 1:
			raise("proposal layer only supports single batch size")

		cfg_key = 'TEST' # either 'TRAIN' or 'TEST'
		pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N
		post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
		nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH
		min_size      = cfg[cfg_key].RPN_MIN_SIZE

		pyCL.enqueue_copy(q, self.temp_buf[0], input_0.buf.getDevBuf(),is_blocking = False)
		pyCL.enqueue_copy(q, self.temp_buf[1], input_1.buf.getDevBuf(),is_blocking = False)
		pyCL.enqueue_copy(q, self.temp_buf[2], input_2.buf.getDevBuf())

		# the first set of _num_anchors channels are bg probs
		# the second set are the fg probs, which we want
		scores = self.temp_buf[0][:, self._num_anchors:, :, :]
		bbox_deltas = self.temp_buf[1]
		im_info = self.temp_buf[2][0, :]

		# 1. Generate proposals from bbox deltas and shifted anchors
		height, width = scores.shape[-2:]

		# Enumerate all shifts
		shift_x = np.arange(0, width) * self._feat_stride
		shift_y = np.arange(0, height) * self._feat_stride
		shift_x, shift_y = np.meshgrid(shift_x, shift_y)
		shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
		shift_x.ravel(), shift_y.ravel())).transpose()

		# Enumerate all shifted anchors:
		#
		# add A anchors (1, A, 4) to
		# cell K shifts (K, 1, 4) to get
		# shift anchors (K, A, 4)
		# reshape to (K*A, 4) shifted anchors
		A = self._num_anchors
		K = shifts.shape[0]
		anchors = self._anchors.reshape((1, A, 4)) + \
                  shifts.reshape((1, K, 4)).transpose((1, 0, 2))
		anchors = anchors.reshape((K * A, 4))

		# Transpose and reshape predicted bbox transformations to get them
		# into the same order as the anchors:
		#
		# bbox deltas will be (1, 4 * A, H, W) format
		# transpose to (1, H, W, 4 * A)
		# reshape to (1 * H * W * A, 4) where rows are ordered by (h, w, a)
		# in slowest to fastest order
		bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

		# Same story for the scores:
		#
		# scores are (1, A, H, W) format
		# transpose to (1, H, W, A)
		# reshape to (1 * H * W * A, 1) where rows are ordered by (h, w, a)
		scores = scores.transpose((0, 2, 3, 1)).reshape((-1, 1))

		# Convert anchors into proposals via bbox transformations
		proposals = bbox_transform_inv(anchors, bbox_deltas)

		# 2. clip predicted boxes to image
		proposals = clip_boxes(proposals, im_info[:2])

		# 3. remove predicted boxes with either height or width < threshold
		# (NOTE: convert min_size to input image scale stored in im_info[2])
		keep = _filter_boxes(proposals, min_size * im_info[2])
		proposals = proposals[keep, :]
		scores = scores[keep]

		# 4. sort all (proposal, score) pairs by score from highest to lowest
		# 5. take top pre_nms_topN (e.g. 6000)
		order = scores.ravel().argsort()[::-1]
		if pre_nms_topN > 0:
				order = order[:pre_nms_topN]
		proposals = proposals[order, :]
		scores = scores[order]

		# 6. apply nms (e.g. threshold = 0.7)
		# 7. take after_nms_topN (e.g. 300)
		# 8. return the top proposals (-> RoIs top)
	
		#keep = nms(np.hstack((proposals, scores)), nms_thresh)
		keep = py_gpu_nms(np.hstack((proposals, scores)), nms_thresh, q, self.ctx, self.prg)

		if post_nms_topN > 0:
				keep = keep[:post_nms_topN]
		proposals = proposals[keep, :]
		scores = scores[keep]

		# Output rois blob
		# Our RPN implementation only supports a single input image, so all
		# batch inds are 0
		batch_inds = np.zeros((proposals.shape[0], 1), dtype=np.float32)
		blob = np.hstack((batch_inds, proposals.astype(np.float32, copy=False)))

		output.shape = blob.shape
		output.buf.shape = blob.shape
		output.buf.dev_buf = pyCL.Buffer(
			self.deep_func.buf_region.buf_mgr.ctx_mgr.ctx, pyCL.mem_flags.READ_WRITE,
			blob.nbytes )
		output.host_buf = blob

		pyCL.enqueue_copy(q, output.buf.getDevBuf(), blob, is_blocking = False)

		#print(('blob',blob.shape))
		pass
		
class ROIPooling(Operator):
	def __init__(self, proto, blob, deep_func):
		super(ROIPooling, self).__init__(proto, deep_func)
		rp = proto.roi_pooling_param

		self.pooled_height = rp.pooled_h
		self.pooled_width = rp.pooled_w
		self.spatial_scale = rp.spatial_scale 

		input_act_0 = deep_func.m_act[self.input[0]]
		input_act_0.ref_count += 1
		input_shape_0 = input_act_0.shape

		self.channel = input_shape_0[1]
		self.height = input_shape_0[2]
		self.width = input_shape_0[3]
		
		input_act_1 = deep_func.m_act[self.input[1]]
		input_act_1.ref_count += 1
		#input_shape_1 = (5,5) #input_act_1.shape
		# reshape 만들면서 고쳐야함
		#print(("input_shape_1:", input_act_1.shape))

##152
		output_shape = (300, self.channel, 
			self.pooled_height, self.pooled_width)

		if self.output[0] in deep_func.m_act:
			raise("Output %s is pre allocated??"%(self.output[0]))
		deep_func.m_act[self.output[0]] = Activation(self.output[0], output_shape)

	def forward(self, q):		
		input_0 = self.deep_func.m_act[self.input[0]]
		input_1 = self.deep_func.m_act[self.input[1]]
		output = self.deep_func.m_act[self.output[0]]
		
		count = reduce(mul, output.shape)

		self.prg.roi_pool_kernel(q, (globalSize(count),), (LOCAL_SIZE,),
			np.int32(count),
			np.float32(self.spatial_scale),			
			np.int32(self.channel), np.int32(self.height), np.int32(self.width),
			np.int32(self.pooled_height), np.int32(self.pooled_width), 
			input_0.buf.getDevBuf(), input_1.buf.getDevBuf(), output.buf.getDevBuf()
		)

			 
			









	


