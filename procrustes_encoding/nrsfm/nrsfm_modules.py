import torch
import torch.nn as nn
from torch.nn.functional import conv_transpose2d, relu, conv2d

import numpy as np
import sys

from .batch_svd import batch_svd

def relu_threshold(input, thrsh):
	return relu(input + thrsh.view(1, -1, 1, 1))

def block_soft_threshold(input, thrsh):
	"""
	- input: bsize x block_num x block_h, block_w
	- thrsh: block_num
	- print('block soft threshold')
	"""
	bsize = input.shape[0]
	block_num = input.shape[1]
	input_l2_norm = input.view(bsize, block_num, -1).norm(dim=-1)
	return relu(1 - thrsh.view(1, block_num) / input_l2_norm).view(bsize, block_num, 1, 1) * input

def batch_det_3x3(x):
	
	x00 = x[..., 0, 0]
	x01 = x[..., 0, 1]
	x02 = x[..., 0, 2]

	x10 = x[..., 1, 0]
	x11 = x[..., 1, 1]
	x12 = x[..., 1, 2]

	x20 = x[..., 2, 0]
	x21 = x[..., 2, 1]
	x22 = x[..., 2, 2]

	return x00 * x11 * x22 \
		 + x10 * x21 * x02 \
		 + x20 * x12 * x01 \
		 - x02 * x11 * x20 \
		 - x12 * x21 * x00 \
		 - x22 * x10 * x01


def make_orthonormal(input_mat):
	"""
	input_mat: [-1 x 3 x 3]
	return: orthonormalized matrix with same size
	"""
	batch_size = input_mat.size(0)
	mat_col_num = input_mat.size(2)

	u, s, v = batch_svd(input_mat)

	orth_mat = torch.matmul(u, v.transpose(1, 2))

	""" Check reflection """
	if orth_mat.shape[-1] == 3:
		# do flip
		orth_mat_det = batch_det_3x3(orth_mat)
		u_flip = torch.cat((
				u[..., :2],
				u[..., 2:3]*orth_mat_det.view(-1, 1, 1)), 2)
		orth_mat = torch.matmul(u_flip, v.transpose(1,2))

	return orth_mat


class SfMSparseCodingLayer(nn.Module):
	def __init__(self, dict_size, pts_num, share_weight=True, encode_with_relu=1):
		super(SfMSparseCodingLayer, self).__init__()
		self.dict_size = dict_size
		self.pts_num = pts_num
		self.share_weight = share_weight
		self.encode_with_relu = encode_with_relu

		dictionary = torch.empty(pts_num, 3, dict_size)

		nn.init.kaiming_uniform_(dictionary)

		self.dictionary = nn.Parameter(dictionary)

		if not share_weight:
			self.dictionary_decode = nn.Parameter(dictionary.clone())

		
		self.bias_encode_with_cam = nn.Parameter(torch.zeros(dict_size))
		self.bias_decode = nn.Parameter(torch.zeros(pts_num*3))

		# self.encode_with_cam_thresh_func = block_soft_threshold
		if (encode_with_relu == 1):
			self.encode_with_cam_thresh_func = relu_threshold
		else:
			self.encode_with_cam_thresh_func = block_soft_threshold
		

	def encode_with_cam(self, pts):
		"""
		pts: [-1 x pts_num x 3]
		"""
		dictionary = self.dictionary
		conv_weights = dictionary.transpose(1,2).unsqueeze(-1)			# [pts_num x dict_size x 3 x 1]
		block_code = conv_transpose2d(pts.unsqueeze(-2),				# [-1 x pts_num x 1 x 3]
									  conv_weights,
									  stride=1,
									  padding=0)

		return self.encode_with_cam_thresh_func(block_code, self.bias_encode_with_cam)

	def decode(self, code):
		"""
		code: [-1 x dim (x 1)]
		output: [-1 x pts_num x 3]
		"""
		if self.share_weight:
			dictionary = self.dictionary
		else:
			dictionary = self.dictionary_decode

		return conv2d(code.view(-1, self.dict_size, 1 , 1),
					  dictionary.view(-1, self.dict_size, 1, 1),
					  self.bias_decode).view(-1, self.pts_num, 3)


class BlockSparseCodingLayer(nn.Module):
	def __init__(self, dict_size, input_dim, share_weight=True, encode_with_relu=1):
		super(BlockSparseCodingLayer, self).__init__()

		""" dictionary is of size dict_size x input_dim """

		self.dict_size = dict_size
		self.input_dim = input_dim
		self.encode_with_relu = encode_with_relu

		dictionary = torch.empty(input_dim, dict_size, 1, 1)
		nn.init.kaiming_uniform_(dictionary.view(input_dim, dict_size))

		self.dictionary = nn.Parameter(dictionary)
		self.bias_encode_with_cam = nn.Parameter(torch.zeros(dict_size))
		self.bias_decode = nn.Parameter(torch.zeros(input_dim))

		if (encode_with_relu == 1):
			self.encode_with_cam_thresh_func = relu_threshold
		else:
			self.encode_with_cam_thresh_func = block_soft_threshold
		
		self.ae_thresh_func = relu_threshold

		self.share_weight = share_weight
		if not self.share_weight:
			self.dictionary_decode = nn.Parameter(dictionary.clone())

	def encode_with_cam(self, input):
		"""
		input: PoseCodeCalibrateLayer[-1 x input_dim x 3 x 3]
		"""

		block_code = conv_transpose2d(
				input,
				self.dictionary.view(self.input_dim, self.dict_size, 1, 1),
				stride=1,
				padding=0)

		return self.encode_with_cam_thresh_func(block_code, self.bias_encode_with_cam)

	def decode(self, code):
		"""
		code: [-1 x dict_size (x 1)]
		"""
		if self.share_weight:
			dictionary = self.dictionary
		else:
			dictionary = self.dictionary_decode

		output = conv2d(
				code.view(-1, self.dict_size, 1, 1),
				dictionary.view(self.input_dim, self.dict_size, 1, 1),
				padding=0,
				stride=1)

		return self.ae_thresh_func(output, self.bias_decode).view(-1, self.input_dim)

class CameraEstimator(nn.Module):
	def __init__(self, input_chan):
		super(CameraEstimator, self).__init__()
		self.linear_comb_layer = nn.Conv2d(input_chan, 1, kernel_size=1, stride=1, bias=False)

	def forward(self, input):
		camera = self.linear_comb_layer(input).squeeze(1)
		""" Use SVD to make camera matrix be orthonormal """
		return make_orthonormal(camera)

class PoseCodeCalibrateLayer(nn.Module):
	def __init__(self, code_dim, is_2d=True):
		super(PoseCodeCalibrateLayer, self).__init__()
		if is_2d:
			kernel_size = [3, 2]
		else:
			kernel_size = [3, 3]
		self.fc_layer = nn.Conv2d(code_dim, code_dim, kernel_size, stride=1, bias=False)

	def forward(self, input):
		"""
		input: [-1 x code_dim x 3 x 3]
		"""
		return self.fc_layer(input)


















