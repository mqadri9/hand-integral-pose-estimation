import torch
import torch.nn as nn
from torch.nn.functional import conv_transpose2d, relu, conv2d

from .nrsfm_modules import SfMSparseCodingLayer, \
	BlockSparseCodingLayer, \
	CameraEstimator, \
	PoseCodeCalibrateLayer

import numpy as np


class CamInvariantSparseCodingNet(nn.Module):
	def __init__(self, pts_num=21,
				 dict_size_list=list(np.arange(125, 7, -10)),
				 encode_with_relu=1):
		super(CamInvariantSparseCodingNet, self).__init__()
		self.pts_num = pts_num
		self.encode_with_relu = encode_with_relu
		share_weight=True
		self.sparse_coding_layers = nn.ModuleList([
			SfMSparseCodingLayer(dict_size_list[0], pts_num,
								 share_weight, encode_with_relu)])

		for idx in range(1, len(dict_size_list)):
			self.sparse_coding_layers.append(
				BlockSparseCodingLayer(dict_size_list[idx], dict_size_list[idx-1],
									   share_weight, encode_with_relu))
		self.camera_estimator = CameraEstimator(dict_size_list[-1])

	def forward(self, pts):
		""" 
		pts -- num_sample x num_jts x 3
		"""
		pass

	def load_model(self, ckpt_file):
		self.load_state_dict(torch.load(ckpt_file))

	def save_model(self, ckpt_file):
		torch.save(self.state_dict(), ckpt_file)


class PANet(CamInvariantSparseCodingNet):
	
	def __init__(self, pts_num=21,
				 dict_size_list=list(np.arange(125, 7, -10)) + [8], encode_with_relu=1):
		super().__init__(pts_num, dict_size_list, encode_with_relu)
		self.code_estimator = PoseCodeCalibrateLayer(code_dim=dict_size_list[-1], is_2d=False)

	def forward(self, pts_3d):
		"""
		pts_3d: -1xPx3
		"""

		layer_num = len(self.sparse_coding_layers)

		##### 1. Estimate the camera pose #####
		code_block = pts_3d
		for idx in range(layer_num):
			code_block = self.sparse_coding_layers[idx].encode_with_cam(code_block)
		camera_matrix = self.camera_estimator(code_block)
		code = self.code_estimator(code_block)

		##### 2. Run the decoder #####
		for idx in range(layer_num - 1, 0, -1):
			code = self.sparse_coding_layers[idx].decode(code)
		pts_recon_canonical = self.sparse_coding_layers[0].decode(code)

		##### 3. Rotate the reconstructed pts to align with the input (Procrustes) #####
		pts_recon = pts_recon_canonical.matmul(camera_matrix)
		return pts_recon, pts_recon_canonical, camera_matrix, code
