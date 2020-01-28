from train_pytorch.train_kernel import TrainKernel, \
									   add_train_args, \
									   train_batch_main, \
									   predict_batch_main
from nrsfm.nrsfmnet import PANet

import argparse
import torch
import numpy as np
import pickle
import os
import copy

def frobenius_norm_loss(input_1, input_2):
	batch_size = input_1.shape[0]
	d = (input_1 - input_2).view(batch_size, -1)
	return d.norm(dim=-1).mean()

def computeMPJPE(pred, gt):
	return (pred - gt).norm(dim=2).mean(-1).mean()

class NRSfM_learner(TrainKernel):

	def __init__(self,
			pts_num=18,
			ae_config=0,
			code_sparsity_weight=0,
			encode_with_relu=1):
		super(NRSfM_learner, self).__init__()

		if ae_config == 0:
			dict_size_list = [512, 256, 128, 64, 32, 16, 8]
		elif ae_config == 1:
			dict_size_list = [256, 128, 64, 32, 16, 8]
		elif ae_config == 2:
			dict_size_list = [125, 115, 104, 94, 83, 73, 62, 52, 41, 31, 20, 10]
		elif ae_config == 3:
			# dict_size_list = [128, 64, 32, 16, 4, 2]					# Old
			dict_size_list = [1024, 512, 256, 128, 64, 32, 16, 8]		# New
		elif ae_config == 4:
			# dict_size_list = [128, 100, 64, 50, 32, 16, 8, 4]			# Old
			dict_size_list = [128, 100, 64, 50, 32, 16, 8]				# New
		elif ae_config == 5:
			# dict_size_list = [128, 100, 64, 50, 32, 16, 8, 4, 2]		# Old
			dict_size_list = [128, 100, 64, 50, 32, 16, 8, 6]			# New
		elif ae_config == 6:
			# dict_size_list = [38, 128, 100, 64, 50, 32, 16, 8, 4, 2]	# Old
			dict_size_list = [38, 128, 100, 64, 50, 32, 16, 8, 6]		# New
		elif ae_config == 7:
			dict_size_list = [2048, 1024, 512, 256, 128, 115, 104, 94, 83, 73, 62, 52, 41, 31, 20, 10, 16, 8]
		elif ae_config == 8:
			dict_size_list = [4096, 2048, 1024, 512, 256, 128, 64, 32, 16, 8]

		self.code_sparsity_weight = code_sparsity_weight
		self.nrsfm_net = PANet(pts_num = pts_num,
							   dict_size_list = dict_size_list,
							   encode_with_relu = encode_with_relu)
		self.encode_with_relu = encode_with_relu
		self.pts_num = pts_num


	def load_model(self, ckpt_file):
		self.nrsfm_net.load_model(ckpt_file)

	def save_model(self, ckpt_file):
		self.nrsfm_net.save_model(ckpt_file)

	def get_parameters(self):
		return self.nrsfm_net.parameters()

	def init_weights(self):
		pass


	def _predict(self, pts_3d):
		pts_recon, pts_recon_canonical, camera_matrix, code = self.nrsfm_net(pts_3d)
		return pts_recon, pts_recon_canonical, camera_matrix, code

	def predict(self, input):
		pts_3d, mask = input[:5]
		return self._predict(pts_3d)

	def forward(self, input, do_log_im, do_log_scalar, return_per_sample_loss=False):
		pts_3d, mask = input[:2]
		pts_recon, pts_recon_canonical, camera_matrix, code = self._predict(pts_3d)

		loss_sparsity = code.abs().sum(-1).mean()
		loss_recon = frobenius_norm_loss(pts_recon, pts_3d)

		loss_mpjpe = computeMPJPE(pts_recon, pts_3d)		

		loss = loss_recon + self.code_sparsity_weight * loss_sparsity

		log_dict = {}
		if do_log_scalar:
			log_scalar = dict(
				mpjpe = loss_mpjpe.data.cpu().numpy(),				
				error = loss_recon.data.cpu().numpy(),
				loss = loss.data.cpu().numpy(),
				loss_recon = loss_recon.data.cpu().numpy(),
				loss_sparsity = loss_sparsity.data.cpu().numpy())
			log_dict.update(log_scalar)

		return loss, log_dict


class NRSfM_tester(NRSfM_learner):

	def forward(self, input):
		pts_recon, pts_recon_canonical, camera_matrix, code = self.predict(input)		
		return pts_recon, pts_recon_canonical, camera_matrix, code		

def load_hand_data(localdir):
	f_train = os.path.join(localdir, 'hand_train.npy')
	f_test = os.path.join(localdir, 'hand_test.npy')
	data_file_list = [f_train, f_test]
	data_list = []
	for file in data_file_list:
		pts3d = np.load(file)
		num_samples = pts3d.shape[0]
		num_pts = pts3d.shape[1]
		data_list.append(
			{
				'pts3d': pts3d - pts3d.mean(1, keepdims=True),
				'mask': np.ones((num_samples, num_pts))
			}
		)
	return data_list[0], data_list[1]

def load_dataset_to_gpu(data):
	data_cuda = (
		torch.from_numpy(data['pts3d']).float().cuda(),
		torch.from_numpy(data['mask']).float().cuda())
	return data_cuda

def load_dataset(dataset_id, localdir):
	if dataset_id == 'hand':
		train_data, validation_data = load_hand_data(localdir)

	train_data_cuda = load_dataset_to_gpu(train_data)
	validation_data_cupa = load_dataset_to_gpu(validation_data)
	return train_data_cuda, validation_data_cupa

def load_test_dataset(dataset_id, localdir):
	if dataset_id.startswith('hand'):
		train_data, validation_data = load_hand_data(localdir)

		if dataset_id == 'hand_train':
			test_data = train_data
		else:
			test_data = validation_data
	test_data_cuda = load_dataset_to_gpu(test_data)
	return test_data_cuda

def parse_args():
	parser = argparse.ArgumentParser()
	add_train_args(parser)
	parser.add_argument('--mode', type=str, default='eval', help='mode: train, eval(default), mining')
	parser.add_argument('--dataset_id', type=str, help='picks dataset')
	parser.add_argument('--weight', type=float, default=0, help='code sparsity loss weight')
	parser.add_argument('--ae_config', type=int, default=0, help='dictionary configuration')
	parser.add_argument('--pts_num', type=int, default=21, help='number of keypoints for each sample')
	parser.add_argument('--comp_num', type=int, default=1, help='number of component if to learn a composition of models')
	parser.add_argument('--localdir', type=str, help='local data directory')

	args = parser.parse_args()

	return args

def train(args):
	""" Create training data on CUDA """	
	train_data, validation_data = load_dataset(args.dataset_id, args.localdir)
	pts_num = train_data[0].shape[1]	
	train_kernel = NRSfM_learner(
		code_sparsity_weight = args.weight,
		pts_num = pts_num,
		ae_config = args.ae_config,
		encode_with_relu = args.encode_with_relu)
	train_batch_main(train_data, train_kernel, validation_data, args)

def test(args):
	with torch.no_grad():
		test_data =load_test_dataset(args.dataset_id, args.localdir)
		pts_num = test_data[0].shape[1]

		test_kernel = NRSfM_tester(pts_num=pts_num,
								   ae_config=args.ae_config,
								   encode_with_relu=args.encode_with_relu)

		pts_recon_ts, pts_recon_canonical_ts, camera_matrix_ts, code_ts = \
			predict_batch_main(test_data, test_kernel, args)

		pts_recon = pts_recon_ts.data.cpu().numpy()

	output_file = os.path.join(args.logdir, 'pred_%s.npz'%args.dataset_id)
	np.savez(output_file, pts=pts_recon)

if __name__ == '__main__':
	args = parse_args()
	if args.mode == 'train':
		train(args)
	elif args.mode == 'test':
		test(args)






















