from train_pytorch.train_kernel import TrainKernel, \
									   add_train_args, \
									   predict_batch_main

from nrsfm.nrsfmnet import PANet

import argparse
import torch
import numpy as np

def parse_args():
	parser = argparse.ArgumentParser()
	add_train_args(parser)		
	# parser.add_argument('--pts_num', type=int, default=21, help='number of keypoints in each sample')
	parser.add_argument('--regressor_output', type=str, help='regressor output npy file')
	args = parser.parse_args()
	return args

def computeMPJPE(pred, gt):
	return (pred - gt).norm(dim=2).mean(-1).mean()

class NRSfM_learner(TrainKernel):

	def __init__(self,
				 pts_num=21,				 
				 code_sparsity_weight=0,
				 encode_with_relu=1):
		super(NRSfM_learner, self).__init__()

		dict_size_list = [512, 256, 128, 64, 32, 16, 8]

		self.code_sparsity_weight = code_sparsity_weight
		self.nrsfm_net = PANet(pts_num=pts_num,
							   dict_size_list=dict_size_list,
							   encode_with_relu=encode_with_relu)
		self.encode_with_relu = encode_with_relu
		self.pts_num = pts_num

	def load_model(self, ckpt_file):
		self.nrsfm_net.load_model(ckpt_file)

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


class NRSfM_tester(NRSfM_learner):

	def forward(self, input):
		pts_recon, pts_recon_canonical, camera_matrix, code = self.predict(input)
		return pts_recon, pts_recon_canonical, camera_matrix, code

def load_dataset_to_gpu(data):
	data_cuda = (
		torch.from_numpy(data['pts3d']).float().cuda(),
		torch.from_numpy(data['mask']).float().cuda())
	return data_cuda

def PANet_reconstruction(regressor_output):
	args = parse_args()
	with torch.no_grad():
		num_samples = regressor_output.shape[0]
		num_pts = regressor_output.shape[1]
		test_data = []
		test_data.append(
			{
				'pts3d': regressor_output - regressor_output.mean(1, keepdims=True),
				'mask': np.ones((num_samples, num_pts))
			})

		test_data = load_dataset_to_gpu(test_data[0])

		pts_num = test_data[0].shape[1]

		test_kernel = NRSfM_tester(pts_num=pts_num)

		pts_recon_ts, pts_recon_canonical_ts, camera_matrix_ts, code_ts = predict_batch_main(test_data, test_kernel, args)
		
		loss_mpjpe = computeMPJPE(pts_recon_ts.to("cuda"), test_data[0])		

		pts_recon_PANet = pts_recon_ts.data.cpu().numpy()

	return pts_recon_PANet, loss_mpjpe

if __name__ == '__main__':
	### Replace the line below with the reconstructed pts from regressor ###
	args = parse_args()
	regressor_output = np.load(args.regressor_output)
	out_pts, loss_mpjpe = PANet_reconstruction(regressor_output)

	print("Shape of output: ", out_pts.shape)
	print("MPJPE loss of pred: ", loss_mpjpe)



















