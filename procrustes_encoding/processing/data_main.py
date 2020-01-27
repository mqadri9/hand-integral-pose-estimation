import numpy as np
import argparse
import os

import data_Freihand
import data_RHD
import norm_lite


""" General utility functions """
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--dataset', type=str, help='dataset type')
	parser.add_argument('--dataset_dir', type=str, help='dataset dir')
	# parser.add_argument('--dataset_size', type=int, help='dataset size')
	parser.add_argument('--datasplit_ratio', type=float, default=0.5, help='picks supervised data proportion')
	parser.add_argument('--fix_split_flag', type=int, default=1, help='splits test data in a fixed manner')
	parser.add_argument('--num_keypoints', type=int, default=21, help='total number of keypoints/joints in the image')
	parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio for bounding box generation')
	parser.add_argument('--scaling_constant', type=int, default=100, help='scaling constant for z depth estimation')
	parser.add_argument('--Frei_sample_version', type=str, default='gs',
						help='Which sample version to use when showing the training set.')
	parser.add_argument('--localdir', type=str, help='local data directory')
	args = parser.parse_args()
	return args

def data_idxs_generation(dataset_size, split_ratio, fix_split_flag):
	train_idxs = np.arange(0,int(dataset_size*split_ratio))

	if fix_split_flag == 1:
		test_idxs = np.arange(int(dataset_size*0.9), dataset_size)
	else:
		test_idxs = np.arange(int(dataset_size*split_ratio), dataset_size)
	return train_idxs, test_idxs

def normalized_gen(K, xyz, args, strType):
	xyz_normalized_mat = np.zeros((xyz.shape[0], args.num_keypoints, 3))
	for idx in range(xyz.shape[0]):
		xyz_normalized_mat[idx,:,:] = norm_lite.generate_joint_cam_normalized(xyz[idx,:,:], K[idx,:,:],
																			  args.aspect_ratio, args.num_keypoints,
																			  args.scaling_constant)
		# print("Norm %s idx: " % strType, idx)

	if strType == 'train':
		np.save(os.path.join(args.localdir, 'train.npy'), xyz_normalized_mat)
	else:
		np.save(os.path.join(args.localdir, 'test.npy'), xyz_normalized_mat)

if __name__ == '__main__':
	args = parse_args()

	if args.dataset == 'Freihand':
		db_data_anno = list(data_Freihand.load_db_annotation(args.dataset_dir, 'training'))
		train_idxs, test_idxs = data_idxs_generation(len(db_data_anno), args.datasplit_ratio, args.fix_split_flag)

		train_K, train_xyz = data_Freihand.generate_data(args, train_idxs, db_data_anno)
		test_K, test_xyz = data_Freihand.generate_data(args, test_idxs, db_data_anno)
		print("Freihand internal data is generated.")

	elif args.dataset == 'RHD':
		db_data_anno = data_RHD.generate_annotations(args, 'training')
		full_idxs, full_values = np.array(list(db_data_anno.keys())), np.array(list(db_data_anno.values()))
		train_idxs, test_idxs = data_idxs_generation(len(db_data_anno), args.datasplit_ratio, args.fix_split_flag)

		train_K, train_xyz = data_RHD.generate_data(args, train_idxs, full_idxs, full_values)
		test_K, test_xyz = data_RHD.generate_data(args, test_idxs, full_idxs, full_values)
		print("RHD internal data is generated.")

	elif args.dataset == 'LSMV':
		xyz_fname = os.path.join(args.dataset_dir, 'xyz_world_LSMV.npy')
		xyz_LSMV = np.load(xyz_fname)
		xyz_LSMV = xyz_LSMV - xyz_LSMV.mean(1, keepdims=True)

		K_fname = os.path.join(args.dataset_dir, 'K_LSMV.npy')
		K_LSMV = np.load(K_fname)
		train_idxs, test_idxs = data_idxs_generation(xyz_LSMV.shape[0], args.datasplit_ratio, args.fix_split_flag)

		train_xyz, train_K = xyz_LSMV[train_idxs,:,:], K_LSMV[train_idxs,:,:]
		test_xyz, test_K = xyz_LSMV[test_idxs,:,:], K_LSMV[test_idxs,:,:]

	elif args.dataset == 'HM36':
		print("Under development...")


	if args.dataset != 'LSMV':
		normalized_gen(train_K, train_xyz, args, 'train')
		normalized_gen(test_K, test_xyz, args, 'test')
	else:
		print("Warning: LSMV's normalization is yet to be done")
		np.save(os.path.join(args.localdir, 'train.npy'), train_xyz)
		np.save(os.path.join(args.localdir, 'test.npy'), test_xyz)
	print("------------------ Data is split and ready to use. ------------------")



















