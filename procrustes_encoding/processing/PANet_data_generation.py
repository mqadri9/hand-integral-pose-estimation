import numpy as np
import argparse
import os
import json

import norm_lite


""" General utility functions """
def parse_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--imgdir', type=str, help='image dataset dir')
	parser.add_argument('--train_split_proportion', type=float, default=0.5, help='picks supervised data proportion')
	parser.add_argument('--fix_split_flag', type=int, default=1, help='splits test data in a fixed manner')
	parser.add_argument('--image_dataset_size', type=int, default=32560, help='total number of images in the dataset')
	parser.add_argument('--num_keypoints', type=int, default=21, help='total number of keypoints/joints in the image')
	parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio for bounding box generation')
	parser.add_argument('--scaling_constant', type=int, default=100, help='scaling constant for z depth estimation')
	parser.add_argument('--image_sample_version', type=str, default=sample_version.gs,
						help='Which sample version to use when showing the training set.'
						' Valid choices are %s' % sample_version.valid_options())
	parser.add_argument('--localdir', type=str, help='local data directory')
	args = parser.parse_args()

	return args

def data_idxs_generation(image_dataset_size, split_ratio, fix_split_flag):
	train_idxs = np.arange(0,int(image_dataset_size*split_ratio))

	if fix_split_flag == 1:
		test_idxs = np.arange(int(image_dataset_size*0.9), image_dataset_size)
	else:
		test_idxs = np.arange(int(image_dataset_size*split_ratio), image_dataset_size)
	return train_idxs, test_idxs

def _assert_exist(p):
	msg = 'File does not exist: %s' % p
	assert os.path.exists(p), msg

def json_load(p):
    _assert_exist(p)
    with open(p, 'r') as fi:
        d = json.load(fi)
    return d

class sample_version:
	gs = 'gs'				### green screen version
	hom = 'hom'				### homogenized version
	sample = 'sample'		### auto colorization with sample points
	auto = 'auto'			### auto colorization without sample points: automatic color hallucination

	# db_size = args.image_dataset_size
	db_size = 32560

	@classmethod
	def valid_options(cls):
		return [cls.gs, cls.hom, cls.sample, cls.auto]

	@classmethod
	def check_valid(cls, version):
		msg = 'Invalid choice: "%s" (must be in %s)' % (version, cls.valid_options())
		assert version in cls.valid_options(), msg

	@classmethod
	def map_id(cls, id, version):
		cls.check_valid(version)
		return id + cls.db_size*cls.valid_options().index(version)

def load_db_annotation(imgdir, fixed_str):

	print("Load dataset indices...")
	# t = time.time()

	### Path to json data containers ###
	k_path = os.path.join(imgdir, '%s_K.json' % fixed_str)
	xyz_path = os.path.join(imgdir, '%s_xyz.json' % fixed_str)

	### Load the corresponding json containers if they exist ###
	K_list = json_load(k_path)
	xyz_list = json_load(xyz_path)

	### Assert the length of the loaded containers ###
	assert len(K_list) == len(xyz_list), 'Size mismatch.'

	# print('Loaded the containers in %.2f seconds' % (time.time() - t))
	print('Loaded the containers')
	return zip(K_list, xyz_list)

def train_gen(train_idxs, args):
	train_xyz_normalized_mat = np.zeros((train_idxs.shape[0], args.num_keypoints, 3))
	for train_idx in train_idxs:
		### Retrieve the image, K, and xyz ###
		train_K, train_xyz = db_data_anno[train_idx]
		train_K, train_xyz = [np.array(x) for x in [train_K, train_xyz]]

		### Retrieve the normalized gt coords ###
		train_xyz_normalized_mat[train_idx,:,:] = norm_lite.generate_joint_cam_normalized(train_xyz, train_K, args.aspect_ratio,
																					   args.num_keypoints, args.scaling_constant)
		print("train idx: ", train_idx)
	# train_xyz_normalized_mat = train_xyz_normalized_mat - train_xyz_normalized_mat.mean(1, keepdims=True)
	np.save(os.path.join(args.localdir, 'hand_train.npy'), train_xyz_normalized_mat)
	print("------------------ Train data is split and ready to use. -----------------")

def test_gen(test_idxs, args):
	test_xyz_normalized_mat = np.zeros((test_idxs.shape[0], args.num_keypoints, 3))
	counter = 0
	for test_idx in test_idxs:
		### Retrieve the image, K, and xyz ###
		test_K, test_xyz = db_data_anno[test_idx]
		test_K, test_xyz = [np.array(x) for x in [test_K, test_xyz]]

		### Retrieve the normalized gt coords ###
		test_xyz_normalized_mat[counter,:,:] = norm_lite.generate_joint_cam_normalized(test_xyz, test_K, args.aspect_ratio,
																					   args.num_keypoints, args.scaling_constant)
		counter = counter + 1
		print("test idx: ", test_idx)

	# test_xyz_normalized_mat = test_xyz_normalized_mat - test_xyz_normalized_mat.mean(1, keepdims=True)
	np.save(os.path.join(args.localdir, 'hand_test.npy'), test_xyz_normalized_mat)
	print("------------------ Test data is split and ready to use. -----------------")


if __name__ == '__main__':
	args = parse_args()
	train_idxs, test_idxs = data_idxs_generation(args.image_dataset_size, args.train_split_proportion, args.fix_split_flag)

	db_data_anno = list(load_db_annotation(args.imgdir, 'training'))

	train_gen(train_idxs, args)
	test_gen(test_idxs, args)

	print("--------------------- Data generation accomplished. ------------------")