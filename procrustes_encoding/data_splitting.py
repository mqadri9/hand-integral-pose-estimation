import numpy as np
import argparse
import os

def parse_args():
	parser = argparse.ArgumentParser()	
	parser.add_argument('--datadir', type=str, help='root dataset dir')
	parser.add_argument('--train_split_proportion', type=float, default=0.8, help='picks supervised data proportion')
	parser.add_argument('--fix_split_flag', type=int, default=1, help='splits test data in a fixed manner')
	parser.add_argument('--localdir', type=str, help='local data directory')
	args = parser.parse_args()

	return args

def load_data(args):
	data = np.load(args.datadir)
	return data

def split_data(root_data, split_ratio, fix_split_flag):
	train_idxs = int(root_data.shape[0]*split_ratio)
	train_data = root_data[0:train_idxs, :, :]	

	if fix_split_flag == 1:		
		test_data = root_data[int(root_data.shape[0]*0.9):root_data.shape[0], :, :]
	else:		
		test_data = root_data[train_idxs:root_data.shape[0], :, :]
	return train_data, test_data


if __name__ == '__main__':
	args = parse_args()
	root_data = load_data(args)
	train_data, test_data = split_data(root_data, args.train_split_proportion, args.fix_split_flag)
	np.save(os.path.join(args.localdir, 'hand_train.npy'), train_data)
	np.save(os.path.join(args.localdir, 'hand_test.npy'), test_data)
	print("------------------ Data is split and ready to use! -----------------")

