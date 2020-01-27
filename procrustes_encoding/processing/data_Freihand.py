import numpy as np
import os
import json

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
	# print("Load dataset indices...")
	### Path to json data containers ###
	k_path = os.path.join(imgdir, '%s_K.json' % fixed_str)
	xyz_path = os.path.join(imgdir, '%s_xyz.json' % fixed_str)

	### Load the corresponding json containers if they exist ###
	K_list = json_load(k_path)
	xyz_list = json_load(xyz_path)

	### Assert the length of the loaded containers ###
	assert len(K_list) == len(xyz_list), 'Size mismatch.'

	print('Loaded the Freihand containers')

	return zip(K_list, xyz_list)

def generate_data(args, idxs, db_data_anno):
	xyz_mat = np.zeros((idxs.shape[0], args.num_keypoints, 3))
	K_mat = np.zeros((idxs.shape[0], 3, 3))
	counter = 0
	for idx in idxs:
		K, xyz = db_data_anno[idx]
		K_mat[counter,:,:], xyz_mat[counter,:,:] = [np.array(x) for x in [K, xyz]]
		counter = counter + 1

	return K_mat, xyz_mat









