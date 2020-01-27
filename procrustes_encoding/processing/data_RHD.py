from __future__ import print_function, unicode_literals

import pickle
import os
import numpy as np
import imageio


def generate_annotations(args, set_type):
	image_path = os.path.join(args.dataset_dir, set_type, 'color')
	annotations_path = os.path.join(args.dataset_dir, set_type, 'anno_%s.pickle' % set_type)
	with open(annotations_path, 'rb') as fi:
		anno_all = pickle.load(fi)

	return anno_all

def generate_data(args, idxs, full_idxs, full_values):

	xyz_mat = np.zeros((0, args.num_keypoints, 3))
	K_mat = np.zeros((0, 3, 3))
	counter = 0
	for idx in idxs:
		sample_id = full_idxs[idx]
		anno = full_values[sample_id]
		kp_visible = (anno['uv_vis'][:, 2] == 1)
		kp_coord_xyz = anno['xyz']
		kp_coord_xyz = (kp_coord_xyz[kp_visible, 0], kp_coord_xyz[kp_visible, 1], kp_coord_xyz[kp_visible, 2])
		kp_coord_xyz = np.asarray(kp_coord_xyz).T
		camera_intrinsic_matrix = anno['K']

		if (np.all(kp_visible)):
			xyz_mat = np.append(xyz_mat, kp_coord_xyz[0:args.num_keypoints,:][None,:], axis=0)
			xyz_mat = np.append(xyz_mat, kp_coord_xyz[args.num_keypoints:args.num_keypoints+args.num_keypoints,:][None,:], axis=0)
			K_mat = np.append(K_mat, camera_intrinsic_matrix[None,:], axis=0)
			K_mat = np.append(K_mat, camera_intrinsic_matrix[None,:], axis=0)
		else:
			if kp_coord_xyz.shape[0] % args.num_keypoints != 0:
				continue
			xyz_mat = np.append(xyz_mat, kp_coord_xyz[None,:], axis=0)
			K_mat = np.append(K_mat, camera_intrinsic_matrix[None,:], axis=0)

	return K_mat, xyz_mat

# train_idxs = np.array(list(anno_all.keys()))
# values = np.array(list(anno_all.values()))


# xyz_coords_final = np.zeros((0,21,3))
# K_final = np.zeros((0,3,3))
# # return_image = 0

# for idx in range(train_idxs.shape[0]):

# 	# idx = 0						# 40 -- Single Hand, 400 -- Double Hands
# 	sample_id = train_idxs[idx]
# 	anno = values[sample_id]

# 	kp_visible = (anno['uv_vis'][:, 2] == 1)
# 	kp_coord_xyz = anno['xyz']
# 	kp_coord_xyz = (kp_coord_xyz[kp_visible, 0], kp_coord_xyz[kp_visible, 1], kp_coord_xyz[kp_visible, 2])
# 	kp_coord_xyz = np.asarray(kp_coord_xyz).T
# 	camera_intrinsic_matrix = anno['K']


# 	if (np.all(kp_visible)):
# 		xyz_coords_final = np.append(xyz_coords_final, kp_coord_xyz[0:21,:][None,:], axis=0)
# 		xyz_coords_final = np.append(xyz_coords_final, kp_coord_xyz[21:42,:][None,:], axis=0)
# 		K_final = np.append(K_final, camera_intrinsic_matrix[None,:], axis=0)
# 		K_final = np.append(K_final, camera_intrinsic_matrix[None,:], axis=0)
# 	else:
# 		if kp_coord_xyz.shape[0] % 21 != 0:
# 			continue
# 		xyz_coords_final = np.append(xyz_coords_final, kp_coord_xyz[None,:], axis=0)
# 		K_final = np.append(K_final, camera_intrinsic_matrix[None,:], axis=0)

# 	print("RHD Index: ", idx)


# print(K_final.shape)
# print(xyz_coords_final.shape)





