import os
import os.path as osp
import numpy as np
import cv2
import random
import time
import torch
import copy
from torch.utils.data.dataset import Dataset
import pickle as pk

from config import cfg
import sys

class DatasetLoader(Dataset):
    """Create the dataset_loader
    db: a list of datasets (Listed in the config.py trainset list)
    is_train: a flag indicating of this is training data
    """
    def __init__(self, db, is_train, transform):
        
        if isinstance(db, list):
            self.multiple_db = True
            # This will call Freihand.load_data()
            self.db = [d.load_data() for d in db]
            #self.joints_name = [d.joints_name for d in db]
            self.joint_num = [d.joint_num for d in db]
            self.skeleton = [d.skeleton for d in db]
            #self.lr_skeleton = [d.lr_skeleton for d in db]
            #self.flip_pairs = [d.flip_pairs for d in db]
            self.joints_have_depth = [d.joints_have_depth for d in db]
        else:
            self.multiple_db = False
            self.db = db.load_data()
            self.joint_num = d.joint_num
            self.skeleton = db.skeleton
            #self.lr_skeleton = db.lr_skeleton
            #self.flip_pairs = db.flip_pairs
            self.joints_have_depth = db.joints_have_depth
        
        self.transform = transform
        self.is_train = is_train

        if self.is_train:
            self.do_augment = True
        else:
            self.do_augment = False
            
    def __getitem__(self, index):
        
        if self.multiple_db:
            db_idx = index // max([len(db) for db in self.db])
            joint_num = self.joint_num[db_idx]
            skeleton = self.skeleton[db_idx]
            joints_have_depth = self.joints_have_depth[db_idx]
            item_idx = index % max([len(db) for db in self.db]) % len(self.db[db_idx])
            data = copy.deepcopy(self.db[db_idx][item_idx])
            #print(data)
            #print(joints_have_depth)
            #print(self.is_train)
        else:
            joint_num = self.joint_num
            skeleton = self.skeleton
            joints_have_depth = self.joints_have_depth
            data = copy.deepcopy(self.db[index])
        
        bbox = data['bbox']
        joint_img = data['joint_img']
        joint_vis = data['joint_vis']

        # 1. load image
        cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        
        if not isinstance(cvimg, np.ndarray):
            raise IOError("Fail to read %s" % data['img_path'])
        img_height, img_width, img_channels = cvimg.shape
        
        # 2. get augmentation params
        if self.do_augment:
            scale, rot, color_scale = get_aug_config()
        else:
            scale, rot, color_scale = 1.0, 0, [1.0, 1.0, 1.0]

        # 3. crop patch from img and perform data augmentation (scale, rot, color scale)
        img_patch, trans = generate_patch_image(cvimg, bbox, scale, rot)
        for i in range(img_channels):
            img_patch[:, :, i] = np.clip(img_patch[:, :, i] * color_scale[i], 0, 255)

        # 4. generate patch joint ground truth


        for i in range(len(joint_img)):
            joint_img[i, 0:2] = trans_point2d(joint_img[i, 0:2], trans)
            joint_img[i, 2] /= (cfg.bbox_3d_shape[0]/2. * scale) # expect depth lies in -bbox_3d_shape[0]/2 ~ bbox_3d_shape[0]/2 -> -1.0 ~ 1.0
            joint_img[i, 2] = (joint_img[i,2] + 1.0)/2. # 0~1 normalize
            joint_vis[i] *= (
                            (joint_img[i,0] >= 0) & \
                            (joint_img[i,0] < cfg.input_shape[1]) & \
                            (joint_img[i,1] >= 0) & \
                            (joint_img[i,1] < cfg.input_shape[0]) & \
                            (joint_img[i,2] >= 0) & \
                            (joint_img[i,2] < 1)
                            )

        vis = False
        if vis:
            filename = str(random.randrange(1,500))
            tmpimg = img_patch.copy().astype(np.uint8)
            tmpkps = np.zeros((3,joint_num))
            tmpkps[:2,:] = joint_img[:,:2].transpose(1,0)
            tmpkps[2,:] = joint_vis[:,0]
            tmpimg = vis_keypoints(tmpimg, tmpkps, skeleton)
            cv2.imwrite(osp.join(cfg.vis_dir, filename + '_gt.jpg'), tmpimg)
        
        vis = False
        if vis:
            vis_3d_skeleton(joint_img, joint_vis, skeleton, filename)

        # change coordinates to output space
        joint_img[:, 0] = joint_img[:, 0] / cfg.input_shape[1] * cfg.output_shape[1]
        joint_img[:, 1] = joint_img[:, 1] / cfg.input_shape[0] * cfg.output_shape[0]
        joint_img[:, 2] = joint_img[:, 2] * cfg.depth_dim
        
        # change joint coord, vis to reference dataset. 0th db is reference dataset
        if self.multiple_db:
            joint_img = transform_joint_to_other_db(joint_img, joints_name, ref_joints_name)        
            joint_vis = transform_joint_to_other_db(joint_vis, joints_name, ref_joints_name) 
        if self.is_train:
            img_patch = self.transform(img_patch)
            joint_img = joint_img.astype(np.float32)
            # TODO following line need to be done in Freihand            
            joint_vis = (joint_vis > 0).astype(np.float32)
            joints_have_depth = np.array([joints_have_depth]).astype(np.float32)
            return img_patch, joint_img, joint_vis, joints_have_depth
        else:
            img_patch = self.transform(img_patch)
            return img_patch
        
    def __len__(self):
        if self.multiple_db:
            return max([len(db) for db in self.db]) * len(self.db)
        else:
            return len(self.db)        
        
            
# helper functions
def transform_joint_to_other_db(src_joint, src_name, dst_name):

    src_joint_num = len(src_name)
    dst_joint_num = len(dst_name)

    new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]))

    for src_idx in range(len(src_name)):
        name = src_name[src_idx]
        if name in dst_name:
            dst_idx = dst_name.index(name)
            new_joint[dst_idx] = src_joint[src_idx]

    return new_joint

def get_aug_config():
    
    scale_factor = 0.25
    rot_factor = 30
    color_factor = 0.2
    
    scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    rot = np.clip(np.random.randn(), -2.0,
                  2.0) * rot_factor if random.random() <= 0.6 else 0
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]

    return scale, rot, color_scale


def generate_patch_image(cvimg, bbox, scale, rot):
    print("======== INSIDE generate_patch_image=================")
    print(bbox)
    print(scale)
    print(rot)
    img = cvimg.copy()
    print(img.shape)

    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0])
    bb_c_y = float(bbox[1])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])
    
    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, cfg.input_shape[1], cfg.input_shape[0], scale, rot, inv=False)
    img_patch = cv2.warpAffine(img, trans, (int(cfg.input_shape[1]), int(cfg.input_shape[0])), flags=cv2.INTER_LINEAR)

    nn = str(random.random())
    nn = img_path.split('/')[-1]
    cv2.imwrite('/home/mqadri/integral-human-pose/pytorch_projects/hm36_challenge/mytests/{}'.format(nn), cvimg)
    
    img_patch = img_patch[:,:,::-1].copy()
    img_patch = img_patch.astype(np.float32)

    return img_patch, trans

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
    """
    Input is: 
    c_x: bb center x
    c_y: bb center y
    src_width: bb_width
    src_height: bb_height
    dst_width: cfg.input_shape[1]
    dst_height: cfg.input_shape[0]
    scale/rot
    inv: 
        True: find a transformation from destination to source
    """
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    src_center = np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]           
             
            
            
        