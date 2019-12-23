import os
import numpy as np
import cv2
import random
import time
import torch
import copy
from torch.utils.data.dataset import Dataset
from FreiHand import FreiHand
import pickle as pk
import matplotlib.pyplot as plt

from config import cfg
import sys
import augment
plt.switch_backend('agg')
class DatasetLoader(Dataset):
    """Create the dataset_loader
    db: a list of datasets (Listed in the config.py trainset list)
    is_train: a flag indicating of this is training data
    """
    def __init__(self, db, is_train, transform, main_loop=True):
        
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
        self.main_loop = main_loop
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
        K = data['K']
        
        R = augment.sample_rotation_matrix()
        #print("Rotation matrix is")
        #print(R)
        #sys.exit()
        # 1. load image
        
        cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if not self.main_loop:
            return cvimg
        #print("joint_img")
        #print(joint_img)
        #sys.exit()
        if not isinstance(cvimg, np.ndarray):
            raise IOError("Fail to read %s" % data['img_path'])
        img_height, img_width, img_channels = cvimg.shape
        
        # 2. get augmentation params
        if self.do_augment:
            scale, rot, color_scale = augment.get_aug_config()
            #scale, rot, color_scale = 1.0, 0, [1.0, 1.0, 1.0]
        else:
            scale, rot, color_scale = 1.0, 0, [1.0, 1.0, 1.0]
        
        
        #homo = K.dot(R).dot(np.linalg.inv(K))
         
        #img2_w = cv2.warpPerspective(cvimg, homo, (cvimg.shape[1], cvimg.shape[0]))
        
        #nn = str(random.randint(1001,2000))
        #nn2 = str(random.randint(1,100))
        #print("=================================================")
        #print(nn)
        #print(data['img_path'])
        #print("==================================================")
        #joint_img2 = np.zeros(joint_img.shape)
        joint_img2 = np.copy(joint_img)
        #for i in range(len(joint_img)):
        #    joint_img2[i, 0:2] = augment.trans_point2d(joint_img[i, 0:2], homo)
        #fig = plt.figure()
        #ax1 = fig.add_subplot(121)
        #ax2 = fig.add_subplot(122)
        #ax1.imshow(cvimg)
        #ax2.imshow(img2_w)
        #print("-------------------------------------------------------")
        #print(data['img_path'])
        #FreiHand.plot_hand(ax1, joint_img[:, 0:2], order='uv')
        #FreiHand.plot_hand(ax2, joint_img2[:, 0:2], order='uv')
        #ax1.axis('off')
        #ax2.axis('off')
        #cv2.imwrite('/home/mqadri/hand-integral-pose-estimation/tests/{}.jpg'.format(nn), img2_w)
        #cv2.imwrite('/home/mqadri/hand-integral-pose-estimation/tests/{}.jpg'.format(nn2), cvimg)
        #plt.savefig('/home/mqadri/hand-integral-pose-estimation/tests/{}.jpg'.format(nn))
        #sys.exit()
        # 3. crop patch from img and perform data augmentation (scale, rot, color scale)
        img_patch, trans, joint_img = augment.generate_patch_image2(cvimg, data["joint_cam"], scale, rot, K)
        # 4. generate patch joint ground truth
        # color 
        # random noise 

        for i in range(len(joint_img)):
            joint_img[i, 0:2] = augment.trans_point2d(joint_img[i, 0:2], trans)
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
        #print("---------------------------")
        #print(cvimg)
        #print(img_patch)
        #print("---------------------------")

        
        #joint_img2 = np.zeros(joint_img.shape)
        #joint_img2 = np.copy(joint_img)
        fig = plt.figure()
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        
        ax1.imshow((255*img_patch/np.max(img_patch)).astype(np.uint8))
        ax2.imshow(cvimg)
        #ax1.imshow(img2_w)
        
        FreiHand.plot_hand(ax1, joint_img[:, 0:2], order='uv')
        FreiHand.plot_hand(ax2, joint_img2[:, 0:2], order='uv')
        ax1.axis('off')
        nn = str(random.randint(2001,3000))
        plt.savefig('/home/mqadri/hand-integral-pose-estimation/tests/{}.jpg'.format(nn))
        #print("hehehehehehe")
        sys.exit()
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
            # TODO Right now transform_joint_to_other_db just returns the inputs
            # If implementing multiple datasources in the future, this method needs to be implemented
            joints_name = None
            ref_joints_name = None
            joint_img = augment.transform_joint_to_other_db(joint_img, joints_name, ref_joints_name)        
            joint_vis = augment.transform_joint_to_other_db(joint_vis, joints_name, ref_joints_name) 
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
    