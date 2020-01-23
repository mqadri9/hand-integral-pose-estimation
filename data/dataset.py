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
import scipy.optimize

from config import cfg
import sys
import augment

from FreiHand_config import FreiHandConfig
from hand_detector import HandDetector

plt.switch_backend('agg')

class DatasetLoader(Dataset):
    """Create the dataset_loader
    db: a list of datasets (Listed in the config.py trainset list)
    is_train: a flag indicating of this is training data
    """
    def __init__(self, db, is_train, transform, main_loop=True, is_eval=False):
        
        if isinstance(db, list):
            self.multiple_db = True
            # This will call Freihand.load_data()
            if not is_eval:
                self.db = [d.load_data() for d in db]
            else:
                self.db = [d.load_evaluation_data() for d in db]
            #self.joints_name = [d.joints_name for d in db]
            self.joint_num = [d.joint_num for d in db]
            self.skeleton = [d.skeleton for d in db]
            #self.lr_skeleton = [d.lr_skeleton for d in db]
            #self.flip_pairs = [d.flip_pairs for d in db]
            self.joints_have_depth = [d.joints_have_depth for d in db]
        else:
            self.multiple_db = False
            if not is_eval:
                self.db = db.load_data()
            else:
                self.db = db.load_evaluation_data()
            self.joint_num = db.joint_num
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
        self.is_eval = is_eval
        if is_eval or cfg.use_hand_detector:
            self.hand_detector = HandDetector(cfg.checksession, cfg.checkepoch, cfg.checkpoint, cuda=True, thresh=0.001)
            self.hand_detector.load_faster_rcnn_detector() 
        
    def __getitem__(self, index):
        if self.multiple_db:
            db_idx = index // max([len(db) for db in self.db])
            joint_num = self.joint_num[db_idx]
            skeleton = self.skeleton[db_idx]
            joints_have_depth = self.joints_have_depth[db_idx]
            item_idx = index % max([len(db) for db in self.db]) % len(self.db[db_idx])
            data = copy.deepcopy(self.db[db_idx][item_idx])
        else:
            joint_num = self.joint_num
            skeleton = self.skeleton
            joints_have_depth = self.joints_have_depth
            data = copy.deepcopy(self.db[index])
          
        if not self.is_eval:
            K = data['K']
            joint_cam = data["joint_cam"]
            faster_rcnn_bbox = data['faster_rccn_bbox']
            # 1. load image
            cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            if not self.main_loop:
                return cvimg
            if not isinstance(cvimg, np.ndarray):
                raise IOError("Fail to read %s" % data['img_path'])
            img_height, img_width, img_channels = cvimg.shape
            
            # 2. get augmentation params
            #self.do_augment = True
            if self.do_augment:
                scale, R, color_scale = augment.get_aug_config()
                #scale, rot, color_scale = 1.0, 0, [1.0, 1.0, 1.0]
            else:
                scale, R, color_scale = 1.0, np.eye(3), [1.0, 1.0, 1.0]
            if cfg.use_hand_detector:
                img_patch, trans, joint_img, joint_img_orig, joint_cam_normalized, joint_vis, xyz_rot, bbox, tprime = augment.generate_patch_image(cvimg, joint_cam, scale, R, K, inv=False, 
                                                                                                                                             hand_detector=self.hand_detector, 
                                                                                                                                             img_path=data['img_path'],
                                                                                                                                             faster_rcnn_bbox=faster_rcnn_bbox)
            else:
                img_patch, trans, joint_img, joint_img_orig, joint_cam_normalized, joint_vis, xyz_rot, bbox, tprime = augment.generate_patch_image(cvimg, joint_cam, scale, R, K, inv=False)
            
            # 4. generate patch joint ground truth
            
            for n_jt in range(len(joint_img)):
                joint_img[n_jt, 0:2] = augment.trans_point2d(joint_img[n_jt, 0:2], trans)
            
              
            params = {
                "R": R,
                "cvimg": cvimg,
                "K": K,
                "joint_cam": joint_cam,
                "scale": scale,
                "img_path": data['img_path'],
                "tprime": tprime,
                "bbox": bbox,
                "trans": trans,
                "joint_cam_normalized": joint_cam_normalized,
                "joint_img_orig": joint_img_orig,
                "ref_bone_len": data["ref_bone_len"]
            }
            #===================================================================
            # fig = plt.figure()
            #          
            # ax1 = fig.add_subplot(121)
            # ax2 = fig.add_subplot(122)
            # # 
            # ax1.imshow((255*img_patch/np.max(img_patch)).astype(np.uint8))
            # ax2.imshow(cvimg)
            # #ax1.imshow(img2_w)
            # # 
            # FreiHand.plot_hand(ax1, joint_img[:, 0:2], order='uv')
            # FreiHand.plot_hand(ax2, joint_img_orig[:, 0:2], order='uv')
            # ax1.axis('off')
            # nn = str(random.randint(1,3000))
            # #print("=============================================================")
            # #print(nn)
            # plt.savefig('/home/mqadri/hand-integral-pose-estimation/tests/{}.jpg'.format(nn)) 
            #===================================================================
            
            img_patch = self.transform(img_patch)
            # apply normalization
            for n_c in range(img_channels):
                img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
            
            #for n_jt in range(len(joint_img)):
            #    zoom_factor = max(bbox[3], bbox[2])
                #joint_img[n_jt, 2] = (joint_img[n_jt, 2] * f * zoom_factor) / (z_mean * cfg.patch_width)
            #    joint_img[n_jt, 2] = joint_img[n_jt, 2] / (zoom_factor * scale) * cfg.patch_width
                #joint_img[n_jt, 2] = joint_img[n_jt, 2] / (cfg.bbox_3d_shape[0] * scale) * cfg.patch_width
            label, label_weight = augment.generate_joint_location_label(cfg.patch_width, cfg.patch_height, joint_img, joint_vis)
            if self.is_train:
                return img_patch, label, label_weight, params
            else:
                return img_patch, label, label_weight, params
        else:
            cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            #nn = str(random.randint(4000,5000))
            #cv2.imwrite('/home/mqadri/hand-integral-pose-estimation/tests/{}.jpg'.format(nn), cv2.cvtColor(cvimg, cv2.COLOR_RGB2BGR))
            #try:
            if cfg.online_hand_detection:
                bbox = augment.find_bb_hand_detector(data['img_path'])
            else:
                bbox = data["faster_rccn_bbox"]
            center_x = bbox[0]
            center_y = bbox[1]
            bb_width = bbox[2]
            bb_height = bbox[3]
            trans = augment.gen_trans_from_patch_cv(center_x, center_y, bb_width, bb_height, cfg.input_shape[1], cfg.input_shape[0], 1.0, inv = False)
            img_patch = cv2.warpPerspective(cvimg, trans, (int(cfg.input_shape[1]), int(cfg.input_shape[0])), flags=cv2.INTER_LINEAR)
            #print("img path before transformation")
            #nn = str(random.randint(0,1000))
            #print(nn)
            # Swap first and last columns # BGR to RGB
            img_patch = img_patch[:,:,::-1].copy()
            img_patch = img_patch.astype(np.float32)
            L = max(bbox[2], bbox[3])
            K = data["K"]
            if L == bbox[2]:
                # multiply by a 100 to increase the value ranges of joint_cam_normalized at line 375
                # so basically scale the hand to be a constant length of 100 instead of 1
                tprime = cfg.scaling_constant * K[0, 0] / L
            else: 
                tprime = cfg.scaling_constant * K[1, 1] / L
            params = {
                "K": data["K"],
                "ref_bone_len": data["ref_bone_len"],
                "img_path": data["img_path"],
                "bbox": np.array([center_x, center_y, bb_width, bb_height]),
                "tprime": tprime
            }
            
            #===================================================================
            #===================================================================
            # fig = plt.figure()        
            # ax1 = fig.add_subplot(121)
            # ax1.imshow((255*img_patch/np.max(img_patch)).astype(np.uint8))
            # ax1.axis('off')
            # nn = str(random.randint(1,3000))
            # plt.savefig('/home/mqadri/hand-integral-pose-estimation/tests/{}.jpg'.format(nn))
            #===================================================================
            #===================================================================
            
            img_patch = self.transform(img_patch)
            return img_patch, params
        
    def __len__(self):
        if self.multiple_db:
            return max([len(db) for db in self.db]) * len(self.db)
        else:
            return len(self.db)
    
def F(alpha, X, U):
    s = [np.sqrt((alpha*X[i, 0] - U[i,0])**2 +  (alpha*X[i, 1] - U[i,1])**2) for i in range(FreiHandConfig.num_joints)]
    sol = np.sum(s)
    return sol





        #augmentation["joint_img2"] = np.copy(joint_img)
        #=======================================================================
        # if "../data/FreiHand/training/rgb/00030151.jpg" == data['img_path']:
        #     fig = plt.figure()
        #            
        #     ax1 = fig.add_subplot(121)
        #     ax2 = fig.add_subplot(122)
        #     # 
        #     ax1.imshow((255*img_patch/np.max(img_patch)).astype(np.uint8))
        #     ax2.imshow(cvimg)
        #     #ax1.imshow(img2_w)
        #     # 
        #     FreiHand.plot_hand(ax1, joint_img[:, 0:2], order='uv')
        #     FreiHand.plot_hand(ax2, joint_img_orig[:, 0:2], order='uv')
        #     ax1.axis('off')
        #     nn = str(random.randint(1,3000))
        #     #print("=============================================================")
        #     #print(nn)
        #     plt.savefig('/home/mqadri/hand-integral-pose-estimation/tests/{}.jpg'.format(nn))
        #=======================================================================
        
        #sys.exit()
        #nn = str(random.randint(2001,3000))
        #cv2.imwrite('/home/mqadri/hand-integral-pose-estimation/tests/{}.jpg'.format(nn), cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR))
    