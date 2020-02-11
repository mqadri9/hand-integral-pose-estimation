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
                if is_train: 
                    if cfg.use_filtered_data:
                        self.db = [d.load_filtered_data() for d in db]
                    else:
                        self.db = [d.load_data() for d in db]
                else:
                    self.db = [d.load_data() for d in db]
            else:
                self.db = [d.load_evaluation_data() for d in db]
            #self.joints_name = [d.joints_name for d in db]
            self.joint_num = [d.joint_num for d in db]
            self.skeleton = [d.skeleton for d in db]
            #self.lr_skeleton = [d.lr_skeleton for d in db]
            #self.flip_pairs = [d.flip_pairs for d in db]
            self.joints_have_depth = [d.joints_have_depth for d in db]
            self.num_labelled = [d.num_labelled for d in db]
            self.num_unlabelled = [d.num_unlabelled for d in db]
        else:
            self.multiple_db = False
            if not is_eval:
                if is_train:
                    if cfg.use_filtered_data:
                        self.db = db.load_filtered_data()
                    else:
                        self.db = db.load_data()
                else:
                    self.db = db.load_data()
            else:
                self.db = db.load_evaluation_data()
            self.joint_num = db.joint_num
            self.skeleton = db.skeleton
            #self.lr_skeleton = db.lr_skeleton
            #self.flip_pairs = db.flip_pairs
            self.joints_have_depth = db.joints_have_depth
            self.num_labelled = db.num_labelled
            self.num_unlabelled = db.num_unlabelled
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
            if self.is_train and cfg.custom_batch_selection:
               if random.random() < cfg.labelled_selection_prob:
                   item_idx = np.random.randint(self.num_labelled[db_idx])
               else:
                   item_idx = np.random.randint(self.num_labelled[db_idx], self.num_labelled[db_idx] + self.num_unlabelled[db_idx])       
            else:
                item_idx = index % max([len(db) for db in self.db]) % len(self.db[db_idx])
            data = copy.deepcopy(self.db[db_idx][item_idx])
        else:
            joint_num = self.joint_num
            skeleton = self.skeleton
            joints_have_depth = self.joints_have_depth
            if self.is_train and cfg.custom_batch_selection:
                if random.random() < cfg.labelled_selection_prob:
                    index = np.random.randint(self.num_labelled)
                else:
                    index = np.random.randint(self.num_labelled, self.num_labelled + self.num_unlabelled)
            data = copy.deepcopy(self.db[index])
            
        
        cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img_height, img_width, img_channels = cvimg.shape
        if self.do_augment:
            scale, R, color_scale = augment.get_aug_config()
            #scale, rot, color_scale = 1.0, 0, [1.0, 1.0, 1.0]
        else:
            scale, R, color_scale = 1.0, np.eye(3), [1.0, 1.0, 1.0]
        
        if self.is_train and cfg.use_filtered_data:
            img_path = data["img_path"]
            K = data["K"]
            version = data["version"]
            ref_bone_len = data["ref_bone_len"]
            joint_cam_normalized = data["joint_cam_normalized"]
            tprime = data["tprime"]
            faster_rcnn_bbox = data["faster_rcnn_bbox"]
            joint_cam = data["joint_cam"]
            # generate pseudo label from the saved filtered output
            img_patch, trans, joint_img, joint_vis, xyz_rot_scaled  = augment.generate_patch_image_from_normalized(cvimg, img_path, joint_cam_normalized, tprime, R, K, scale, inv=False,
                                                                                                            hand_detector = self.hand_detector, faster_rcnn_bbox=faster_rcnn_bbox)
            

            for n_jt in range(len(joint_img)):
                joint_img[n_jt, 0:2] = augment.trans_point2d(joint_img[n_jt, 0:2], trans)
            label_teacher, label_weight = augment.generate_joint_location_label(cfg.patch_width, cfg.patch_height, joint_img, joint_vis)           
            
            # generate true label to compare if 3D grountruth exists. In the case of labelled data that we are treating as unlabelled, 
            # the generated label is only used as a way to generate predictions accuracy metrics from the unlabelled samples
            # in case the sample as marked as Labelled, the label is used to calculate the supervised loss               
            if cfg.use_hand_detector:
                if faster_rcnn_bbox is None:
                    print("(Warning) use_hand_detector is set to True but faster_rcnn_bbox is None")
                _, _, joint_img, _, joint_cam_normalized, joint_vis, xyz_rot, _, tprime = augment.generate_patch_image(cvimg, joint_cam, scale, R, K, inv=False, 
                                                                                                                              hand_detector=self.hand_detector, 
                                                                                                                              img_path=data['img_path'],
                                                                                                                              faster_rcnn_bbox=faster_rcnn_bbox)
            else:
                _, _, joint_img, _, joint_cam_normalized, joint_vis, xyz_rot, _, tprime = augment.generate_patch_image(cvimg, joint_cam, scale, R, K, inv=False)   
            
            for n_jt in range(len(joint_img)):
                joint_img[n_jt, 0:2] = augment.trans_point2d(joint_img[n_jt, 0:2], trans)
            label, _ = augment.generate_joint_location_label(cfg.patch_width, cfg.patch_height, joint_img, joint_vis) 
            
            img_patch = self.transform(img_patch)
            for n_c in range(img_channels):
                img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)

            params = {
                "R": R,
                "K": K,
                "joint_cam": joint_cam,
                "scale": scale,
                "img_path": data['img_path'],
                "tprime": data["tprime"],
                "tprime_torch": torch.from_numpy(np.array([data["tprime"]])),
                "bbox": np.array(faster_rcnn_bbox),
                "trans": trans,
                "joint_cam_normalized": data["joint_cam_normalized"],
                "joint_img_orig": np.zeros(data["joint_cam_normalized"].shape),
                "ref_bone_len": data["ref_bone_len"],
                "labelled": data["labelled"],
                "label": label,
                "label_weight": label_weight,
                "label_teacher": label_teacher
            }
                      
            return img_patch, params
        
        elif not self.is_eval:
            K = data['K']
            joint_cam = data["joint_cam"]
            faster_rcnn_bbox = data['faster_rccn_bbox']
            # 1. load image
            
            if not self.main_loop:
                return cvimg
            if not isinstance(cvimg, np.ndarray):
                raise IOError("Fail to read %s" % data['img_path'])
           
            
            # 2. get augmentation params
            #self.do_augment = True
            if cfg.use_hand_detector:
                if faster_rcnn_bbox is None:
                    print("(Warning) use_hand_detector is set to True but faster_rcnn_bbox is None")
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
                "K": K,
                "joint_cam": joint_cam,
                "scale": scale,
                "img_path": data['img_path'],
                "tprime": tprime,
                "tprime_torch": torch.from_numpy(np.array([tprime])),
                "bbox": np.array(bbox),
                "trans": trans,
                "joint_cam_normalized": joint_cam_normalized,
                "joint_img_orig": joint_img_orig,
                "ref_bone_len": data["ref_bone_len"],
                "labelled": data["labelled"]
            }
            label, label_weight = augment.generate_joint_location_label(cfg.patch_width, cfg.patch_height, joint_img, joint_vis)            
            params["label"] = label
            params["label_weight"] = label_weight
            params["label_teacher"] = np.zeros(label.shape) 
            #===================================================================
            # else:
            #     img_patch, params = augment.generate_input_unlabelled(cvimg, R, scale, data)
            #     params["R"] = R
            #     params["joint_cam"] = np.zeros((21, 3))
            #     params["scale"] = scale
            #     params["trans"] = np.eye(3)
            #     params["joint_cam_normalized"] = np.zeros((21, 3))
            #     params["joint_img_orig"] = np.zeros((21, 3))
            #     params["label"] = np.zeros((63,))
            #     params["label_weight"] = np.ones((63,))
            #===================================================================
           
            img_patch = self.transform(img_patch)

            for n_c in range(img_channels):
                img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
            return img_patch, params
        
        else:
            img_patch, params = augment.generate_input_unlabelled(cvimg, R, scale, data)
            img_patch = self.transform(img_patch)
            return img_patch, params
        
    def __len__(self):
        if self.multiple_db:
            return max([len(db) for db in self.db]) * len(self.db)
        else:
            return len(self.db)

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
    