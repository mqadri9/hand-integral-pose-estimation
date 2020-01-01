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
        
        K = data['K']
        joint_cam = data["joint_cam"]
        # 1. load image
        cvimg = cv2.imread(data['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        if not self.main_loop:
            return cvimg
        if not isinstance(cvimg, np.ndarray):
            raise IOError("Fail to read %s" % data['img_path'])
        img_height, img_width, img_channels = cvimg.shape
        
        # 2. get augmentation params
        if self.do_augment:
            scale, R, color_scale = augment.get_aug_config()
            #scale, rot, color_scale = 1.0, 0, [1.0, 1.0, 1.0]
        else:
            scale, R, color_scale = 1.0, np.eye(3), [1.0, 1.0, 1.0]

        #img_patch = self.transform(cvimg)
        #print(data['img_path'])
        #R = np.eye(3)
        img_patch, trans, joint_img, joint_img_orig, joint_vis, xyz_rot, bbox = augment.generate_patch_image(cvimg, joint_cam, scale, R, K)
        #img_patch = self.transform(cvimg)
        # 4. generate patch joint ground truth
        # color 
        # random noise
        #cons = ({'type': 'ineq', 'fun': lambda x:  x-0.1})
        #init = 10000 
        #g = scipy.optimize.minimize(F, init, args=(xyz_rot[:,0:2], joint_img[:,0:2],), method="BFGS", tol=1e-6,
        #                           options={'disp': False, 'maxiter': 50000})#, constraints=cons)
        #sol = g.x
        #print("==================================================")
        #print("xyz_rot")
        #print(xyz_rot)
        #print("joint_img")
        #print(joint_img)
        #joint_img[:,2] = joint_img[:,2]*sol + 112.5
        #print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        #print("max: {} | min: {}".format(str(np.min(joint_img[:,2])), str(np.max(joint_img[:,2]))))
        #print("===================================")
        #print("max: {} | min: {}".format(str(np.min(joint_img[:,0])), str(np.max(joint_img[:,0]))))
        #print("====================================")
        #print("max: {} | min: {}".format(str(np.min(joint_img[:,1])), str(np.max(joint_img[:,1]))))
        #print(joint_img[:,2]*sol)


        # 4. generate patch joint ground truth        
        for n_jt in range(len(joint_img)):
            joint_img[n_jt, 0:2] = augment.trans_point2d(joint_img[n_jt, 0:2], trans)


        #=======================================================================
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
        # plt.savefig('/home/mqadri/hand-integral-pose-estimation/tests/{}.jpg'.format(nn))
        # 
        # sys.exit()
        #=======================================================================
        #nn = str(random.randint(2001,3000))
        #cv2.imwrite('/home/mqadri/hand-integral-pose-estimation/tests/{}.jpg'.format(nn), cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR))
         
        img_patch = self.transform(img_patch)
        # apply normalization
        for n_c in range(img_channels):
            img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
        
        for n_jt in range(len(joint_img)):
            # TODO divide by rect_3d_width
            joint_img[n_jt, 2] = joint_img[n_jt, 2] / (cfg.bbox_3d_shape[0] * scale) * cfg.patch_width
        #print(bbox[2])
        #sys.exit()
            #else:
            #    joints[n_jt, 2] = joints[n_jt, 2] / (rect_3d_width * scale) * patch_width
        
        #print("joints")
        #print(joints)
        #sys.exit()  
    
        # 5. get label of some type according to certain need

        #joint_img = joint_img_orig
        #img_patch = self.transform(cvimg)

        label, label_weight = augment.generate_joint_location_label(cfg.patch_width, cfg.patch_height, joint_img, joint_vis)
        #print(label)
        #print(label_weight)
        #sys.exit()
        #=======================================================================
        #print("label")
        #print(label)
        #print("label_weight")
        #print(label_weight)
        #print("img_patch")
        #print(img_patch)
        # sys.exit()    
        #=======================================================================
        #=======================================================================
        # print(label_weight.shape)
        # print(label.shape)
        # print(img_patch.shape)
        #=======================================================================
        return img_patch, label, label_weight
        

        
    def __len__(self):
        if self.multiple_db:
            return max([len(db) for db in self.db]) * len(self.db)
        else:
            return len(self.db)
    
def F(alpha, X, U):
    s = [np.sqrt((alpha*X[i, 0] - U[i,0])**2 +  (alpha*X[i, 1] - U[i,1])**2) for i in range(FreiHandConfig.num_joints)]
    sol = np.sum(s)
    return sol



    