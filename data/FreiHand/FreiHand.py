import os, time, sys
from FreiHand_config import FreiHandConfig
import skimage.io as io
import numpy as np
import json
import pickle as pk
import random
import cv2
import matplotlib.pyplot as plt
import augment
from config import cfg
plt.switch_backend('agg')

class FreiHand:
    gs = 'gs'  # green screen
    hom = 'hom'  # homogenized
    sample = 'sample'  # auto colorization with sample points
    auto = 'auto'  # auto colorization without sample points: automatic color hallucination
    
    def __init__(self, data_split="training"):
        self.data_split = data_split
        #self.data_split = "training"
        self.data_dir = os.path.join('..', 'data', 'FreiHand')
        #if data_split == "training":
        #    self.data_dir = os.path.join('..', 'data', 'FreiHand', 'training', 'rgb')
        #elif data_split == "evaluation":
        #    self.data_dir = os.path.join('..', 'data', 'FreiHand', 'evaluation', 'rgb')
        self.name = "FreiHand"
        self.joint_num = 21
        self.skeleton = FreiHandConfig.bones
        self.joints_have_depth = True
        self.eval_joint = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20)
        self.root_idx = 9
        self.size_db = 32560

    def _assert_exist(self, p):
        msg = 'File does not exists: %s' % p
        assert os.path.exists(p), msg
    
    def json_load(self, p):
        self._assert_exist(p)
        with open(p, 'r') as fi:
            d = json.load(fi)
        return d
    
    """ Draw functions. """
    @classmethod
    def plot_hand(cls, axis, coords_hw, vis=None, color_fixed=None, linewidth='1', order='hw', draw_kp=True):
        """ Plots a hand stick figure into a matplotlib figure. """
        if order == 'uv':
            coords_hw = coords_hw[:, ::-1]
            
        colors = FreiHandConfig.colors
        
        # define connections and colors of the bones
        bones = FreiHandConfig.bones_color
    
        if vis is None:
            vis = np.ones_like(coords_hw[:, 0]) == 1.0
    
        for connection, color in bones:
            if (vis[connection[0]] == False) or (vis[connection[1]] == False):
                continue
    
            coord1 = coords_hw[connection[0], :]
            coord2 = coords_hw[connection[1], :]
            coords = np.stack([coord1, coord2])
            if color_fixed is None:
                axis.plot(coords[:, 1], coords[:, 0], color=color, linewidth=linewidth)
            else:
                axis.plot(coords[:, 1], coords[:, 0], color_fixed, linewidth=linewidth)
    
        if not draw_kp:
            return
    
        for i in range(21):
            if vis[i] > 0.5:
                axis.plot(coords_hw[i, 1], coords_hw[i, 0], 'o', color=colors[i, :])


    
    # Since we need labelled evaluation data, 
    # need to figure out a way to split the training data into training and evaluation
    def _sample_dataset(self, data_split):
        # We split the training data which is labelled to a training and testing set
        if data_split == "training" or data_split == "testing":
            folders = [os.path.join(self.data_dir, 'training', 'rgb')]
        elif data_split == "evaluation":
            folders = [os.path.join(self.data_dir, 'evaluation', 'rgb')]
        else: 
            print("Unknown subset")
            assert 0
        return folders

    @classmethod
    def valid_options(cls):
        return [cls.gs, cls.hom, cls.sample, cls.auto]
    
    @classmethod
    def check_valid(cls, version):
        msg = 'Invalid choice: "%s" (must be in %s)' % (version, cls.valid_options())
        assert version in cls.valid_options(), msg

    def map_id(self, id, version):
        self.check_valid(version)
        return id + self.size_db*self.valid_options().index(version)

    """ Dataset related functions. 
    Need to change how to do the datasplit to add testing and evaluation splits"""
    def db_size(self, data_split):
        """ Hardcoded size of the datasets. """
        if data_split == 'training':
            #return 32560  # number of unique samples (they exists in multiple 'versions')
            return 30000
        elif data_split == "testing":
            return 2560
        elif data_split == 'evaluation':
            return 3960
        else:
            assert 0, 'Invalid choice.'
    
    def load_db_annotation(self, base_path, data_split=None):
        if data_split is None:
            # only training set annotations are released so this is a valid default choice
            data_split = 'training'
    
        print('Loading FreiHAND dataset index ...')
        t = time.time()
    
        # assumed paths to data containers
        k_path = os.path.join(base_path, '%s_K.json' % data_split)
        mano_path = os.path.join(base_path, '%s_mano.json' % data_split)
        xyz_path = os.path.join(base_path, '%s_xyz.json' % data_split)
    
        # load if exist
        K_list = self.json_load(k_path)
        mano_list = self.json_load(mano_path)
        xyz_list = self.json_load(xyz_path)
    
        # should have all the same length
        assert len(K_list) == len(mano_list), 'Size mismatch.'
        assert len(K_list) == len(xyz_list), 'Size mismatch.'
    
        print('Loading of %d samples done in %.2f seconds' % (len(K_list), time.time()-t))
        return list(zip(K_list, mano_list, xyz_list))
    
    def projectPoints(self, xyz, R, K, p=False):
        """ Project 3D coordinates into image space. """
        xyz = np.array(xyz)
        K = np.array(K)
        #uv = np.matmul(K, xyz.T).T
        xyz_rot = np.matmul(R, xyz.T).T
        uv = np.matmul(K, xyz_rot.T).T
        if p:
            print("uv")
            print(uv)
            print("uv1")
            print(uv[:, :2] / uv[:, -1:])
            print("xyz orig")
            print(xyz)  
        return uv[:, :2] / uv[:, -1:], xyz_rot[:, -1]*1000, xyz_rot
   
    def read_img(self, idx, base_path, set_name, version=None):
        if version is None:
            version = gs
    
        if set_name == 'evaluation':
            assert version == gs, 'This the only valid choice for samples from the evaluation split.'

        img_rgb_path = os.path.join(base_path, set_name, 'rgb',
                                    '%08d.jpg' % self.map_id(idx, version))
        self._assert_exist(img_rgb_path)
        return io.imread(img_rgb_path), img_rgb_path
    
    
    def read_msk(self, idx, base_path):
        mask_path = os.path.join(base_path, 'training', 'mask',
                                 '%08d.jpg' % idx)
        self._assert_exist(mask_path)
        return io.imread(mask_path)
    
    def process_coordinates(self, xyz, vis, K, aspect_ratio = 1.0):
        """ 
        Returns: 
        1) joint_cam: the xyz in camera reference frame
        2) center_cam: the xyz coordinates of the root joint
        """
        # xyz already in camera coordinates in m? convert to mm  
        joint_cam = xyz                
        return joint_cam
    
    def load_evaluation_data(self):
        if cfg.eval_version == 1:
            print("Evaluating version 1")
            data_directory = os.path.join(self.data_dir, "evaluation_v1")
            img_rgb_path = os.path.join(self.data_dir, "evaluation_v1", 'rgb')
        else:
            data_directory = self.data_dir
            img_rgb_path = os.path.join(self.data_dir, "evaluation", 'rgb')
        
        k_path = os.path.join(data_directory, 'evaluation_K.json')
        scale_path = os.path.join(data_directory, 'evaluation_scale.json')
        t = time.time()
        # load if exist
        K_list = self.json_load(k_path)
        scale_list = self.json_load(scale_path)
        
        imglist = os.listdir(img_rgb_path)
        lst = [os.path.splitext(x)[0] for x in imglist]
        lst.sort(key = int)
        num_images = len(imglist)    
        # should have all the same length
        assert len(K_list) == len(scale_list), 'Size mismatch.'
    
        print('Loading of %d samples done in %.2f seconds' % (len(K_list), time.time()-t))
        data = []
        for i in range(num_images):
            d = {
                "K": K_list[i],
                "bone_length": scale_list[i],
                "img_path": os.path.join(img_rgb_path, lst[i] + '.jpg')
            }
            data.append(d)
        self.num_samples = num_images
        print('{} samples read'.format(len(data)))
        return data

   
    def load_data(self):
        # Need to see if we should train on all versions in the datasplit
        #version = 'gs'
        versions = ['gs', 'hom', 'sample', 'auto']
        #versions = ['gs']
        # Read the directory containing the rgb images
        folders = self._sample_dataset(self.data_split)
        db_data_anno = self.load_db_annotation(self.data_dir, 'training')
        
        cache_file = '{}_keypoint_bbox_db_{}.pkl'.format(self.name, self.data_split)
        cache_file = os.path.join(self.data_dir, self.data_split, cache_file)
        db = None
        if os.path.exists(cache_file):
            with open(cache_file, 'rb') as fid:
                db = pk.load(fid)
            print('{} gt db loaded from {}, {} samples are loaded'.format(self.name, cache_file, len(db)))
    
        if db != None:
            self.num_samples = len(db)
            return db
        data = []
        
        if self.data_split == "training":
            start = 0
            end = self.db_size('training')
            d_s = "training"
            
        elif self.data_split == "testing":
            start = self.db_size('training') + 1
            end = start + self.db_size('testing') -1
            d_s = "training"

        print('split: {}. start index is: {}. End index is: {}'.format(self.data_split, start, end))
        for version in versions:
            print("==================================version==========")
            print(version)
            for idx in range(start, end):
                if idx%1000 == 0:
                    print(idx)
                img, img_path = self.read_img(idx, self.data_dir, d_s, version)
                msk = self.read_msk(idx, self.data_dir)
                # annotation for this frame
                K, mano, xyz = db_data_anno[idx]
                K, mano, xyz = [np.array(x) for x in [K, mano, xyz]]
                # right now assume all points are visible . However, we can modify the
                # db_data_annot to return the visibility for each sample
                # can create a training_visiblity.json that we can read
                # Right now assume all points are visible
                #===============================================================
                # uv, _, _ = self.projectPoints(xyz, np.eye(3), K)
                # fig = plt.figure()
                #   
                # ax1 = fig.add_subplot(121)
                # ax2 = fig.add_subplot(122)
                # # 
                # #ax1.imshow((255*img_patch/np.max(img_patch)).astype(np.uint8))
                # ax2.imshow(img)
                # #ax1.imshow(img2_w)
                # # 
                # #FreiHand.plot_hand(ax1, joint_img[:, 0:2], order='uv')
                # self.plot_hand(ax2, uv, order='uv')
                # ax1.axis('off')
                # nn = str(random.randint(1,3000))
                # plt.savefig('/home/mqadri/hand-integral-pose-estimation/tests/{}.jpg'.format(nn))    
                #===============================================================
                vis = np.ones(xyz.shape)          
                joint_cam = self.process_coordinates(xyz, vis, K)
                data.append({
                    'img_path': img_path,
                    'joint_cam': joint_cam, # [X, Y, Z] in camera coordinate
                    'K': K,
                    'version': version,
                    "idx": idx
                })
            
        with open(cache_file, 'wb') as fid:
            pk.dump(data, fid, pk.HIGHEST_PROTOCOL)
        print('{} samples read wrote {}'.format(len(data), cache_file))
        self.num_samples = len(data)
        return data
    
    def gen_test_data(self, augmentation_list):
        #data = self.load_data()
        len_gts = len(augmentation_list["img_path"])
        gts = []
        for i in range(len_gts):
            K = augmentation_list['K'][i]
            joint_cam = augmentation_list["joint_cam"][i]
            R = augmentation_list["R"][i]
            scale = augmentation_list["scale"][i]
            #cvimg = cv2.imread(d['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            #===================================================================
            # print(R)
            # img_patch, trans1, joint_img, joint_img_orig, joint_vis, xyz_rot, bbox = augment.generate_patch_image(augmentation_list["cvimg"][i], 
            #                                                                                                      joint_cam, 
            #                                                                                                      scale, R, K, 
            #                                                                                                      aspect_ratio=1.0, inv=False)
            #===================================================================
            
            img_patch, trans, joint_img, joint_img_orig, joint_vis, xyz_rot, bbox, zoom_factor, f, z_mean = augment.generate_patch_image(augmentation_list["cvimg"][i], 
                                                                                                                                         joint_cam, 
                                                                                                                                         scale, R, K, 
                                                                                                                                         aspect_ratio=1.0, inv=True)
            #===================================================================
            # print(trans1)
            # print(trans)
            # print(np.linalg.inv(trans))
            #===================================================================
            data = {
                'image': augmentation_list['img_path'][i],
                'center_x': bbox[0],
                'center_y': bbox[1],
                'width': bbox[2],
                'height': bbox[3],
                'joints_3d': joint_img_orig, # [org_img_x, org_img_y, depth - root_depth]
                'joints_3d_vis': joint_vis,
                'joints_3d_cam': joint_cam, # [X, Y, Z] in camera coordinate
                'K': K,
                'R': R,
                'trans': trans,
                'cvimg': augmentation_list["cvimg"][i],
                'scale': scale,
                'z_mean': z_mean,
                'f': f,
                'zoom_factor': zoom_factor
            }
            gts.append(data)
        return gts
    
    def test_verify_identity(self, n, gt_3d_kpt, gts, joints_3d):
        #=======================================================================
        # joint_vis = np.ones(joints_3d.shape, dtype=np.float)
        # # Augment
        # #print("===================================")
        # #print(gt_3d_kpt)
        gt_3d_kpt_save = np.copy(gt_3d_kpt)
        # xyz_rot = np.matmul(gts[n]["R"], gt_3d_kpt.T).T
        # scale = gts[n]["scale"]
        # img_patch, trans, joint_img, joint_img_orig, joint_vis, xyz_rot, bbox, zoom_factor, f, z_mean = augment.generate_patch_image(gts[n]["cvimg"], 
        #                                                                                                                              gt_3d_kpt, 
        #                                                                                                                              scale, 
        #                                                                                                                              gts[n]["R"], 
        #                                                                                                                              gts[n]["K"])
        # joint_img_sav_1 = np.copy(joint_img)
        # for n_jt in range(len(joint_img)):
        #     joint_img[n_jt, 0:2] = augment.trans_point2d(joint_img[n_jt, 0:2], trans)
        # 
        # #=======================================================================
        # # for n_jt in range(len(joint_img)):
        # #     # TODO divide by rect_3d_width
        # #     zoom_factor = max(bbox[3], bbox[2])
        # #     joint_img[n_jt, 2] = joint_img[n_jt, 2] / (zoom_factor * scale) * cfg.patch_width
        # #=======================================================================
        #     #joint_img[n_jt, 2] = joint_img[n_jt, 2] / (cfg.bbox_3d_shape[0] * scale) * cfg.patch_width
        # 
        # for n_jt in range(len(joint_img)):
        #     # TODO divide by rect_3d_width
        #     #zoom_factor = max(bbox[3], bbox[2])
        #     joint_img[n_jt, 2] = (joint_img[n_jt, 2] * f * zoom_factor) / (z_mean * cfg.patch_width)
        # 
        # joint_img_sav_2 = np.copy(joint_img)
        # label, label_weight = augment.generate_joint_location_label(cfg.patch_width, cfg.patch_height, joint_img, joint_vis)
        #=======================================================================
        
        # UnAugment
        label = label.astype(float)
        label = label.reshape((int(label.shape[0] / 3), 3))
        label[:, 0] = (label[:, 0] + 0.5) * cfg.patch_width
        label[:, 1] = (label[:, 1] + 0.5) * cfg.patch_height
        label[:, 2] = label[:, 2] * cfg.patch_width
        
        assert np.allclose(label, joint_img_sav_2, rtol=1e-10, atol=1e-10)
        # add score to last dimension
                
        pre_2d_kpt = augment.trans_coords_from_patch_to_org_3d(label, gts[n]['center_x'],
                                                               gts[n]['center_y'], gts[n]['width'],
                                                               gts[n]['height'], cfg.patch_width, 
                                                               cfg.patch_height, scale, gts[n]['trans'], 
                                                               gts[n]['zoom_factor'], gts[n]['z_mean'], gts[n]['f'])
        
        assert np.allclose(pre_2d_kpt, joint_img_sav_1, rtol=1e-10, atol=1e-10)
        
        
        pre_2d_kpt[:,2] = pre_2d_kpt[:,2] + xyz_rot[:,2][9]*1000
        
        pre_3d_kpt = augment.pixel2cam(pre_2d_kpt, gts[n]['K']) 
        pre_3d_kpt = np.matmul(gts[n]["R"].T, pre_3d_kpt.T).T
        
        #print(pre_3d_kpt)
        #print(gt_3d_kpt_save)
        assert np.allclose(pre_3d_kpt, gt_3d_kpt_save, rtol=1e-10, atol=1e-10)
          
    def evaluate(self, preds_in_patch_with_score, label_list, augmentation_list, result_dir):
        
        print() 
        print('Evaluation start...')
        
        gts = self.gen_test_data(augmentation_list)
        
        print("len gts is {}".format(len(gts)))
        # From patch to original image coordinate system
        preds_in_img_with_score = []
        label_in_img_with_score = []
        for n_sample in range(len(gts)):
            #print(gts[n_sample]['trans'])
            #print("shape")
            #print(preds_in_patch_with_score[n_sample].shape)
            preds_in_img_with_score.append(
                augment.trans_coords_from_patch_to_org_3d(preds_in_patch_with_score[n_sample], gts[n_sample]['center_x'],
                                                          gts[n_sample]['center_y'], gts[n_sample]['width'],
                                                          gts[n_sample]['height'], cfg.patch_width, cfg.patch_height, gts[n_sample]['scale'], 
                                                          gts[n_sample]['trans'], gts[n_sample]['zoom_factor'], gts[n_sample]['z_mean'], gts[n_sample]['f']))
            label_in_img_with_score.append(
                augment.trans_coords_from_patch_to_org_3d(label_list[n_sample], gts[n_sample]['center_x'],
                                                          gts[n_sample]['center_y'], gts[n_sample]['width'],
                                                          gts[n_sample]['height'], cfg.patch_width, cfg.patch_height, gts[n_sample]['scale'], 
                                                          gts[n_sample]['trans'], gts[n_sample]['zoom_factor'], gts[n_sample]['z_mean'], gts[n_sample]['f']))
    
        preds_in_img_with_score = np.asarray(preds_in_img_with_score)
        label_in_img_with_score = np.asarray(label_in_img_with_score)
        preds = preds_in_img_with_score[:, :, 0:3]
        sample_num = preds.shape[0]
        joint_num = self.joint_num
        p1_error = np.zeros((sample_num, joint_num, 3)) # PA MPJPE (protocol #1 metric)
        p2_error = np.zeros((sample_num, joint_num, 3)) # MPJPE (protocol #2 metroc)
        pred_to_save = []
        pr = []
        pr_procr = []
        gtss = []
        file_name = []
        for n in range(sample_num):
            gt = gts[n]
            R = gt["R"]
            K = gt['K']
            gt_3d_kpt = gt['joints_3d_cam']
            xyz_rot = np.matmul(R, gt_3d_kpt.T).T
            gt_vis = gt['joints_3d_vis'].copy()
            #self.test_verify_identity(n, gt_3d_kpt, gts, gt['label'])
            pre_2d_kpt = preds[n].copy()
            _label = label_in_img_with_score[n].copy()
            #print(pre_2d_kpt)
            #pre_2d_kpt[:,2] = np.squeeze(pre_2d_kpt[:,2] - pre_2d_kpt[:,2][FreiHandConfig.root_idx])
            pre_2d_kpt[:,2] = pre_2d_kpt[:,2] + xyz_rot[:,2][9]*1000
            _label[:,2] = _label[:,2] + xyz_rot[:,2][9]*1000
            pre_3d_kpt = np.zeros((joint_num,3))
            pre_3d_kpt = augment.pixel2cam(pre_2d_kpt, K)
            pre_3d_kpt = np.matmul(R.T, pre_3d_kpt.T).T
            label_3d_kpt = np.zeros((joint_num,3))
            label_3d_kpt = augment.pixel2cam(_label, K)
            label_3d_kpt = np.matmul(R.T, label_3d_kpt.T).T
            assert np.allclose(label_3d_kpt, gt_3d_kpt, rtol=1e-6, atol=1e-6)
            #pre_3d_kpt = pre_3d_kpt - pre_3d_kpt[self.root_idx]
            #gt_3d_kpt  = gt_3d_kpt - gt_3d_kpt[self.root_idx]
            # rigid alignment for PA MPJPE (protocol #1)
            _, pre_3d_kpt_align, T, b, c = augment.compute_similarity_transform(gt_3d_kpt, pre_3d_kpt, compute_optimal_scale=True)

            uv1, _, _ = augment.projectPoints(gt_3d_kpt, np.eye(3), K)
            uv2, _, _ = augment.projectPoints(pre_3d_kpt_align, np.eye(3), K)
            #print("=====================================")
            #print(pre_3d_kpt)
            #print(gt_3d_kpt)

            # prediction save
            
            #if (n==1000):
            #    print("------")
            #    print(n)
            #    print(gt["image"])
            #    print(gt_3d_kpt)
            pred_to_save.append({'pred': pre_3d_kpt,
                                 'align_pred': pre_3d_kpt_align,
                                 'gt': gt_3d_kpt})
           
            pr.append(pre_3d_kpt)
            pr_procr.append(pre_3d_kpt_align)
            gtss.append(gt_3d_kpt)
            file_name.append(gt["image"])
            # error save
            p1_error[n] = np.power(pre_3d_kpt_align - gt_3d_kpt,2) # PA MPJPE (protocol #1)
            p2_error[n] = np.power(pre_3d_kpt - gt_3d_kpt,2)  # MPJPE (protocol #2)

#===============================================================================
#             img = cv2.imread(gt['image'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
# 
#             #print(gt["image"])
#             #print(uv1)
#             #print(uv2)
#             fig = plt.figure()
#                 
#             ax1 = fig.add_subplot(121)
#             ax2 = fig.add_subplot(122)
#             # 
#             #ax1.imshow((255*img_patch/np.max(img_patch)).astype(np.uint8))
#             ax1.imshow(img)
#             ax2.imshow(img)
#             #ax1.imshow(img2_w)
#             # 
#             self.plot_hand(ax1, uv1, order='uv')
#             self.plot_hand(ax2, uv2, order='uv')
#             ax1.axis('off')
#             nn = str(random.randint(1,3000))
#             plt.savefig('/home/mqadri/hand-integral-pose-estimation/tests/{}.jpg'.format(nn))    
#===============================================================================
             
            #print("=================================================")
            #print(gt_3d_kpt)
            #print(pre_3d_kpt_align)
            #if n==20:
            #    break
            
        # total error calculate
        np.save("ground_truth_test", gtss)
        np.save("pred", pr)
        np.save("pred_procr", pr_procr)
        np.save("file_name", file_name)
        p1_error = np.take(p1_error, self.eval_joint, axis=1)
        p2_error = np.take(p2_error, self.eval_joint, axis=1)
        p1_error = np.mean(np.power(np.sum(p1_error,axis=2),0.5))
        p2_error = np.mean(np.power(np.sum(p2_error,axis=2),0.5))
        
        p1_eval_summary = 'Protocol #1 error (PA MPJPE) >> %.8f' % (p1_error)
        p2_eval_summary = 'Protocol #2 error (MPJPE) >> %.8f' % (p2_error)
        print()
        print(p1_eval_summary)
        print(p2_eval_summary)
        # result save
        f_pred_3d_kpt = open(os.path.join(result_dir, 'pred_3d_kpt.txt'), 'w')
        f_pred_3d_kpt_align = open(os.path.join(result_dir, 'pred_3d_kpt_align.txt'), 'w')
        f_gt_3d_kpt = open(os.path.join(result_dir, 'gt_3d_kpt.txt'), 'w')
        for i in range(len(pred_to_save)):
            for j in range(joint_num):
                for k in range(3):
                    f_pred_3d_kpt.write('%.3f ' % pred_to_save[i]['pred'][j][k])
                    f_pred_3d_kpt_align.write('%.3f ' % pred_to_save[i]['align_pred'][j][k])
                    f_gt_3d_kpt.write('%.3f ' % pred_to_save[i]['gt'][j][k])
            f_pred_3d_kpt.write('\n')
            f_pred_3d_kpt_align.write('\n')
            f_gt_3d_kpt.write('\n')
        f_pred_3d_kpt.close()
        f_pred_3d_kpt_align.close()
        f_gt_3d_kpt.close()

        f_eval_result = open(os.path.join(result_dir, 'eval_result.txt'), 'w')
        f_eval_result.write(p1_eval_summary)
        f_eval_result.write('\n')
        f_eval_result.write(p2_eval_summary)
        f_eval_result.write('\n')


    def dump(self, pred_out_path, xyz_pred_list, verts_pred_list):
        """ Save predictions into a json file. """
        # make sure its only lists
        xyz_pred_list = [x.tolist() for x in xyz_pred_list]
        verts_pred_list = [x.tolist() for x in verts_pred_list]
    
        # save to a json
        with open(pred_out_path, 'w') as fo:
            json.dump(
                [
                    xyz_pred_list,
                    verts_pred_list
                ], fo)
        print('Dumped %d joints and %d verts predictions to %s' % (len(xyz_pred_list), len(verts_pred_list), pred_out_path))
        
    def estimate_depth(self, bone_length, K, pre_2d_kpt):
        fx = K[0, 0]
        fy = K[1, 1]
        U0 = K[0, 2]
        V0 = K[1, 2]
        
        Un = pre_2d_kpt[9, 0]
        Vn = pre_2d_kpt[9, 1]
        Zn = pre_2d_kpt[9, 2]
        Um = pre_2d_kpt[10, 0]
        Vm = pre_2d_kpt[10, 1]
        Zm = pre_2d_kpt[10, 2]
        
        Unm = (Un - Um) / fx
        Un0 = (Un - U0) / fx
        Um0 = (Um - U0) / fx
        
        Vnm = (Vn - Vm) / fy
        Vn0 = (Vn - V0) / fy
        Vm0 = (Vm - V0) / fy
        
        r_A = Unm ** 2 +  Vnm ** 2
        r_B = Unm * (Un0*Zn - Um0*Zm) + Vnm*(Vn0 * Zn - Vm0 * Zm)
        r_B*=2
        r_C = (Un0*Zn  - Um0*Zm)**2 + (Vn0*Zn - Vm0*Zm)**2 + (Zn - Zm) **2 - bone_length**2
        coeffs = [r_A, r_B, r_C]
        root = np.roots(coeffs)
        if np.iscomplexobj(root):
            print("Complex")
            print(root)
            root[0] = np.absolute(root[0])
            root[1] = np.absolute(root[1])
            return max(root[0], root[1]), True
        else:
            return max(root[0], root[1]), False
        #np.linalg.norm(pre_2d_kpt[9] - pre_2d_kpt[10], 2) 
    
    
    
    def evaluate_evaluations(self, preds_in_patch_with_score, params, result_dir):
        #=======================================================================
        # print(np.array(params["bone_length"]).shape)
        # print(np.array(params["K"]).shape)
        # print(np.array(params["img_path"]).shape)
        # print(np.array(params["bbox"]).shape)        
        #=======================================================================
        preds_in_img_with_score = []
        for n in range(preds_in_patch_with_score.shape[0]):
            pred = preds_in_patch_with_score[n]
            bbox = params["bbox"][n]
            bone_length = params["bone_length"][n]
            K = params["K"][n]
            img_path = params["img_path"][n]
            center_x = bbox[0, 0]
            center_y = bbox[0, 1]
            width = bbox[0, 2]
            height = bbox[0, 3]
            trans = augment.gen_trans_from_patch_cv(center_x, center_y, width, height, cfg.input_shape[1], cfg.input_shape[0], 1.0, inv = True)
            preds_in_img_with_score.append(augment.trans_coords_from_patch_to_org_3d(pred, center_x, center_y, width, height, cfg.patch_width, cfg.patch_height, 1.0, 
                                           trans, 1.0, 1.0, 1.0))
            
        preds_in_img_with_score = np.asarray(preds_in_img_with_score)
        preds = preds_in_img_with_score[:, :, 0:3]
        sample_num = preds.shape[0]
        predictions = []
        vertices = []
        for n in range(sample_num):
            pre_2d_kpt = preds[n].copy()
            #print(pre_2d_kpt)
            bone_length = params["bone_length"][n]
            # convert to mm
            bone_length *= 1000
            #print(bone_length)
            K = params["K"][n]
            img_path = params["img_path"][n]
            d, iscomplex = self.estimate_depth(bone_length, K, pre_2d_kpt)
            print(d)
            print(pre_2d_kpt)
            pre_2d_kpt[:,2] = pre_2d_kpt[:,2] + d
            pre_3d_kpt = np.zeros((21,3))
            pre_3d_kpt = augment.pixel2cam(pre_2d_kpt, K)
            verts = np.zeros((778, 3))
       #========================================================================
       #      if iscomplex:
       #          uv1, _, _ = augment.projectPoints(pre_3d_kpt, np.eye(3), K)
       #          img = cv2.imread(img_path[0], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
       # 
       #          #print(gt["image"])
       #          #print(uv1)
       #          #print(uv2)
       #          fig = plt.figure()
       #                 
       #          ax1 = fig.add_subplot(121)
       #          ax2 = fig.add_subplot(122)
       #          # 
       #          #ax1.imshow((255*img_patch/np.max(img_patch)).astype(np.uint8))
       #          ax1.imshow(img)
       #          ax2.imshow(img)
       #          #ax1.imshow(img2_w)
       #          # 
       #          self.plot_hand(ax1, uv1, order='uv')
       #          self.plot_hand(ax2, uv1, order='uv')
       #          ax1.axis('off')
       #          nn = str(random.randint(1,200000))
       #          plt.savefig('/home/mqadri/hand-integral-pose-estimation/tests/{}.jpg'.format(nn))
       #          plt.close(fig)
       #      
       #========================================================================
            vertices.append(verts)
            print(pre_3d_kpt)
            predictions.append(pre_3d_kpt)
            sys.exit()
        np.save("evaluation_predictions", predictions)
        np.save("vertices", vertices)
        self.dump('pred.json', predictions, vertices)
        print("completed")
            
      
    def evaluate_kp(self):
        return


        
        