import argparse
from config import cfg
from base import Trainer
import torch.backends.cudnn as cudnn
import sys, time
import numpy as np
from base import Tester
from tqdm import tqdm
from torch.nn.parallel.scatter_gather import gather
import torch
import config_panet
from nets.loss import softmax_integral_tensor, softmax_integral_tensor2, JointLocationLoss
from FreiHand_config import FreiHandConfig
from FreiHand import FreiHand
import os
import matplotlib.pyplot as plt 
from scipy import stats, optimize, interpolate
import cv2
from random import uniform
import random
import torchvision.transforms as transforms
import pickle as pk

def make_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)


def convert_cvimg_to_tensor(cvimg):
    # from h,w,c(OpenCV) to c,h,w
    tensor = cvimg.copy()
    tensor = np.transpose(tensor, (2, 0, 1))
    # from BGR(OpenCV) to RGB
    tensor = tensor[::-1, :, :]
    # from int to float
    tensor = tensor.astype(np.float32)
    return tensor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    args = parser.parse_args()

    if not args.gpu_ids:
        args.gpu_ids = str(np.argmin(mem_info()))

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    return args

from config import cfg
from base import Trainer
import augment
import config_panet


def get_aug_config(theta=0):
    
    scale_factor = 0.25
    color_factor = 0.2
    
    #scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    scale = 1.0
    #rot = np.clip(np.random.randn(), -2.0,
    #              2.0) * rot_factor if random.random() <= 0.6 else 0
    rot = sample_rotation_matrix(theta=0)
    
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]

    return scale, rot, color_scale

def sample_rotation_matrix(theta=0):
    # Rotate with a probability of 40 percent
    # Right now the only rotation is around the z axis from -30 deg tp 30 deg
    
    if np.abs(theta) < 1e-4:
        R1 = np.eye(3)
    else:
        s = np.zeros((2,1))
        r = np.random.randn(1,1)
        r = np.vstack((s, r))
        r = theta*(r/np.linalg.norm(r))
        #print(r.shape)
        R1, _ = cv2.Rodrigues(r)
    #R1 = np.eye(3)
    theta = uniform(-0.05, 0.05)
    #theta = 0.05
    if np.abs(theta) < 1e-4:
        R2 = np.eye(3)
    else:
        r = np.random.randn(3,1)    
        r = theta*(r/np.linalg.norm(r))
        R2, _ = cv2.Rodrigues(r)
    R = np.matmul(R1, R2)
    return np.array(R)

def crop_and_get_patch(cvimg, joint_cam, K, R, scale,
                       inv=False, 
                       hand_detector=None, 
                       img_path = None,
                       faster_rcnn_bbox = None):  
    return augment.generate_patch_image(cvimg, joint_cam, scale, R, K, inv=False, hand_detector=hand_detector, img_path=img_path, faster_rcnn_bbox=faster_rcnn_bbox)    


def generate_label(joint_img, trans, joint_vis, p=False):
    for n_jt in range(len(joint_img)):
        joint_img[n_jt, 0:2] = augment.trans_point2d(joint_img[n_jt, 0:2], trans)
    
    if p:
        print(joint_img)
        
    label, label_weight = augment.generate_joint_location_label(cfg.patch_width, cfg.patch_height, joint_img, joint_vis) 
    return label, label_weight

def convert_to_cam_coord(coord_in_patch, bbox, scale, trans, tprime, K, R):
    coord_in_image = augment.trans_coords_from_patch_to_org_3d(coord_in_patch, bbox[0],
                                                               bbox[1], bbox[2], bbox[3], cfg.patch_width, cfg.patch_height, scale, 
                                                               np.linalg.inv(trans), tprime)
    coord_in_image = coord_in_image[:, 0:3]
    pre_3d_kpt = augment.pixel2cam(coord_in_image, K)
    pre_3d_kpt = np.matmul(R.T, pre_3d_kpt.T).T
    return pre_3d_kpt

def computeMPJPE(pred, gt):
    pred = pred.reshape((pred.shape[0], FreiHandConfig.num_joints, 3))
    gt = gt.reshape((gt.shape[0], FreiHandConfig.num_joints, 3))
    return (pred - gt).norm(dim=2).mean(-1).mean()

def computeMPJPE_per_joint(pred, gt):
    pred = pred.reshape((pred.shape[0], FreiHandConfig.num_joints, 3))
    gt = gt.reshape((gt.shape[0], FreiHandConfig.num_joints, 3))
    #print((pred - gt).norm(dim=2).shape)
    return (pred - gt).norm(dim=2)


def _plot(joint_0_mpjpe, var_threshold, joint_0_var, index, thr_m):
    joint_0_mpjpe = np.array(joint_0_mpjpe)
    joint_0_var = np.array(joint_0_var)
    fig,ax = plt.subplots()
    ax.scatter(joint_0_var, joint_0_mpjpe, s = 0.4)
    ax.set_xlabel('variance')
    ax.set_ylabel('mpjpe')
    ax.set_xlim((0, 1e-4))
    fig.savefig('joint_{}_var_mpjpe.png'.format(index), dpi=fig.dpi)
    plt.close(fig)
    values = [] 
    values_less_than_threshold = []
    for th in var_threshold:  
        tmp = 0
        tmp2 = 0
        for i in range(len(joint_0_var)):
            if joint_0_var[i] < th:
                tmp2 +=1
            if joint_0_var[i] < th and joint_0_mpjpe[i] < thr_m:
                tmp+=1
        values.append(100.0*tmp/len(joint_0_var))
        values_less_than_threshold.append(100.0*tmp2/len(joint_0_var))
    values = np.array(values)
    values_less_than_threshold = np.array(values_less_than_threshold)
    ratio = 100*values/values_less_than_threshold
    fig,ax = plt.subplots()
    ax.scatter(var_threshold, values, s = 0.4)
    ax.set_xlabel('thresholds')
    ax.set_ylabel('num of samples < 0.005')
    ax.set_xlim((0, 1e-4))
    #fig.savefig('joint_{}_precision_{}.png'.format(index, thr_m), dpi=fig.dpi)
    plt.close(fig)    
    fig,ax = plt.subplots()
    ax.scatter(var_threshold, values_less_than_threshold, s = 0.4)
    ax.set_xlabel('thresholds')
    ax.set_ylabel('num of samples less than threshold')
    ax.set_xlim((0, 1e-4))
    #fig.savefig('joint_{}_precision_total_{}.png'.format(index, thr_m), dpi=fig.dpi)
    plt.close(fig)   
    fig,ax = plt.subplots()
    ax.scatter(var_threshold, ratio, s = 0.4)
    ax.set_xlabel('thresholds')
    ax.set_ylabel('ratio of samples than threshold with mpjpe < 5mm')
    ax.set_xlim((0, 1e-4))
    #fig.savefig('ratio_joint_{}_precision_total_{}.png'.format(index, thr_m), dpi=fig.dpi)
    plt.close(fig)
    return ratio 
    
def get_variance_measure(cfg):
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(cur_dir, '..', '..')
    common_dir = os.path.join(root_dir, 'common')
    main_dir = os.path.join(root_dir, 'main')
    util_dir = os.path.join(root_dir, 'common', 'utils')
    sys.path.insert(0, os.path.join(common_dir))
    sys.path.insert(0, os.path.join(main_dir))
    sys.path.insert(0, os.path.join(util_dir))
    cfg.batch_size = 1
    cfg.custom_batch_selection = False
    
    data_dir = os.path.join('..', 'data', 'FreiHand')
    name = "FreiHand"
    data_split= "testing"
    #cache_file = '{}_keypoint_bbox_db_{}_filtered.pkl'.format(name, data_split)
    #cache_file = os.path.join(data_dir, data_split, cache_file)
    trainer = Trainer(cfg)
    trainer._make_batch_generator()
    trainer._make_model()
    freihand = FreiHand(data_split="testing")
    data = freihand.load_data()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])
    i = 0
    joint_0_mpjpe = []
    joint_0_var = []
    
    joint_4_mpjpe = []
    joint_4_var = []
    
    joint_8_mpjpe = []
    joint_8_var = []
    
    joint_12_mpjpe = []
    joint_12_var = []
    
    joint_16_mpjpe = []
    joint_16_var = []
    
    joint_20_mpjpe = []
    joint_20_var = []
    var = [] 
    mpjpe = []
    for d in data:
        #if i % 1000 == 0:
        print("samples processed {}".format(i))
        if i > 1000:
            break
        K = d['K']
        joint_cam = d["joint_cam"]
        faster_rcnn_bbox = d['faster_rccn_bbox']
        img_path = d["img_path"]
        element = {
            "img_path": img_path,
            'K': K,
            'version': d['version'],
            "idx": d['idx'],
            "ref_bone_len": d['ref_bone_len'],
            "faster_rcnn_bbox": faster_rcnn_bbox,
            "joint_cam": joint_cam
        }
        cvimg = cv2.imread(d['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img_height, img_width, img_channels = cvimg.shape 
        r = np.arange(-0.52, 0.53, 0.05)
        stacked_predictions = np.zeros((21, 3, len(r)))
        j = 0
        for theta in r:
            scale, R, color_scale = get_aug_config(theta=theta)
            img_patch, trans, joint_img, joint_img_orig, joint_cam_normalized, joint_vis, xyz_rot, bbox, tprime = crop_and_get_patch(cvimg, joint_cam, K, R, scale, hand_detector=None, img_path = img_path,
                                                                                                                                    faster_rcnn_bbox = faster_rcnn_bbox)
            img_patch = transform(img_patch)
            for n_c in range(img_channels):
                img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
            
            img_patch = img_patch.cuda()
            img_patch = torch.unsqueeze(img_patch, 0)
            heatmap_out = trainer.teacher_network(img_patch)
            joint_num = 21
            hm_width = heatmap_out.shape[-1]
            hm_height = heatmap_out.shape[-2]
            hm_depth = heatmap_out.shape[-3] // FreiHandConfig.num_joints
            coord_in_patch = augment.get_joint_location_result(cfg.patch_width, cfg.patch_height, heatmap_out)
            coord_in_patch = np.squeeze(coord_in_patch)
            pre_3d_kpt = convert_to_cam_coord(coord_in_patch, bbox, scale, trans, tprime, K, R)
            stacked_predictions[:,:,j] = pre_3d_kpt
            j+=1
        stacked_predictions = np.array(stacked_predictions)
        variances = np.var(stacked_predictions, axis=2)
        variances_joint = np.sum(variances, axis=1)
        total_variance = np.sum(variances)
        average_predictions = np.mean(stacked_predictions, axis=2)
        average_predictions = np.expand_dims(average_predictions, 0)
        joint_cam_normalized = np.expand_dims(joint_cam_normalized, 0)
        mp = torch.squeeze(computeMPJPE_per_joint(torch.from_numpy(joint_cam_normalized), torch.from_numpy(average_predictions)))
        total_mpjpe = computeMPJPE(torch.from_numpy(joint_cam_normalized), torch.from_numpy(average_predictions))
        joint_0_mpjpe.append(mp[0])
        joint_0_var.append(variances_joint[0])

        joint_4_mpjpe.append(mp[4])
        joint_4_var.append(variances_joint[4])
        
        joint_12_mpjpe.append(mp[12])
        joint_12_var.append(variances_joint[12])
        
        joint_20_mpjpe.append(mp[20])
        joint_20_var.append(variances_joint[20])
        
        var.append(total_variance)
        mpjpe.append(total_mpjpe)
        i+=1
    
    
    print("=======================")
    print(len(joint_0_var))
    var_threshold_2= np.arange(0, 1e-3, 1e-5)
    ratio_5 = _plot(mpjpe, var_threshold_2, var, 0, 0.005)
    ratio_7 = _plot(mpjpe, var_threshold_2, var, 0, 0.007)
    ratio_10 = _plot(mpjpe, var_threshold_2, var, 0, 0.010)
    fig,ax = plt.subplots()
    p1 = ax.scatter(var_threshold_2, ratio_5, s = 0.4)
    p2 = ax.scatter(var_threshold_2, ratio_7, s = 0.4)
    p3 = ax.scatter(var_threshold_2, ratio_10, s = 0.4)
    ax.set_xlabel('variance thresholds')
    ax.set_ylabel('ratio of samples than threshold with mpjpe < x mm')
    ax.set_xlim((0, 1e-3))
    plt.legend((p1, p2, p3),
               ('< 5mm', '< 7mm', '< 10 mm'),
               scatterpoints=1,
               loc='upper right',
               ncol=3,
               fontsize=12)
    fig.savefig('ratio_sum_var_precision.png', dpi=fig.dpi)
    plt.close(fig)

    
    var_threshold= np.arange(0, 1e-4, 5e-7)
    
    ratio_0_5 = _plot(joint_0_mpjpe, var_threshold, joint_0_var, 0, 0.005)
    ratio_4_5 = _plot(joint_4_mpjpe, var_threshold,joint_4_var, 4, 0.005)
    ratio_12_5 = _plot(joint_12_mpjpe, var_threshold,joint_12_var, 12, 0.005)
    ratio_20_5 = _plot(joint_20_mpjpe, var_threshold,joint_20_var, 20, 0.005)

    ratio_0_7 = _plot(joint_0_mpjpe, var_threshold,joint_0_var, 0, 0.007)
    ratio_4_7 =_plot(joint_4_mpjpe, var_threshold,joint_4_var, 4, 0.007)
    ratio_12_7 =_plot(joint_12_mpjpe, var_threshold,joint_12_var, 12, 0.007)
    ratio_20_7 =_plot(joint_20_mpjpe, var_threshold,joint_20_var, 20, 0.007)          
    
    ratio_0_10 =_plot(joint_0_mpjpe, var_threshold, joint_0_var, 0, 0.010)
    ratio_4_10 =_plot(joint_4_mpjpe, var_threshold, joint_4_var, 4, 0.010)
    ratio_12_10 =_plot(joint_12_mpjpe, var_threshold, joint_12_var, 12, 0.010)
    ratio_20_10 =_plot(joint_20_mpjpe, var_threshold, joint_20_var, 20, 0.010)
    
    fig,ax = plt.subplots()
    p1 = ax.scatter(var_threshold, ratio_0_5, s = 0.4)
    p2 = ax.scatter(var_threshold, ratio_4_5, s = 0.4)
    p3 = ax.scatter(var_threshold, ratio_12_5, s = 0.4)
    p4 = ax.scatter(var_threshold, ratio_20_5, s = 0.4)
    ax.set_xlabel('variance thresholds')
    ax.set_ylabel('ratio of samples than threshold with mpjpe < 5mm')
    ax.set_xlim((0, 1e-4))
    plt.legend((p1, p2, p3, p4),
               ('joint 1', 'joint 4', 'joint 7', 'joint 20'),
               scatterpoints=1,
               loc='upper right',
               ncol=3,
               fontsize=12)
    fig.savefig('ratio_joint_precision_combined_{}.png'.format(5), dpi=fig.dpi)
    plt.close(fig)
    
    fig,ax = plt.subplots()
    p1 =ax.scatter(var_threshold, ratio_0_7, s = 0.4)
    p2 =ax.scatter(var_threshold, ratio_4_7, s = 0.4)
    p3 =ax.scatter(var_threshold, ratio_12_7, s = 0.4)
    p4 =ax.scatter(var_threshold, ratio_20_7, s = 0.4)
    ax.set_xlabel('variance thresholds')
    ax.set_ylabel('ratio of samples than threshold with mpjpe < 7mm')
    ax.set_xlim((0, 1e-4))
    plt.legend((p1, p2, p3, p4),
               ('joint 1', 'joint 4', 'joint 7', 'joint 20'),
               scatterpoints=1,
               loc='upper right',
               ncol=3,
               fontsize=12)
    fig.savefig('ratio_joint_precision_combined_{}.png'.format(7), dpi=fig.dpi)
    plt.close(fig)
    
    fig,ax = plt.subplots()
    p1 = ax.scatter(var_threshold, ratio_0_10, s = 0.4)
    p2 = ax.scatter(var_threshold, ratio_4_10, s = 0.4)
    p3 = ax.scatter(var_threshold, ratio_12_10, s = 0.4)
    p4 = ax.scatter(var_threshold, ratio_20_10, s = 0.4)
    ax.set_xlabel('variance thresholds')
    ax.set_ylabel('ratio of samples than threshold with mpjpe < 10mm')
    ax.set_xlim((0, 1e-4))
    plt.legend((p1, p2, p3, p4),
               ('joint 1', 'joint 4', 'joint 7', 'joint 20'),
               scatterpoints=1,
               loc='upper right',
               ncol=3,
               fontsize=12)
    fig.savefig('ratio_joint_precision_combined_{}.png'.format(10), dpi=fig.dpi)
    plt.close(fig)

    
    sys.exit()

def main():
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    # Refer to https://github.com/soumith/cudnn.torch for a detailed 
    # explanation of these parameters
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(cur_dir, '..', '..')
    common_dir = os.path.join(root_dir, 'common')
    main_dir = os.path.join(root_dir, 'main')
    util_dir = os.path.join(root_dir, 'common', 'utils')
    sys.path.insert(0, os.path.join(common_dir))
    sys.path.insert(0, os.path.join(main_dir))
    sys.path.insert(0, os.path.join(util_dir))
    cfg.batch_size = 1
    cfg.custom_batch_selection = False
    get_variance_measure(cfg)
    data_dir = os.path.join('..', 'data', 'FreiHand')
    name = "FreiHand"
    data_split= "training"
    cache_file = '{}_keypoint_bbox_db_{}_filtered.pkl'.format(name, data_split)
    cache_file = os.path.join(data_dir, data_split, cache_file)
    trainer = Trainer(cfg)
    trainer._make_batch_generator()
    trainer._make_model()
    freihand = FreiHand(data_split="training")
    data = freihand.load_data()
    kept_data = []
    preds_in_patch_with_score = []
    i = 0
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=cfg.pixel_mean, std=cfg.pixel_std)])
    for d in data:
        if i % 1000 == 0:
            print("samples processed {}".format(i))
        K = d['K']
        joint_cam = d["joint_cam"]
        faster_rcnn_bbox = d['faster_rccn_bbox']
        img_path = d["img_path"]
        element = {
            "img_path": img_path,
            'K': K,
            'version': d['version'],
            "idx": d['idx'],
            "ref_bone_len": d['ref_bone_len'],
            "faster_rcnn_bbox": faster_rcnn_bbox,
            "joint_cam": joint_cam
        }
        cvimg = cv2.imread(d['img_path'], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        img_height, img_width, img_channels = cvimg.shape 
        if d['labelled']:
            element['labelled'] = True
            scale, R, color_scale = get_aug_config(theta=0) 
            img_patch, trans, joint_img, joint_img_orig, joint_cam_normalized, joint_vis, xyz_rot, bbox, tprime = crop_and_get_patch(cvimg, joint_cam, K, R, scale,
                                                                                                                                    hand_detector=None, 
                                                                                                                                    img_path = img_path,
                                                                                                                                    faster_rcnn_bbox = faster_rcnn_bbox)
            element['joint_cam_normalized'] = joint_cam_normalized
            element['tprime'] = tprime
            element['tprime_torch'] = torch.from_numpy(np.array([tprime]))
            element['variance'] = 0
        else:
            r = np.arange(-0.52, 0.53, 0.05)
            stacked_predictions = np.zeros((21, 3, len(r)))
            j = 0
            for theta in r:
                scale, R, color_scale = get_aug_config(theta=theta)
                img_patch, trans, _, _, joint_cam_normalized, joint_vis, xyz_rot, bbox, tprime = crop_and_get_patch(cvimg, joint_cam, K, R, scale, hand_detector=None, img_path = img_path,
                                                                                                faster_rcnn_bbox = faster_rcnn_bbox)
                img_patch = transform(img_patch)
                for n_c in range(img_channels):
                    img_patch[n_c, :, :] = np.clip(img_patch[n_c, :, :] * color_scale[n_c], 0, 255)
                
                img_patch = img_patch.cuda()
                img_patch = torch.unsqueeze(img_patch, 0)
                heatmap_out = trainer.teacher_network(img_patch)
                joint_num = 21
                hm_width = heatmap_out.shape[-1]
                hm_height = heatmap_out.shape[-2]
                hm_depth = heatmap_out.shape[-3] // FreiHandConfig.num_joints
                coord_in_patch = augment.get_joint_location_result(cfg.patch_width, cfg.patch_height, heatmap_out)
                coord_in_patch = np.squeeze(coord_in_patch)
                pre_3d_kpt = convert_to_cam_coord(coord_in_patch, bbox, scale, trans, tprime, K, R)
                stacked_predictions[:,:,j] = pre_3d_kpt
                j+=1
            stacked_predictions = np.array(stacked_predictions)
            variances = np.var(stacked_predictions, axis=2)
            average_predictions = np.mean(stacked_predictions, axis=2)
            m = np.sum(variances)
            if m > 1e-4:
                continue
            element['joint_cam_normalized'] = average_predictions
            element['tprime'] = tprime
            element['tprime_torch'] = torch.from_numpy(np.array([tprime]))
            element['labelled'] = False
            element['variance'] = m
            average_predictions = np.expand_dims(average_predictions, 0)
            joint_cam_normalized = np.expand_dims(joint_cam_normalized, 0)
            mp = computeMPJPE(torch.from_numpy(joint_cam_normalized), torch.from_numpy(average_predictions))
            print(m, mp)
        kept_data.append(element)
        i+=1
    with open(cache_file, 'wb') as fid:
        pk.dump(kept_data, fid, pk.HIGHEST_PROTOCOL)
    print('{} samples read wrote {}'.format(len(kept_data), cache_file))


if __name__ == "__main__":
    main()



#===============================================================
# label, label_weight = generate_label(joint_img, trans, joint_vis, p=False)
# label = np.expand_dims(label, 0)
# label_patch = augment.test_get_joint_loc_res(label)
# label_patch = np.squeeze(label_patch)
# label_3d = convert_to_cam_coord(label_patch, bbox, scale, trans, tprime, K, R)
# assert np.allclose(label_3d, joint_cam_normalized, rtol=1e-6, atol=1e-6)
#===============================================================


#print("============================================")

#===================================================================
# print("=======================================")
# print(m)
# print(mp)
#===================================================================
#print(torch.from_numpy(average_predictions))
#print(torch.from_numpy(joint_cam_normalized))
#print(average_predictions.shape)
#sys.exit()    