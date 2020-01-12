import os, sys
import argparse
from tqdm import tqdm
import numpy as np
import cv2
from config import cfg
import torch
from base import Tester
from torch.nn.parallel.scatter_gather import gather
from nets.loss import softmax_integral_tensor, JointLocationLoss
#from utils.vis import vis_keypoints
#from utils.pose_utils import flip
import torch.backends.cudnn as cudnn
import augment
from FreiHand_config import FreiHandConfig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        args.gpu_ids = str(np.argmin(mem_info()))

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    assert args.test_epoch, 'Test epoch is required.'
    return args

def main():
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    
    tester = Tester(cfg, args.test_epoch)
    tester._make_batch_generator()
    tester._make_model()
    preds = []
    preds_in_patch_with_score = []
    label_list = []
    augmentation_list = {
        "R": [],
        "cvimg": [],
        "K": [],
        "joint_cam": [],
        "scale": [],
        "img_path": []
    }
    with torch.no_grad():
        for itr, data in enumerate(tqdm(tester.batch_generator)):
            input_img = data["img_patch"].cuda()
            label = data["label"].cuda()
            label_weight = data["label_weight"].cuda()
            heatmap_out = tester.model(input_img)
            if cfg.num_gpus > 1:
                heatmap_out = gather(heatmap_out,0)
            JointLocationLoss = tester.JointLocationLoss(heatmap_out, label, label_weight)
            #JointLocationLoss2 = tester.JointLocationLoss2(heatmap_out, label, label_weight, data['augmentation'])
            #print("Loss loc2 {}".format(JointLocationLoss2.detach()))
            print("Loss loc {}".format(JointLocationLoss.detach()))
            hm_width = heatmap_out.shape[-1]
            hm_height = heatmap_out.shape[-2]
            hm_depth = heatmap_out.shape[-3] // FreiHandConfig.num_joints
            coord_out = softmax_integral_tensor(heatmap_out, tester.joint_num, hm_width, hm_height, hm_depth)
            coord_out = coord_out.cpu().numpy()
            preds.append(coord_out)
            augmentation_list["R"].append(data["augmentation"]["R"])
            augmentation_list["cvimg"].append(data["augmentation"]["cvimg"])
            augmentation_list["K"].append(data["augmentation"]["K"])
            augmentation_list["joint_cam"].append(data["augmentation"]["joint_cam"])
            augmentation_list["scale"].append(data["augmentation"]["scale"])
            augmentation_list["img_path"].append(data["augmentation"]["img_path"])
            preds_in_patch_with_score.append(augment.get_joint_location_result(cfg.patch_width, cfg.patch_height, heatmap_out))
            #print(data['label'].cpu().detach().numpy().shape)
            label_list.append(augment.test_get_joint_loc_res(data['label'].cpu().detach().numpy()))
            
    preds = np.concatenate(preds, axis=0)
    _p = np.concatenate(preds_in_patch_with_score, axis=0)
    _p_label = np.concatenate(label_list, axis=0)
    #print(_p.shape)
    #_p = _p.reshape((_p.shape[0] * _p.shape[1], _p.shape[2], _p.shape[3]))
    preds_in_patch_with_score = _p[0: tester.num_samples]
    label_list = _p_label[0: tester.num_samples]
    #print(np.array(preds_in_patch_with_score).shape)
    
    augmentation_list["R"] = np.concatenate(augmentation_list["R"], axis=0)
    augmentation_list["cvimg"] = np.concatenate(augmentation_list["cvimg"], axis=0)
    augmentation_list["K"] = np.concatenate(augmentation_list["K"], axis=0)
    augmentation_list["joint_cam"] = np.concatenate(augmentation_list["joint_cam"], axis=0)
    augmentation_list["scale"] = np.concatenate(augmentation_list["scale"], axis=0)
    augmentation_list["img_path"] = np.concatenate(augmentation_list["img_path"], axis=0)
    
    tester._evaluate(preds_in_patch_with_score, label_list, augmentation_list, cfg.result_dir) 

if __name__ == "__main__":
    main()
    


