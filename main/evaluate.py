import os, sys
import argparse
from tqdm import tqdm
import numpy as np
import cv2
from config import cfg
import torch
from base import Evaluator
from torch.nn.parallel.scatter_gather import gather
from nets.loss import softmax_integral_tensor, JointLocationLoss
from FreiHand import FreiHand
#from utils.vis import vis_keypoints
#from utils.pose_utils import flip
import torch.backends.cudnn as cudnn
import augment
from FreiHand_config import FreiHandConfig
import matplotlib.pyplot as plt
import random

plt.switch_backend('agg')
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--evaluate_epoch', type=str, dest='evaluate_epoch')
    args = parser.parse_args()

    # test gpus
    if not args.gpu_ids:
        args.gpu_ids = str(np.argmin(mem_info()))

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = 0 if not gpus[0].isdigit() else int(gpus[0])
        gpus[1] = len(mem_info()) if not gpus[1].isdigit() else int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))
    
    assert args.evaluate_epoch, 'evaluate epoch is required.'
    return args

def main():
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.fastest = True
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    evaluator = Evaluator(cfg, args.evaluate_epoch)
    evaluator._make_batch_generator()
    evaluator._make_model()
    params = {
        "K": [],
        "img_path": [],
        "ref_bone_len": [],
        "bbox": [],
        "tprime": []
    }
    preds = []
    preds_in_patch_with_score = []
    with torch.no_grad():
        for itr, (img_patch, data) in enumerate(tqdm(evaluator.batch_generator)):
            heatmap_out = evaluator.model(img_patch)
            if cfg.num_gpus > 1:
                heatmap_out = gather(heatmap_out,0)
            hm_width = heatmap_out.shape[-1]
            hm_height = heatmap_out.shape[-2]
            hm_depth = heatmap_out.shape[-3] // FreiHandConfig.num_joints
            coord_out = softmax_integral_tensor(heatmap_out, evaluator.joint_num, hm_width, hm_height, hm_depth)
            coord_out = coord_out.cpu().numpy()
            preds.append(coord_out)
            params["ref_bone_len"].append(data["ref_bone_len"])
            params["tprime"].append(data["tprime"].cpu().detach().numpy())
            params["K"].append(data["K"])
            params["img_path"].append(data["img_path"])
            params["bbox"].append(data["bbox"])
            preds_in_patch_with_score.append(augment.get_joint_location_result(cfg.patch_width, cfg.patch_height, heatmap_out))
    
    preds = np.concatenate(preds, axis=0)
    _p = np.concatenate(preds_in_patch_with_score, axis=0)
    preds_in_patch_with_score = _p[0: evaluator.num_samples]
    
    params["ref_bone_len"] = np.concatenate(params["ref_bone_len"], axis=0)
    params["tprime"] = np.concatenate(params["tprime"], axis=0)
    params["K"] = np.concatenate(params["K"], axis=0)
    params["img_path"] = np.concatenate(params["img_path"], axis=0)
    params["bbox"] = np.concatenate(params["bbox"], axis=0)
    
    evaluator._evaluate(preds_in_patch_with_score, params, cfg.result_dir) 

if __name__ == "__main__":
    main()
    


