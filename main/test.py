import os
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
    with torch.no_grad():
        for itr, (input_img, label, label_weight) in enumerate(tqdm(tester.batch_generator)):
            input_img = input_img.cuda()
            label = label.cuda()
            label_weight = label_weight.cuda()
            heatmap_out = tester.model(input_img)
            if cfg.num_gpus > 1:
                heatmap_out = gather(heatmap_out,0)
            JointLocationLoss = tester.JointLocationLoss(heatmap_out, label, label_weight)
            print("Loss loc {}".format(JointLocationLoss.detach()))
            coord_out = softmax_integral_tensor(heatmap_out, tester.joint_num, cfg.output_shape[0], cfg.output_shape[1], cfg.depth_dim)
            coord_out = coord_out.cpu().numpy()
            label = label.cpu().numpy()
            label_weight = label_weight.cpu().numpy()
            preds.append(coord_out)
            preds_in_patch_with_score.append(augment.get_joint_location_result(cfg.patch_width, cfg.patch_height, heatmap_out))
    preds = np.concatenate(preds, axis=0)
    _p = np.concatenate(preds_in_patch_with_score, axis=0)
    #print(_p.shape)
    #_p = _p.reshape((_p.shape[0] * _p.shape[1], _p.shape[2], _p.shape[3]))
    preds_in_patch_with_score = _p[0: tester.num_samples]
    tester._evaluate(preds_in_patch_with_score, cfg.result_dir) 

if __name__ == "__main__":
    main()
    


