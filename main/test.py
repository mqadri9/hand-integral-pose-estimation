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
    params_list = {
        "R": [],
        "cvimg": [],
        "K": [],
        "joint_cam": [],
        "joint_cam_normalized": [],
        "scale": [],
        "img_path": [],
        "loss": [],
        "ref_bone_len": [],
        "bbox": [],
        'tprime': []
    }
    with torch.no_grad():
        for itr, (img_patch, params) in enumerate(tqdm(tester.batch_generator)):
            img_patch = img_patch.cuda()
            img_patch_sav = np.copy(img_patch[0].permute(1, 2, 0).cpu().detach().numpy())
            label = params["label"].cuda()
            label_weight = params["label_weight"].cuda()
            bbox = params["bbox"].cuda()
            scale = params["scale"].cuda()
            tprime = params["tprime_torch"].cuda()
            R = params["R"].cuda()
            labelled = params["labelled"].cuda()
            trans = params["trans"].cuda()
            K = params["K"].cuda()
            joint_cam = params["joint_cam"].cuda()
            joint_cam_normalized = params["joint_cam_normalized"].cuda()
            joint_img_orig = params["joint_img_orig"][0].numpy()
            #===================================================================
            # tester.model.train()  
            # for i in range(50):
            #     with torch.no_grad():
            #         _ = tester.model(img_patch)
            # tester.model.eval()   
            #===================================================================
            heatmap_out = tester.model(img_patch)
            if cfg.num_gpus > 1:
                heatmap_out = gather(heatmap_out,0)
            
            JointLocationLoss = tester.JointLocationLoss(heatmap_out, label, label_weight)
            JointLocationLoss2 = tester.JointLocationLoss2(heatmap_out, label, label_weight, joint_cam, 
                                                           joint_cam_normalized, bbox, scale, R, trans, K, tprime)
            loss1 = JointLocationLoss.detach()
            #===================================================================
            # if loss1 > 10:
            #     print("Loss loc2 {}".format(JointLocationLoss2.detach()))
            #     print("Loss loc {}".format(JointLocationLoss.detach()))
            #     print(augmentation['img_path'][0])
            #     fig = plt.figure()
            #     cvimg = cv2.imread(augmentation['img_path'][0], cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
            #     ax1 = fig.add_subplot(121)
            #     ax2 = fig.add_subplot(122)
            #     # 
            #     ax1.imshow((255*img_patch_sav/np.max(img_patch_sav)).astype(np.uint8))
            #     ax2.imshow(cvimg)
            #     #ax1.imshow(img2_w)
            #     FreiHand.plot_hand(ax2, joint_img_orig[:, 0:2], order='uv')
            #     FreiHand.plot_hand(ax1, joint_img[:, 0:2], order='uv')
            #     
            #     ax1.axis('off')
            #     nn = str(random.randint(1,3000))
            #     #print("=============================================================")
            #     #print(nn)
            #     plt.savefig('/home/mqadri/hand-integral-pose-estimation/tests/{}.jpg'.format(nn))
            #===================================================================
            
            
            hm_width = heatmap_out.shape[-1]
            hm_height = heatmap_out.shape[-2]
            hm_depth = heatmap_out.shape[-3] // FreiHandConfig.num_joints
            coord_out = softmax_integral_tensor(heatmap_out, tester.joint_num, hm_width, hm_height, hm_depth)
            coord_out = coord_out.cpu().numpy()
            preds.append(coord_out)
            params_list["R"].append(params["R"])
            #params_list["cvimg"].append(params["cvimg"])
            params_list["K"].append(params["K"])
            params_list["joint_cam"].append(params["joint_cam"])
            params_list["joint_cam_normalized"].append(params["joint_cam_normalized"])
            params_list["scale"].append(params["scale"])
            params_list["img_path"].append(params["img_path"])
            params_list["ref_bone_len"].append(params["ref_bone_len"])
            params_list["bbox"].append(params["bbox"])
            params_list["tprime"].append(params["tprime_torch"].data.cpu().numpy())
            params_list["loss"].append(JointLocationLoss.detach())
            preds_in_patch_with_score.append(augment.get_joint_location_result(cfg.patch_width, cfg.patch_height, heatmap_out))
            #print(data['label'].cpu().detach().numpy().shape)
            label_list.append(augment.test_get_joint_loc_res(label.cpu().detach().numpy()))
            #if itr >=0:
            #    break
            
    preds = np.concatenate(preds, axis=0)
    _p = np.concatenate(preds_in_patch_with_score, axis=0)
    _p_label = np.concatenate(label_list, axis=0) 
    #print(_p.shape)
    #_p = _p.reshape((_p.shape[0] * _p.shape[1], _p.shape[2], _p.shape[3]))
    preds_in_patch_with_score = _p[0: tester.num_samples]
    label_list = _p_label[0: tester.num_samples]
    #print(np.array(preds_in_patch_with_score).shape)
    
    params_list["R"] = np.concatenate(params_list["R"], axis=0)
    #params_list["cvimg"] = np.concatenate(params_list["cvimg"], axis=0)
    params_list["K"] = np.concatenate(params_list["K"], axis=0)
    #params_list["bbox"] = np.concatenate(params_list["bbox"], axis=0)
    params_list["joint_cam"] = np.concatenate(params_list["joint_cam"], axis=0)
    params_list["joint_cam_normalized"] = np.concatenate(params_list["joint_cam_normalized"], axis=0)
    params_list["scale"] = np.concatenate(params_list["scale"], axis=0)
    params_list["img_path"] = np.concatenate(params_list["img_path"], axis=0)
    params_list["ref_bone_len"] = np.concatenate(params_list["ref_bone_len"], axis=0)
    params_list["bbox"] = np.concatenate(params_list["bbox"], axis=0)
    params_list["tprime"] = np.concatenate(params_list["tprime"], axis=0)
    
    tester._evaluate(preds_in_patch_with_score, label_list, params_list, cfg.result_dir) 

if __name__ == "__main__":
    main()
    


