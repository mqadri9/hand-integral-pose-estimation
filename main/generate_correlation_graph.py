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
import os
import matplotlib.pyplot as plt 
from scipy import stats, optimize, interpolate

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

def computeMPJPE(pred, gt):
    pred = pred.reshape((pred.shape[0], FreiHandConfig.num_joints, 3))
    gt = gt.reshape((gt.shape[0], FreiHandConfig.num_joints, 3))
    return (pred - gt).norm(dim=2).mean(-1).mean()

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
    trainer = Trainer(cfg)
    trainer._make_batch_generator()
    trainer._make_model()
    trainer.model.eval()
    p = []
    g = []
    y = []
    entropy_joint_0 = []
    entropy_joint_4 = []
    entropy_joint_8 = []
    entropy_joint_12 = []
    entropy_joint_16 = []
    entropy_joint_20 = []
    entropy_sum_joints = []
    for itr, (img_patch, params) in enumerate(trainer.batch_generator):
        trainer.optimizer.zero_grad()
        img_patch = img_patch.cuda()
        label = params["label"].cuda()
        label_weight = params["label_weight"].cuda()
        labelled = params["labelled"].cuda()
        if labelled.item():
            continue
        trans = params["trans"].cuda()
        bbox = params["bbox"].cuda()
        K = params["K"].cuda()
        scale = params["scale"].cuda()
        joint_cam_normalized = params["joint_cam_normalized"].cuda()
        tprime = params["tprime_torch"].cuda()
        R = params["R"].cuda()
        heatmap_out = trainer.teacher_network(img_patch)
        joint_num = 21
        hm_width = heatmap_out.shape[-1]
        hm_height = heatmap_out.shape[-2]
        hm_depth = heatmap_out.shape[-3] // FreiHandConfig.num_joints
        preds = softmax_integral_tensor2(heatmap_out, joint_num, hm_width, hm_height, hm_depth)
        #print(heatmap_out.detach().cpu().numpy().flatten())
        v = preds.detach().cpu().numpy()
        v = np.squeeze(v)
        tmp = 0
        for j in range(v.shape[0]):
            tmp += stats.entropy(v[j, :])
        entropy_sum_joints.append(tmp)
        
        entropy0 = stats.entropy(v[0, :])
        entropy_joint_0.append(entropy0)
        
        entropy4 = stats.entropy(v[4, :])
        entropy_joint_4.append(entropy4)
        
        entropy8 = stats.entropy(v[8, :])
        entropy_joint_8.append(entropy8)
        
        entropy12 = stats.entropy(v[12, :])
        entropy_joint_12.append(entropy12)
        
        entropy16 = stats.entropy(v[16, :])
        entropy_joint_16.append(entropy16)
        
        entropy20 = stats.entropy(v[20, :])
        entropy_joint_20.append(entropy20)

        coord_out_teacher = softmax_integral_tensor(heatmap_out, joint_num, hm_width, hm_height, hm_depth)
        input_to_panet = coord_out_teacher.reshape((coord_out_teacher.shape[0], FreiHandConfig.num_joints, 3))        
        input_to_panet = augment.prepare_panet_input(input_to_panet, tprime, trans, bbox, K, R, scale)
        panet_output,_ ,_, _ = trainer.nrsfm_tester.forward(input_to_panet)

        label = label.reshape((label.shape[0], FreiHandConfig.num_joints, 3))
        gt_norm = augment.prepare_panet_input(label, tprime, trans, bbox, K, R, scale)
        
        gt_teacher = computeMPJPE(input_to_panet, gt_norm)
        panet_teacher = computeMPJPE(panet_output, input_to_panet)
        panet_gt = computeMPJPE(panet_output, gt_norm)
        
        p.append(panet_teacher.item())
        g.append(gt_teacher.item())
        y.append(panet_gt.item())
        #if itr % 1000 == 0:
        if itr > 6000:
            break
        
    p = np.array(p)
    g = np.array(g)
    fig,ax = plt.subplots()
    ax.scatter(g,p, s = 0.4)
    ax.set_xlabel('teacher_output - gt_norm')
    ax.set_ylabel('panet_output - teacher_output')
    fig.savefig('temp.png', dpi=fig.dpi)

    fig,ax = plt.subplots()
    ax.scatter(g,y, s = 0.4)
    ax.set_xlabel('teacher_output - gt_norm')
    ax.set_ylabel('panet_output - gt')
    fig.savefig('temp2.png', dpi=fig.dpi)

    fig,ax = plt.subplots()
    ax.scatter(g, entropy_joint_0, s = 0.4)
    ax.set_xlabel('teacher_output - gt_norm')
    ax.set_ylabel('teacher heatmap entropy joint 0')
    fig.savefig('joint_0.png', dpi=fig.dpi)
 
    fig,ax = plt.subplots()
    ax.scatter(g, entropy_joint_4, s = 0.4)
    ax.set_xlabel('teacher_output - gt_norm')
    ax.set_ylabel('teacher heatmap entropy joint 4')
    fig.savefig('joint_4.png', dpi=fig.dpi)
    
    fig,ax = plt.subplots()
    ax.scatter(g, entropy_joint_8, s = 0.4)
    ax.set_xlabel('teacher_output - gt_norm')
    ax.set_ylabel('teacher heatmap entropy joint 8')
    fig.savefig('joint_8.png', dpi=fig.dpi)
    
    fig,ax = plt.subplots()
    ax.scatter(g, entropy_joint_12, s = 0.4)
    ax.set_xlabel('teacher_output - gt_norm')
    ax.set_ylabel('teacher heatmap entropy joint 12')
    fig.savefig('joint_12.png', dpi=fig.dpi)
    
    fig,ax = plt.subplots()
    ax.scatter(g, entropy_joint_16, s = 0.4)
    ax.set_xlabel('teacher_output - gt_norm')
    ax.set_ylabel('teacher heatmap entropy joint 16')
    fig.savefig('joint_16.png', dpi=fig.dpi)
    
    fig,ax = plt.subplots()
    ax.scatter(g, entropy_joint_20, s = 0.4)
    ax.set_xlabel('teacher_output - gt_norm')
    ax.set_ylabel('teacher heatmap entropy joint 20')
    fig.savefig('joint_20.png', dpi=fig.dpi)

    fig,ax = plt.subplots()
    ax.scatter(g, entropy_sum_joints, s = 0.4)
    ax.set_xlabel('teacher_output - gt_norm')
    ax.set_ylabel('sum entropy all joints')
    fig.savefig('sum_entropy_joints.png', dpi=fig.dpi)
                   
    np.save("p.npy", np.array(p))
    np.save("g.npy", np.array(g))
        
if __name__ == "__main__":
    main()