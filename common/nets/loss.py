import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import cfg
from FreiHand_config import FreiHandConfig

def _assert_no_grad(tensor):
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

def generate_3d_integral_preds_tensor(heatmaps, num_joints, x_dim, y_dim, z_dim):
    assert isinstance(heatmaps, torch.Tensor)
    heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, z_dim, y_dim, x_dim))
    # heatmaps are of size torch.Size([24, 21, 64, 64, 64])
    accu_x = heatmaps.sum(dim=2)
    #print(accu_x.shape) # torch.Size([24, 21, 64, 64])
    accu_x = accu_x.sum(dim=2)
    #print(accu_x.shape) # torch.Size([24, 21, 64])
    accu_y = heatmaps.sum(dim=2)
    accu_y = accu_y.sum(dim=3)
    accu_z = heatmaps.sum(dim=3)
    accu_z = accu_z.sum(dim=3)
    accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(x_dim), devices=[accu_x.device.index])[0]
    accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(y_dim), devices=[accu_y.device.index])[0]
    accu_z = accu_z * torch.cuda.comm.broadcast(torch.arange(z_dim), devices=[accu_z.device.index])[0]

    accu_x = accu_x.sum(dim=2, keepdim=True)
    accu_y = accu_y.sum(dim=2, keepdim=True)
    accu_z = accu_z.sum(dim=2, keepdim=True)
    #print(accu_x.shape) # torch.Size([24, 21, 1])
    return accu_x, accu_y, accu_z


def softmax_integral_tensor(preds, num_joints, hm_width, hm_height, hm_depth):
    # global soft max
    preds = preds.reshape((preds.shape[0], num_joints, -1)) # preds of size [batch_size x 21 x 262144] 26144 = 64x64x64
    preds = F.softmax(preds, 2)
    # integrate heatmap into joint location
    x, y, z = generate_3d_integral_preds_tensor(preds, num_joints, hm_width, hm_height, hm_depth)
    # size of x, y, z is batch_size x 21 x1 

    x = x / float(hm_width) - 0.5
    y = y / float(hm_height) - 0.5
    z = z / float(hm_depth) - 0.5
    preds = torch.cat((x, y, z), dim=2) # preds is now of shape batch_size x 21 x 3
    preds = preds.reshape((preds.shape[0], num_joints * 3)) # preds is of shape batch_size x 63
    return preds

class JointLocationLoss(nn.Module):
    def __init__(self):
        super(JointLocationLoss, self).__init__()
        self.size_average = True

    def forward(self, heatmap_out, gt_coord, gt_vis):
        
        joint_num = int(gt_coord.shape[1]/3)
        
        hm_width = heatmap_out.shape[-1]
        hm_height = heatmap_out.shape[-2]
        hm_depth = heatmap_out.shape[-3] // FreiHandConfig.num_joints
        
        coord_out = softmax_integral_tensor(heatmap_out, joint_num, hm_width, hm_height, hm_depth)
        
        _assert_no_grad(gt_coord)
        _assert_no_grad(gt_vis)
        #=======================================================================
        # print(coord_out.shape)
        # print(gt_coord.shape)
        # print(gt_vis.shape)
        #=======================================================================
        #=======================================================================
        # print("coord_out")
        # print(coord_out.shape)
        # print("gt_coord")
        # print(gt_coord.shape)
        # print("gt_vis")
        # print(gt_vis.shape)
        #=======================================================================
        loss = torch.abs(coord_out - gt_coord) * gt_vis
        if self.size_average:
            return loss.sum() / len(coord_out)
        else:
            return loss.sum()
