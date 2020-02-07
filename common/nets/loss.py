import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from config import cfg
from FreiHand_config import FreiHandConfig
import augment
import config_panet

def _assert_no_grad(tensor):
    if type(tensor) is not "torch.Tensor":
        return
    assert not tensor.requires_grad, \
        "nn criterions don't compute the gradient w.r.t. targets - please " \
        "mark these tensors as not requiring gradients"

def _assert_grad(tensor):
    if type(tensor) is not "torch.Tensor":
        return
    assert tensor.requires_grad, \
        "Make sure that grad is set to true for this tensor"
        
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

class ParallelSoftmaxIntegralTensor(nn.Module):
    def __init__(self):
        super(ParallelSoftmaxIntegralTensor, self).__init__()
        self.size_average = True    
    def forward(self, heatmap_out, gt_coord):
        joint_num = int(gt_coord.shape[1]/3)
        #print(heatmap_out.shape)
        hm_width = heatmap_out.shape[-1]
        hm_height = heatmap_out.shape[-2]
        hm_depth = heatmap_out.shape[-3] // FreiHandConfig.num_joints
        coord_out = softmax_integral_tensor(heatmap_out, joint_num, hm_width, hm_height, hm_depth)        
        return coord_out


class JointLocationLoss(nn.Module):
    def __init__(self):
        super(JointLocationLoss, self).__init__()
        self.size_average = True

    def forward(self, heatmap_out, gt_coord, gt_vis):
        
        joint_num = int(gt_coord.shape[1]/3)
        #print(heatmap_out.shape)
        hm_width = heatmap_out.shape[-1]
        hm_height = heatmap_out.shape[-2]
        hm_depth = heatmap_out.shape[-3] // FreiHandConfig.num_joints
        coord_out = softmax_integral_tensor(heatmap_out, joint_num, hm_width, hm_height, hm_depth)
        
        _assert_no_grad(gt_coord)
        _assert_no_grad(gt_vis)
        
        loss = torch.abs(coord_out - gt_coord) * gt_vis

        if self.size_average:
            return loss.sum() / len(coord_out)
        else:
            return loss.sum()
        
def computeMPJPE(pred, gt):
    pred = pred.reshape((pred.shape[0], FreiHandConfig.num_joints, 3))
    gt = gt.reshape((gt.shape[0], FreiHandConfig.num_joints, 3))
    return (pred - gt).norm(dim=2).mean(-1).mean()

 
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.size_average = True
    
    def forward(self, heatmap_out, coord_out_teacher, gt_coord, gt_vis, labelled, tprime, trans, bbox, K, R, scale, 
                joint_cam_normalized, nrsfm_tester):
        #=======================================================================
        # print("====================================")
        # print(heatmap_out.shape)
        # 
        # print(coord_out_teacher.shape)
        # print(gt_coord.shape)
        # print(gt_vis.shape)
        # print(labelled.shape)  
        # print("=====================")
        #=======================================================================
        
        joint_num = int(gt_coord.shape[1]/3)
        hm_width = heatmap_out.shape[-1]
        hm_height = heatmap_out.shape[-2]
        hm_depth = heatmap_out.shape[-3] // FreiHandConfig.num_joints
        coord_out = softmax_integral_tensor(heatmap_out, joint_num, hm_width, hm_height, hm_depth)
        
        num_unsupervised_samples = coord_out[~labelled].shape[0]
        num_supervised_samples = coord_out[labelled].shape[0]
        coord_out_teacher = torch.squeeze(coord_out_teacher, dim=0)
        
        loss_unsupervised = 0
        loss_supervised = 0
        loss_unsupervised_tmp = 0
        loss_supervised_tmp = 0   
        student_mpjpe =  1
        teacher_mpjpe = 1
        #=======================================================================
        gt_norm = gt_coord.clone()
        gt_norm = gt_norm.reshape((gt_norm.shape[0], FreiHandConfig.num_joints, 3))
        gt_norm = augment.prepare_panet_input(gt_norm, tprime, trans, bbox, K, R, scale)
        #  
        # coord_out_tmp = coord_out.clone()
        # coord_out_tmp = coord_out_tmp.reshape((coord_out_tmp.shape[0], FreiHandConfig.num_joints, 3))
        # coord_out_tmp = augment.prepare_panet_input(coord_out_tmp, tprime, trans, bbox, K, R, scale)        
        # 
        # coord_out_teacher_tmp = coord_out_teacher.clone()
        # coord_out_teacher_tmp = coord_out_teacher_tmp.reshape((coord_out_teacher_tmp.shape[0], FreiHandConfig.num_joints, 3))
        # coord_out_teacher_tmp = augment.prepare_panet_input(coord_out_teacher_tmp, tprime, trans, bbox, K, R, scale)         
        #=======================================================================
         
        #=======================================================================
        # print("gt_norm")
        # print(gt_norm.shape)
        # print("coord_out_tmp")
        # print(coord_out_tmp.shape)
        # print("coord_out_teacher")
        # print(coord_out_teacher_tmp.shape)
        #=======================================================================
         
        with torch.no_grad():
            student_mpjpe = computeMPJPE(coord_out, gt_coord)
            teacher_mpjpe = computeMPJPE(coord_out_teacher, gt_coord)
        print(num_unsupervised_samples, num_supervised_samples)
        if(num_unsupervised_samples > 0):
            input_to_panet = coord_out[~labelled].reshape((coord_out[~labelled].shape[0], FreiHandConfig.num_joints, 3))
            # to do: remove rotation
            
            input_to_panet = augment.prepare_panet_input(input_to_panet, tprime[~labelled], trans[~labelled], bbox[~labelled], 
                                                         K[~labelled], R[~labelled], scale[~labelled])

            panet_output, pts_recon_canonical, camera_matrix, code = nrsfm_tester.forward(input_to_panet)
            #===================================================================
            # print(input_to_panet)
            # print(panet_output)
            #print("=======================================================")
            #===================================================================
            panet_output = panet_output.reshape(panet_output.shape[0], FreiHandConfig.num_joints * 3)
            input_to_panet = input_to_panet.reshape(input_to_panet.shape[0], FreiHandConfig.num_joints * 3)
            #input_to_panet = input_to_panet.reshape(input_to_panet.shape[0], FreiHandConfig.num_joints * 3)
#            _assert_grad(panet_output)
            #coord_out_reshaped =  coord_out.reshape(coord_out.shape[0], FreiHandConfig.num_joints,  3)
            #coord_out_reshaped = coord_out_reshaped - coord_out_reshaped.mean(1, keepdims=True)
            #coord_out_reshaped = coord_out_reshaped.reshape(coord_out_reshaped.shape[0], FreiHandConfig.num_joints * 3)

            #===================================================================
            # gt_norm = gt_coord.clone()
            # gt_norm = gt_norm.reshape((gt_norm.shape[0], FreiHandConfig.num_joints, 3))
            # gt_norm = augment.prepare_panet_input(gt_norm, tprime, trans, bbox, K, R, scale)
            # 
            # panet_output, pts_recon_canonical, camera_matrix, code = nrsfm_tester.forward(gt_norm)
            # print(gt_norm)
            # print(panet_output)
            # 
            # sys.exit()          
            #===================================================================
            #===================================================================
            # print("gt_coord")
            # print(gt_norm)
            # print('input_to_panet')
            # print(input_to_panet)
            # # print(input_to_panet.shape)
            # print("panet_output")
            # print(panet_output)
            # # print(coord_out_reshaped.shape)
            # print("computeMPJPE gt-input")
            # print(computeMPJPE(gt_norm, input_to_panet))
            # print("computeMPJPE gt-output")
            # print(computeMPJPE(gt_norm, panet_output))
            # print("computeMPJPE input-output")
            # print(computeMPJPE(input_to_panet, panet_output))
            # print(labelled)
            # sys.exit()            
            #===================================================================
            Lteacher = (torch.abs(coord_out[~labelled] - coord_out_teacher[~labelled])) * gt_vis[~labelled]
            LPanet = (cfg._lambda * torch.abs(input_to_panet - panet_output)) * gt_vis[~labelled]
            #loss_unsupervised = (torch.abs(coord_out[~labelled] - coord_out_teacher[~labelled]) + 
            #                     cfg._lambda * torch.abs(coord_out_reshaped[~labelled] - panet_output)) * gt_vis[~labelled]
            #loss_unsupervised = LPanet + Lteacher
            if self.size_average:
                LPanet = LPanet.sum() / num_unsupervised_samples
                #LPanet = 0
                Lteacher = Lteacher.sum() / num_unsupervised_samples
                #print(LPanet)
                #print(Lteacher)
                loss_unsupervised = LPanet + Lteacher
            else:
                loss_unsupervised = LPanet + Lteacher
                loss_unsupervised =  loss_unsupervised.sum()                    
                       
        if (num_supervised_samples > 0):
            input_to_panet = gt_coord[labelled].reshape((gt_coord[labelled].shape[0], FreiHandConfig.num_joints, 3))
            input_to_panet = augment.prepare_panet_input(input_to_panet, tprime[labelled], trans[labelled], 
                                                         bbox[labelled], K[labelled], R[labelled], scale[labelled], p=False)
            #joint_cam_normalized = joint_cam_normalized[labelled]
            #joint_cam_normalized = joint_cam_normalized - joint_cam_normalized.mean(1, keepdims=True)          
            #assert torch.max(input_to_panet - joint_cam_normalized) < 10e-4     
            loss_supervised = torch.abs(coord_out[labelled] - gt_coord[labelled]) * gt_vis[labelled]
            if self.size_average:
                loss_supervised = loss_supervised.sum() / num_supervised_samples
            else:
                loss_supervised =  loss_supervised.sum()
        
        if loss_supervised !=0 and loss_unsupervised !=0:
            loss = loss_supervised + loss_unsupervised
            loss_supervised_tmp = loss_supervised.detach()
            loss_unsupervised_tmp = loss_unsupervised.detach()
        elif loss_supervised !=0:
            loss = loss_supervised
            loss_supervised_tmp = loss_supervised.detach()
        elif loss_unsupervised !=0:
            loss = loss_unsupervised
            loss_unsupervised_tmp = loss_unsupervised.detach()
        #print(loss)
        #print('=============================')
        #print(loss_supervised)
        #print(type(loss_unsupervised))

        #_assert_no_grad(student_mpjpe)
        #_assert_no_grad(teacher_mpjpe)

        _assert_no_grad(loss_supervised_tmp)
        _assert_no_grad(loss_unsupervised_tmp)
        _assert_grad(loss_supervised)
        _assert_grad(loss_unsupervised)
        _assert_grad(loss)
        return loss, student_mpjpe, teacher_mpjpe, loss_supervised_tmp, loss_unsupervised_tmp
        
        
        
class JointLocationLoss2(nn.Module):
    def __init__(self):
        super(JointLocationLoss2, self).__init__()
        self.size_average = True

    def forward(self, heatmap_out, gt_label, gt_vis, joint_cam, joint_cam_normalized, bbox, scale, R, trans, K, tprime):
        
        joint_num = int(gt_label.shape[1]/3)
        #print(heatmap_out.shape)
        hm_width = heatmap_out.shape[-1]
        hm_height = heatmap_out.shape[-2]
        hm_depth = heatmap_out.shape[-3] // FreiHandConfig.num_joints
        coord_out = softmax_integral_tensor(heatmap_out, joint_num, hm_width, hm_height, hm_depth)
        
        _assert_no_grad(gt_label)
        _assert_no_grad(gt_vis)
        
        #print(coord_out)
        #print(gt_coord)
        #loss = torch.abs(coord_out - gt_coord) * gt_vis
        coord_out = coord_out.detach().cpu().numpy()
        label = augment.test_get_joint_loc_res(coord_out)
        label_gt = augment.test_get_joint_loc_res(gt_label.detach().cpu().numpy())
        joint_cam = joint_cam.detach().cpu().numpy()
        joint_cam_normalized = joint_cam_normalized.detach().cpu().numpy()
        bbox = bbox.detach().cpu().numpy()
        scale = scale.detach().cpu().numpy()
        R = R.detach().cpu().numpy()
        trans = trans.detach().cpu().numpy()
        trans = np.linalg.inv(trans)
        K = K.detach().cpu().numpy()
        tprime = tprime.detach().cpu().numpy()
        pre_3d_kpt = []
        for n_sample in range(label.shape[0]):
            xyz_rot = np.matmul(R[n_sample], joint_cam[n_sample].T).T         
            tmp = augment.trans_coords_from_patch_to_org_3d(label[n_sample], bbox[n_sample, 0],
                                                            bbox[n_sample, 1], bbox[n_sample, 2],
                                                            bbox[n_sample, 3], cfg.patch_width, cfg.patch_height, 
                                                            scale[n_sample], trans[n_sample], tprime[n_sample])
            tmp2 = augment.trans_coords_from_patch_to_org_3d(label_gt[n_sample], bbox[n_sample, 0],
                                                            bbox[n_sample, 1], bbox[n_sample, 2],
                                                            bbox[n_sample, 3], cfg.patch_width, cfg.patch_height, 
                                                            scale[n_sample], trans[n_sample], tprime[n_sample])
            
            pre_3d = augment.pixel2cam(tmp, K[n_sample])
            label_3d_kpt = augment.pixel2cam(tmp2, K[n_sample])
            Rn = R[n_sample]
            label_3d_kpt = np.matmul(Rn.T, label_3d_kpt.T).T
            #label_3d_kpt = torch.matmul(R[n_sample], label_3d_kpt.transpose(1, 0)).transpose(1,0)    
            pre_3d = np.matmul(Rn.T, pre_3d.T).T
            #pre_3d = torch.matmul(R[n_sample], pre_3d.transpose(1, 0)).transpose(1,0)
            pre_3d_kpt.append(pre_3d)
            try:
                assert np.allclose(label_3d_kpt, joint_cam_normalized[n_sample], rtol=1e-6, atol=1e-6)
            except:
                print("(loss::JointLocationLoss2) Warning: label_3d_kpt and joint_cam_normalized are not equal.")
        pre_3d_kpt = np.array(pre_3d_kpt)
        loss = []
        for i in range(pre_3d_kpt.shape[0]):          
            diff = joint_cam_normalized[i] - pre_3d_kpt[i]
            #euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))
            euclidean_dist = np.sum(np.square(diff), axis=1)
            loss.append(euclidean_dist)
        loss = torch.from_numpy(np.array(loss)).cuda()
        loss.requires_grad = False
        
        self.size_average = False
        if self.size_average:
            return loss.sum() / len(coord_out)
        else:
            return loss.sum()
