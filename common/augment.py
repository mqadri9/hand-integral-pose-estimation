import os
import numpy as np
import cv2
import random
import time
import torch
import copy
from random import uniform 
from config import cfg
from FreiHand_config import FreiHandConfig
from nets.loss import softmax_integral_tensor
import matplotlib.pyplot as plt

plt.switch_backend('agg')

def compute_similarity_transform(X, Y, compute_optimal_scale=False):
    """
    A port of MATLAB's `procrustes` function to Numpy.
    Adapted from http://stackoverflow.com/a/18927641/1884420

    Args
        X: array NxM of targets, with N number of points and M point dimensionality
        Y: array NxM of inputs
        compute_optimal_scale: whether we compute optimal scale or force it to be 1

    Returns:
        d: squared error after transformation
        Z: transformed Y
        T: computed rotation
        b: scaling
        c: translation
    """
    muX = X.mean(0)
    muY = Y.mean(0)

    X0 = X - muX
    Y0 = Y - muY

    ssX = (X0 ** 2.).sum()
    ssY = (Y0 ** 2.).sum()

    # centred Frobenius norm
    normX = np.sqrt(ssX)
    normY = np.sqrt(ssY)

    # scale to equal (unit) norm
    X0 = X0 / normX
    Y0 = Y0 / normY

    # optimum rotation matrix of Y
    A = np.dot(X0.T, Y0)
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    V = Vt.T
    T = np.dot(V, U.T)

    # Make sure we have a rotation
    detT = np.linalg.det(T)
    V[:, -1] *= np.sign(detT)
    s[-1] *= np.sign(detT)
    T = np.dot(V, U.T)

    traceTA = s.sum()

    if compute_optimal_scale:  # Compute optimum scaling of Y.
        b = traceTA * normX / normY
        d = 1 - traceTA ** 2
        Z = normX * traceTA * np.dot(Y0, T) + muX
    else:  # If no scaling allowed
        b = 1
        d = 1 + ssY / ssX - 2 * traceTA * normY / normX
        Z = normY * np.dot(Y0, T) + muX

    c = muX - b * np.dot(muY, T)

    return d, Z, T, b, c


def projectPoints(xyz, R, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    #uv = np.matmul(K, xyz.T).T
    xyz_rot = np.matmul(R, xyz.T).T
    uv = np.matmul(K, xyz_rot.T).T
    return uv[:, :2] / uv[:, -1:], xyz_rot[:, -1]*1000, xyz_rot

def pixel2cam(pixel_coord, K):

    uv = np.ones(pixel_coord.shape)
    uv[:, 0] = pixel_coord[:, 0]
    uv[:, 1] = pixel_coord[:, 1]
    xyz = np.matmul(np.linalg.inv(K), uv.T).T
    #print("xyz")
    #print(xyz)
    pixel_coord[..., 2] = pixel_coord[..., 2]/1000
    #print("pixel z")
    #print(pixel_coord[..., 2])
    xyz *= np.expand_dims(pixel_coord[..., 2], axis=1)
    #x = (pixel_coord[..., 0] - c[0]) / f[0] * pixel_coord[..., 2]
    #y = (pixel_coord[..., 1] - c[1]) / f[1] * pixel_coord[..., 2]
    #z = pixel_coord[..., 2]
    
    return xyz

def generate_joint_location_label(patch_width, patch_height, joints, joints_vis):
    #print("=============Inside generate_joint_location_label===========")
    #print("JOINTS")
    #print(joints)
    joints[:, 0] = joints[:, 0] / patch_width - 0.5
    joints[:, 1] = joints[:, 1] / patch_height - 0.5
    joints[:, 2] = joints[:, 2] / patch_width
    
    joints = joints.reshape((-1))
    joints_vis = joints_vis.reshape((-1))
    return joints, joints_vis  

def get_joint_location_result(patch_width, patch_height, preds):
    # TODO: This cause imbalanced GPU usage, implement cpu version
    hm_width = preds.shape[-1]
    hm_height = preds.shape[-2]
    #if config.output_3d:
    hm_depth = preds.shape[-3] // FreiHandConfig.num_joints
    num_joints = preds.shape[1] // hm_depth
    #else:
    #hm_depth = 1
    #num_joints = preds.shape[1]

    pred_jts = softmax_integral_tensor(preds, num_joints, hm_width, hm_height, hm_depth)
    coords = pred_jts.detach().cpu().numpy()
    coords = coords.astype(float)
    coords = coords.reshape((coords.shape[0], int(coords.shape[1] / 3), 3))
    # project to original image size
    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * patch_width
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * patch_height
    coords[:, :, 2] = coords[:, :, 2] * patch_width
    scores = np.ones((coords.shape[0], coords.shape[1], 1), dtype=float)

    # add score to last dimension
    coords = np.concatenate((coords, scores), axis=2)

    return coords

def trans_coords_from_patch_to_org(coords_in_patch, c_x, c_y, bb_width, bb_height, patch_width, patch_height, trans):
    coords_in_org = coords_in_patch.copy()
    for p in range(coords_in_patch.shape[0]):
        a = trans_point2d(coords_in_patch[p, 0:2], trans)
        coords_in_org[p, 0:2] = trans_point2d(coords_in_patch[p, 0:2], trans)
    return coords_in_org


def trans_coords_from_patch_to_org_3d(coords_in_patch, c_x, c_y, bb_width, bb_height, patch_width, patch_height, trans):
    res_img = trans_coords_from_patch_to_org(coords_in_patch, c_x, c_y, bb_width, bb_height, patch_width, patch_height, trans)
    res_img[:, 2] = coords_in_patch[:, 2] / cfg.patch_width * bb_width
    return res_img

# helper functions
def transform_joint_to_other_db(src_joint, src_name, dst_name):
    return src_joint
    #src_joint_num = len(src_name)
    #dst_joint_num = len(dst_name)

    #new_joint = np.zeros(((dst_joint_num,) + src_joint.shape[1:]))

    #for src_idx in range(len(src_name)):
    #    name = src_name[src_idx]
    #    if name in dst_name:
    #        dst_idx = dst_name.index(name)
    #        new_joint[dst_idx] = src_joint[src_idx]

    #return new_joint

def get_aug_config():
    
    scale_factor = 0.25
    rot_factor = 30
    color_factor = 0.2
    
    #scale = np.clip(np.random.randn(), -1.0, 1.0) * scale_factor + 1.0
    #rot = np.clip(np.random.randn(), -2.0,
    #              2.0) * rot_factor if random.random() <= 0.6 else 0
    scale = 1
    rot = sample_rotation_matrix()
    
    c_up = 1.0 + color_factor
    c_low = 1.0 - color_factor
    color_scale = [random.uniform(c_low, c_up), random.uniform(c_low, c_up), random.uniform(c_low, c_up)]

    return scale, rot, color_scale

def sample_rotation_matrix():
    # Rotate with a probability of 40 percent
    # Right now the only rotation is around the z axis from -30 deg tp 30 deg
    
    if random.random() <= 0.6:
        return np.eye(3)    
    theta = uniform(-0.52, 0.52)
    if np.abs(theta) < 1e-4:
        return np.eye(3)
    s = np.zeros((2,1))
    r = np.random.randn(1,1)
    r = np.vstack((s, r))
    r = theta*(r/np.linalg.norm(r))
    #print(r.shape)
    R, _ = cv2.Rodrigues(r)
    return np.array(R)

def calc_kpt_bound(kpts, kpts_vis):
    MAX_COORD = 10000
    x = kpts[:, 0]
    y = kpts[:, 1]
    z = kpts_vis[:, 0]
    u = MAX_COORD
    d = -1
    l = MAX_COORD
    r = -1
    for idx, vis in enumerate(z):
        if vis == 0:  # skip invisible joint
            continue
        u = min(u, y[idx])
        d = max(d, y[idx])
        l = min(l, x[idx])
        r = max(r, x[idx])
    return u, d, l, r

def find_bb(uv, joint_vis, aspect_ratio=1.0):
    u, d, l, r = calc_kpt_bound(uv, joint_vis)

    center_x = (l + r) * 0.5
    center_y = (u + d) * 0.5
    assert center_x >= 1

    w = r - l
    h = d - u
    assert w > 0
    assert h > 0

    if w > aspect_ratio * h:
        h = w * 1.0 / aspect_ratio
    elif w < aspect_ratio * h:
        w = h * aspect_ratio
    
    w *= 1.75
    h *= 1.75
    
    bbox = [center_x, center_y, w, h]
    return bbox
  
def generate_patch_image(cvimg, joint_cam, scale, R, K, aspect_ratio=1.0, inv=False): 
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape
    
    uv_orig, z_orig, _ = projectPoints(joint_cam, np.eye(3), K)
    joint_img_orig = np.zeros((FreiHandConfig.num_joints, 3))
    joint_img_orig[:,0] = uv_orig[:,0]
    joint_img_orig[:,1] = uv_orig[:,1]
    # Root centered
    joint_img_orig[:,2] = np.squeeze(z_orig - z_orig[FreiHandConfig.root_idx])

    
    homo = K.dot(R).dot(np.linalg.inv(K))
    img2_w = cv2.warpPerspective(cvimg, homo, (cvimg.shape[1], cvimg.shape[0]))        

    joint_vis = np.ones(joint_cam.shape, dtype=np.float)
    #vis = np.ones(joint_cam.shape)
    #joint_vis = vis[:, 0] > 0
    #joint_vis = np.expand_dims(joint_vis, axis=1)
    uv, z, xyz_rot = projectPoints(joint_cam, R, K)
    
    joint_img = np.zeros((FreiHandConfig.num_joints, 3))
    joint_img[:,0] = uv[:,0]
    joint_img[:,1] = uv[:,1]
    # Root centered
    joint_img[:,2] = np.squeeze(z - z[FreiHandConfig.root_idx])
    
    bbox = find_bb(uv, joint_vis)
        
    bb_c_x = float(bbox[0])
    bb_c_y = float(bbox[1])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])
    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, cfg.input_shape[1], cfg.input_shape[0], scale, inv=inv)
    img_patch = cv2.warpPerspective(img2_w, trans, (int(cfg.input_shape[1]), int(cfg.input_shape[0])), flags=cv2.INTER_LINEAR)
    #print("img path before transformation")
    #nn = str(random.randint(0,1000))
    #print(nn)
    # Swap first and last columns # BGR to RGB
    img_patch = img_patch[:,:,::-1].copy()
    img_patch = img_patch.astype(np.float32)
    #print("img patch after transformation")
    #print(img_patch.shape)
    #cv2.imwrite('/home/mqadri/hand-integral-pose-estimation/tests/{}.jpg'.format(nn), cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR))
    return img_patch, trans, joint_img, joint_img_orig, joint_vis, xyz_rot, bbox

def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)

def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, inv=False):
    """
    Input is: 
    c_x: bb center x
    c_y: bb center y
    src_width: bb_width
    src_height: bb_height
    dst_width: cfg.input_shape[1]
    dst_height: cfg.input_shape[0]
    scale/rot
    inv: 
        True: find a transformation from destination to source
    """
    # augment size with scale
    src_w = src_width * scale
    src_h = src_height * scale
    
    src_l = np.array([c_x-src_w*0.5, c_y-src_h*0.5], dtype=np.float32)
    src_r = np.array([c_x-src_w*0.5, c_y+src_h*0.5], dtype=np.float32)
    src_t = np.array([c_x+src_w*0.5, c_y-src_h*0.5], dtype=np.float32)
    src_b = np.array([c_x+src_w*0.5, c_y+src_h*0.5], dtype=np.float32)

    
    dst_w = dst_width
    dst_h = dst_height
    dst_l = np.array([0, 0], dtype=np.float32)
    dst_r = np.array([0, dst_h], dtype=np.float32)
    dst_t = np.array([dst_w, 0], dtype=np.float32)
    dst_b = np.array([dst_w, dst_h], dtype=np.float32)

    src = np.zeros((4, 2), dtype=np.float32)
    src[0, :] = src_l
    src[1, :] = src_r
    src[2, :] = src_t
    src[3, :] = src_b

    dst = np.zeros((4, 2), dtype=np.float32)
    dst[0, :] = dst_l
    dst[1, :] = dst_r
    dst[2, :] = dst_t
    dst[3, :] = dst_b

    if inv:
        trans = cv2.getPerspectiveTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))

    return trans

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]
    