import os, sys
import numpy as np
import cv2
import random
import time
import torch
import copy
import sys

""" Declare global variables """
patch_width = 224
patch_height = 224

input_shape = (patch_width, patch_height)
pad_factor = 1.75
#pad_factor = 10


def find_bb(uv, aspect_ratio=1.0):
    """ Find the bounding box surrounding the object """
    u, d, l, r = calc_kpt_bound(uv)

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
        
    w *= pad_factor
    h *= pad_factor
    bbox = [center_x, center_y, w, h]
    return bbox

def calc_kpt_bound(kpts):
    MAX_COORD = 10000
    x, y = kpts[:, 0], kpts[:, 1]

    u, l = MAX_COORD, MAX_COORD
    d, r = -1, -1

    for idx, vis in enumerate(y):
        if vis == 0:    # skip invisible joint
            continue

        u = min(u, y[idx])
        d = max(d, y[idx])
        l = min(l, x[idx])
        r = max(r, x[idx])

    return u, d, l, r

def projectPoints(xyz, R, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    # xyz_rot = np.matmul(R, xyz.T).T
    # uv = np.matmul(K, xyz_rot.T).T
    # return uv[:, :2] / uv[:, -1:], xyz_rot[:, -1]*1000, xyz_rot
    uv = np.matmul(K, xyz.T).T
    return uv[:,:2]/uv[:,-1:], xyz[:,-1]*1000

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

    """ augment size with scale """
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

def generate_patch_image(joint_cam, scale, R, K, aspect_ratio, inv, num_joints, root_idx): 
    uv, z = projectPoints(joint_cam, R, K)
    
    joint_img = np.zeros((21, 3))
    joint_img[:,0] = uv[:,0]
    joint_img[:,1] = uv[:,1]
    
    # Root centered
    z_mean = np.mean(z)
    joint_img[:,2] = np.squeeze(z - z[root_idx])
        
    bbox = find_bb(uv)    
    bb_c_x = float(bbox[0])
    bb_c_y = float(bbox[1])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])
    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, input_shape[1], input_shape[0], scale, inv=inv)
    return trans, joint_img, bbox

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]


def generate_joint_location_label(patch_width, patch_height, joints):
    joints[:, 0] = joints[:, 0] / patch_width - 0.5
    joints[:, 1] = joints[:, 1] / patch_height - 0.5
    joints[:, 2] = joints[:, 2] / patch_width

    return joints
    
def start(joint_cam, scale, R, K, aspect_ratio, inv, num_joints, root_idx):
    trans, joint_img, bbox = generate_patch_image(joint_cam, scale, R, K,
                                                               aspect_ratio, inv, num_joints, root_idx)
    zoom_factor = max(bbox[3], bbox[2])

    for n_jt in range(len(joint_img)):
        joint_img[n_jt, 0:2] = trans_point2d(joint_img[n_jt, 0:2], trans)        
        joint_img[n_jt, 2] = joint_img[n_jt, 2] / (zoom_factor * scale) * patch_width


    labels = generate_joint_location_label(patch_width, patch_height, joint_img)
    return labels
