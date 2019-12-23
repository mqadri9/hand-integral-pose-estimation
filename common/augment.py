import os
import numpy as np
import cv2
import random
import time
import torch
import copy
from random import uniform 

from config import cfg


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
    # Right now the only rotation is around the z axis
    
    if random.random() <= 0.6:
        return np.eye(3)    
    theta = uniform(-2, 2)
    s = np.zeros((2,1))
    r = np.random.randn(1,1)
    r = np.vstack((s, r))
    r = theta*(r/np.linalg.norm(r))
    #print(r.shape)
    R, _ = cv2.Rodrigues(r)
    return np.array(R)
    #R = [[1, 0, 0],
    #     [0, np.cos(theta), -np.sin(theta)],
    #     [0, np.sin(theta), np.cos(theta)]
    #     ]
    #R1 = np.array([[np.cos(theta), 0, np.sin(theta)],
    #     [0, 1, 0],
    #     [-np.sin(theta), 0, np.cos(theta)]
    #     ])
    
    #R2 = np.array([[np.cos(theta), -np.sin(theta), 0],
    #     [np.sin(theta), np.cos(theta), 0],
    #     [0, 0, 1]])
    
    #R = R1.dot(R2)
    #R, _ = cv2.Rodrigues(r)
    #return np.array(R)

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
    
def generate_patch_image2(cvimg, joint_cam, scale, R, K, aspect_ratio=1.0): 
    #print("======== INSIDE generate_patch_image=================")
    #print(bbox)
    #print(scale)
    #print(rot)
    #===========================================================================
    # print("R")
    # print(R)
    # print("scale")
    # print(scale)
    # print("K")
    # print(K)
    # print("joint_cam")
    # print(joint_cam)
    #===========================================================================
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape

    homo = K.dot(R).dot(np.linalg.inv(K))
    img2_w = cv2.warpPerspective(cvimg, homo, (cvimg.shape[1], cvimg.shape[0]))        
    
    xyz = np.array(joint_cam)
    vis = np.ones(xyz.shape)
    joint_vis = vis[:, 0] > 0
    joint_vis = np.expand_dims(joint_vis, axis=1)
    K = np.array(K)
    #uv = np.matmul(K, xyz.T).T
    xyz_rot = np.matmul(R, xyz.T).T
    uv = np.matmul(K, xyz_rot.T).T   
    uv = uv[:, :2] / uv[:, -1:]
    
    # convert to mm
    z = xyz[:, -1]*1000
    joint_img = np.zeros((21, 3))
    joint_img[:,0] = uv[:,0]
    joint_img[:,1] = uv[:,1]
    # Root centered
    joint_img[:,2] = np.squeeze(z - z[9])
            
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
        
    bb_c_x = float(bbox[0])
    bb_c_y = float(bbox[1])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])
    rot = 0
    trans = gen_trans_from_patch_cv2(bb_c_x, bb_c_y, bb_width, bb_height, cfg.input_shape[1], cfg.input_shape[0], scale, rot, inv=False)
    img_patch = cv2.warpPerspective(img2_w, trans, (int(cfg.input_shape[1]), int(cfg.input_shape[0])), flags=cv2.INTER_LINEAR)
    #print("img path before transformation")
    print(img_patch.shape)
    #nn = str(random.randint(0,1000))
    #print(nn)
    # Swap first and last columns # BGR to RGB
    img_patch = img_patch[:,:,::-1].copy()
    img_patch = img_patch.astype(np.float32)
    #print("img patch after transformation")
    #print(img_patch.shape)
    #cv2.imwrite('/home/mqadri/hand-integral-pose-estimation/tests/{}.jpg'.format(nn), cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR))
    return img_patch, trans, joint_img

def generate_patch_image(cvimg, bbox, scale, rot):
    #print("======== INSIDE generate_patch_image=================")
    #print(bbox)
    #print(scale)
    #print(rot)
    img = cvimg.copy()
    #print(img.shape)
    #nn = str(random.randint(1001,2000))
    #cv2.imwrite('/home/mqadri/hand-integral-pose-estimation/tests/{}.jpg'.format(nn), cv2.cvtColor(cvimg, cv2.COLOR_RGB2BGR))
    img_height, img_width, img_channels = img.shape

    bb_c_x = float(bbox[0])
    bb_c_y = float(bbox[1])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])
    
    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, cfg.input_shape[1], cfg.input_shape[0], scale, rot, inv=False)
    img_patch = cv2.warpAffine(img, trans, (int(cfg.input_shape[1]), int(cfg.input_shape[0])), flags=cv2.INTER_LINEAR)
    #print("img path before transformation")
    #print(img_patch.shape)
    #nn = str(random.randint(0,1000))
    #print(nn)
    # Swap first and last columns # BGR to RGB
    img_patch = img_patch[:,:,::-1].copy()
    img_patch = img_patch.astype(np.float32)
    #print("img patch after transformation")
    #print(img_patch.shape)
    #cv2.imwrite('/home/mqadri/hand-integral-pose-estimation/tests/{}.jpg'.format(nn), cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR))
    return img_patch, trans



def rotate_2d(pt_2d, rot_rad):
    x = pt_2d[0]
    y = pt_2d[1]
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)
    xx = x * cs - y * sn
    yy = x * sn + y * cs
    return np.array([xx, yy], dtype=np.float32)


def gen_trans_from_patch_cv(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
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
    src_center = np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    rot_rad = np.pi * rot / 180
    src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)

    dst_w = dst_width
    dst_h = dst_height
    dst_center = np.array([dst_w * 0.5, dst_h * 0.5], dtype=np.float32)
    dst_downdir = np.array([0, dst_h * 0.5], dtype=np.float32)
    dst_rightdir = np.array([dst_w * 0.5, 0], dtype=np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = src_center
    src[1, :] = src_center + src_downdir
    src[2, :] = src_center + src_rightdir

    dst = np.zeros((3, 2), dtype=np.float32)
    dst[0, :] = dst_center
    dst[1, :] = dst_center + dst_downdir
    dst[2, :] = dst_center + dst_rightdir

    if inv:
        trans = cv2.getPerspectiveTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getPerspectiveTransform(np.float32(src), np.float32(dst))

    return trans


def gen_trans_from_patch_cv2(c_x, c_y, src_width, src_height, dst_width, dst_height, scale, rot, inv=False):
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
    #src_center = np.array([c_x, c_y], dtype=np.float32)
    # augment rotation
    #rot_rad = np.pi * rot / 180
    #src_downdir = rotate_2d(np.array([0, src_h * 0.5], dtype=np.float32), rot_rad)
    #src_rightdir = rotate_2d(np.array([src_w * 0.5, 0], dtype=np.float32), rot_rad)
    
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
    