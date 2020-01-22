import os, sys
import numpy as np
import cv2
import random
import time
import torch
import copy


patch_width = 224
patch_height = 224

input_shape = (224, 224) 
pad_factor = 1.75
use_hand_detector = False
online_hand_detection = False
scaling_constant = 100

def scale_bb(bbox, aspect_ratio=1.0):
    center_x = bbox[0]
    center_y = bbox[1]
    bb_width = bbox[2]
    bb_height = bbox[3]
    if bb_width > aspect_ratio * bb_height:
        bb_height = bb_width * 1.0 / aspect_ratio
    elif bb_width < aspect_ratio * bb_height:
        bb_width = bb_height * aspect_ratio
        
    bb_width *= pad_factor
    bb_height *= pad_factor
    return [center_x, center_y, bb_width, bb_height]    

def find_bb(uv, joint_vis, aspect_ratio=1.0):
    u, d, l, r = calc_kpt_bound(uv, joint_vis)

    center_x = (l + r) * 0.5
    center_y = (u + d) * 0.5
    assert center_x >= 1

    w = r - l
    h = d - u
    assert w > 0
    assert h > 0
    #w = np.clip(w, 0, cfg.patch_width)
    #h = np.clip(h, 0, cfg.patch_height)
    bbox = [center_x, center_y, w, h]
    bbox = scale_bb(bbox, aspect_ratio=aspect_ratio)
    return bbox

def projectPoints(xyz, R, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    xyz_rot = np.matmul(R, xyz.T).T
    uv = np.matmul(K, xyz_rot.T).T
    return uv[:, :2] / uv[:, -1:], xyz_rot[:, -1]*1000, xyz_rot

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
    #src_w = np.clip(src_width * scale, 0, cfg.patch_width)
    #src_h = np.clip(src_height * scale, 0, cfg.patch_height)
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

def generate_patch_image(cvimg, joint_cam, scale, R, K, aspect_ratio=1.0, inv=False, hand_detector=None, img_path=None, return_bbox=True, faster_rcnn_bbox=None):
    img = cvimg.copy()
    img_height, img_width, img_channels = img.shape
    
    uv_orig, z_orig, _ = projectPoints(joint_cam, np.eye(3), K)
    joint_img_orig = np.zeros((FreiHandConfig.num_joints, 3))
    joint_img_orig[:,0] = uv_orig[:,0]
    joint_img_orig[:,1] = uv_orig[:,1]
    # Root centered
    joint_img_orig[:,2] = np.squeeze(z_orig - z_orig[FreiHandConfig.root_idx])
    
    homo = K.dot(R).dot(np.linalg.inv(K))
    #dst_w, dst_h = find_perspective_bounds(homo, cvimg)
    img2_w = cv2.warpPerspective(cvimg, homo, (cvimg.shape[1], cvimg.shape[0]))
    #img2_w = cv2.warpPerspective(cvimg, homo, (int(dst_h), int(dst_w)))

    joint_vis = np.ones(joint_cam.shape, dtype=np.float)
    #vis = np.ones(joint_cam.shape)
    #joint_vis = vis[:, 0] > 0
    #joint_vis = np.expand_dims(joint_vis, axis=1)
    uv, z, xyz_rot = projectPoints(joint_cam, R, K)

    if use_hand_detector and return_bbox:
        if online_hand_detection:
            bbox = find_bb_hand_detector(img_path, hand_detector, aspect_ratio=1.0)
        else:
            bbox = faster_rcnn_bbox
    else:
        bbox = find_bb(uv, joint_vis)

    # Draw a bounding box around the projected points 
    # bbox is the scaled bounding box + padding has been applied it
    # Find the maximum height and width 
    L = max(bbox[2], bbox[3])
    # L would is the our projected high
    if L == bbox[2]:
        # multiply by a 100 to increase the value ranges of joint_cam_normalized at line 375
        # so basically scale the hand to be a constant length of 100 instead of 1
        tprime = scaling_constant * K[0, 0] / L
    else: 
        tprime = scaling_constant * K[1, 1] / L
    
    joint_cam_normalized = joint_cam * tprime/z[9]
    # joint_cam_normalized is the  new 3D groundthruth
    
    uv_scaled, z_scaled, xyz_rot_scaled = projectPoints(joint_cam_normalized, R, K)
    joint_img = np.zeros((FreiHandConfig.num_joints, 3))
    joint_img[:,0] = uv_scaled[:,0]
    joint_img[:,1] = uv_scaled[:,1]
    # Root centered
    #joint_img[:,2] = np.squeeze(z_scaled - z_scaled[FreiHandConfig.root_idx])
    joint_img[:,2] = np.squeeze(z_scaled - tprime)
    
    bb_c_x = float(bbox[0])
    bb_c_y = float(bbox[1])
    bb_width = float(bbox[2])
    bb_height = float(bbox[3])
    trans = gen_trans_from_patch_cv(bb_c_x, bb_c_y, bb_width, bb_height, input_shape[1], input_shape[0], scale, inv=inv)
    img_patch = cv2.warpPerspective(img2_w, trans, (int(input_shape[1]), int(input_shape[0])), flags=cv2.INTER_LINEAR)
    #print("img path before transformation")
    #nn = str(random.randint(0,1000))
    #print(nn)
    # Swap first and last columns # BGR to RGB
    img_patch = img_patch[:,:,::-1].copy()
    img_patch = img_patch.astype(np.float32)
    #print("img patch after transformation")
    #print(img_patch.shape)
    #cv2.imwrite('/home/mqadri/hand-integral-pose-estimation/tests/{}.jpg'.format(nn), cv2.cvtColor(img_patch, cv2.COLOR_RGB2BGR))
    return img_patch, trans, joint_img, joint_img_orig, joint_cam_normalized, joint_vis, xyz_rot, bbox, tprime 

def trans_point2d(pt_2d, trans):
    src_pt = np.array([pt_2d[0], pt_2d[1], 1.]).T
    dst_pt = np.dot(trans, src_pt)
    return dst_pt[0:2]


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
    
def start(cvimg, joint_cam, scale, R, K, aspect_ratio=1.0, inv=False):
    img_patch, trans, joint_img, joint_img_orig, joint_cam_normalized, joint_vis, xyz_rot, bbox, tprime = augment.generate_patch_image(cvimg, joint_cam, scale, R, K, inv=False)
    for n_jt in range(len(joint_img)):
        joint_img[n_jt, 0:2] = trans_point2d(joint_img[n_jt, 0:2], trans)    
    
    # I believe what PANet need is joint_cam_normalized which is just the scaled version of the 3D hand.
    # If that is the case you can just return: joint_cam_normalized instead of "label"
    # What label is is:
    # the first 2 columns are the projections u and v divided 
    # by patch width and height and the third column is z-z_root of the joint_cam_normalized by patch width
    
    # joint_cam_normalized[:,2] = np.squeeze(joint_cam_normalized[:,2] - tprime)
    
    label, label_weight = generate_joint_location_label(patch_width, patch_height, joint_img, joint_vis)
    return label



if __name__ == "__main__":
    # cvimg = cv2.imread(<PUT IMAGE PATH HERE>, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    # joint_cam : original 3D keypoints
    # R: Rotation augmentation if none: np.eye(3)
    # Intrinsics K
    label = start(cvimg, joint_cam, 1.0, R, K, aspect_ratio=1.0, inv=False)



    
