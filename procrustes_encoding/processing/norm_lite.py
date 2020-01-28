import numpy as np

""" Declare global variables """
patch_width = 224
patch_height = 224

input_shape = (patch_width, patch_height)
pad_factor = 1.75
#pad_factor = 10


def find_bb(uv, joint_vis, aspect_ratio):
    """ Find the bounding box surrounding the object """
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

    w *= pad_factor
    h *= pad_factor
    bbox = [center_x, center_y, w, h]
    return bbox

def calc_kpt_bound(kpts, kpts_vis):
    MAX_COORD = 10000
    x, y = kpts[:, 0], kpts[:, 1]
    z = kpts_vis[:, 0]

    u, l = MAX_COORD, MAX_COORD
    d, r = -1, -1

    for idx, vis in enumerate(z):
        if vis == 0:    # skip invisible joint
            continue

        u = min(u, y[idx])
        d = max(d, y[idx])
        l = min(l, x[idx])
        r = max(r, x[idx])

    return u, d, l, r

def projectPoints(xyz, K):
    """ Project 3D coordinates into image space. """
    xyz = np.array(xyz)
    K = np.array(K)
    uv = np.matmul(K, xyz.T).T
    return uv[:,:2]/uv[:,-1:], xyz[:,-1]*1000

def generate_joint_cam_normalized(joint_cam, K, aspect_ratio, num_joints, scaling_constant):
    uv, z = projectPoints(joint_cam, K)
    joint_vis = np.ones(joint_cam.shape, dtype=np.float)
    bbox = find_bb(uv, joint_vis, aspect_ratio)

    L = max(bbox[2], bbox[3])

    tprime = scaling_constant * K[1, 1] / L
    if L == bbox[2]:
        # multiply by 100 to increase the value ranges of joint_cam_normalized
        # so basically scale the hand to be a constant length of 100 instead of 1
        tprime = scaling_constant * K[0, 0] / L
    else:
        tprime = scaling_constant * K[1, 1] / L

    joint_cam_normalized = joint_cam * tprime/z[9]
    joint_cam_normalized[:,2] = np.squeeze(joint_cam_normalized[:,2] - tprime)

    return joint_cam_normalized
