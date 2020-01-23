from __future__ import print_function, unicode_literals
import os
import argparse
from config import cfg
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
from FreiHand import FreiHand
from scipy.linalg import orthogonal_procrustes
import numpy as np

######## Code included in the FreiHand GIT Repo ######
class EvalUtil:
    """ Util class for evaluation networks.
    """
    def __init__(self, num_kp=21):
        # init empty data storage
        self.data = list()
        self.num_kp = num_kp
        for _ in range(num_kp):
            self.data.append(list())

    def feed(self, keypoint_gt, keypoint_vis, keypoint_pred, skip_check=False):
        """ Used to feed data to the class. Stores the euclidean distance between gt and pred, when it is visible. """
        if not skip_check:
            keypoint_gt = np.squeeze(keypoint_gt)
            keypoint_pred = np.squeeze(keypoint_pred)
            keypoint_vis = np.squeeze(keypoint_vis).astype('bool')

            assert len(keypoint_gt.shape) == 2
            assert len(keypoint_pred.shape) == 2
            assert len(keypoint_vis.shape) == 1

        # calc euclidean distance
        #print(keypoint_gt.shape)
        diff = keypoint_gt - keypoint_pred
        euclidean_dist = np.sqrt(np.sum(np.square(diff), axis=1))

        num_kp = keypoint_gt.shape[0]
        for i in range(num_kp):
            if keypoint_vis[i]:
                self.data[i].append(euclidean_dist[i])

    def _get_pck(self, kp_id, threshold):
        """ Returns pck for one keypoint for the given threshold. """
        if len(self.data[kp_id]) == 0:
            return None

        data = np.array(self.data[kp_id])
        pck = np.mean((data <= threshold).astype('float'))
        return pck

    def _get_epe(self, kp_id):
        """ Returns end point error for one keypoint. """
        if len(self.data[kp_id]) == 0:
            return None, None

        data = np.array(self.data[kp_id])
        epe_mean = np.mean(data)
        epe_median = np.median(data)
        return epe_mean, epe_median

    def get_measures(self, val_min, val_max, steps):
        """ Outputs the average mean and median error as well as the pck score. """
        thresholds = np.linspace(val_min, val_max, steps)
        thresholds = np.array(thresholds)
        norm_factor = np.trapz(np.ones_like(thresholds), thresholds)

        # init mean measures
        epe_mean_all = list()
        epe_median_all = list()
        auc_all = list()
        pck_curve_all = list()

        # Create one plot for each part
        for part_id in range(self.num_kp):
            # mean/median error
            mean, median = self._get_epe(part_id)

            if mean is None:
                # there was no valid measurement for this keypoint
                continue

            epe_mean_all.append(mean)
            epe_median_all.append(median)

            # pck/auc
            pck_curve = list()
            for t in thresholds:
                pck = self._get_pck(part_id, t)
                pck_curve.append(pck)

            pck_curve = np.array(pck_curve)
            pck_curve_all.append(pck_curve)
            auc = np.trapz(pck_curve, thresholds)
            auc /= norm_factor
            auc_all.append(auc)

        epe_mean_all = np.mean(np.array(epe_mean_all))
        epe_median_all = np.mean(np.array(epe_median_all))
        auc_all = np.mean(np.array(auc_all))
        pck_curve_all = np.mean(np.array(pck_curve_all), 0)  # mean only over keypoints

        return epe_mean_all, epe_median_all, auc_all, pck_curve_all, thresholds

def load_data(f, gt_path, pred_file, set_name):
    versions = ['gs', 'hom', 'sample', 'auto']
    tmp = f.json_load(os.path.join(gt_path, '%s_xyz.json' % set_name))
    
    sp = "testing"
    if sp == "training":
        start = 0
        end = f.db_size('training')
        d_s = "training"
        
    elif sp == "testing":
        start = f.db_size('training') + 1
        end = start + f.db_size('testing') -1
        d_s = "training"
            
    data_dir = os.path.join(cfg.data_dir, "FreiHand")
    db_data_anno = f.load_db_annotation(data_dir, 'training')
    xyz_list = []
    for version in versions:
       #print("=============version==========")
       #print(version)
       for idx in range(start, end):
           #if idx%1000 == 0:
           #    print(idx)
           # annotation for this frame
           _, _, xyz, _ = db_data_anno[idx]
           xyz_list.append(xyz)
    xyz_list = np.load("ground_truth_test.npy")
    print("Ground truth list length: {}".format(len(xyz_list)))
    print("Loading predictions")
    predictions = np.load(pred_file)
    print("Prediction list length: {}".format(len(predictions)))
    return predictions, xyz_list

def align_w_scale(mtx1, mtx2, return_trafo=False):
    """ Align the predicted entity in some optimality sense with the ground truth. """
    # center
    t1 = mtx1.mean(0)
    t2 = mtx2.mean(0)
    mtx1_t = mtx1 - t1
    mtx2_t = mtx2 - t2

    # scale
    s1 = np.linalg.norm(mtx1_t) + 1e-8
    mtx1_t /= s1
    s2 = np.linalg.norm(mtx2_t) + 1e-8
    mtx2_t /= s2

    # orth alignment
    R, s = orthogonal_procrustes(mtx1_t, mtx2_t)

    # apply trafos to the second matrix
    mtx2_t = np.dot(mtx2_t, R.T) * s
    mtx2_t = mtx2_t * s1 + t1
    if return_trafo:
        return R, s, s1, t1 - t2
    else:
        return mtx2_t

def main(gt_path, pred_path, output_dir, pred_file_name=None, set_name=None):
    if pred_file_name is None:
        pred_file_name = 'pred.json'
    if set_name is None:
        set_name = 'evaluation'
        
    f = FreiHand(data_split = set_name)
    pred_file = os.path.join(pred_path, pred_file_name)
    pred, xyz_list = load_data(f, gt_path, pred_file, set_name)    
    
    assert len(pred) == len(xyz_list), 'Expected format mismatch.'
    eval_xyz, eval_xyz_aligned = EvalUtil(), EvalUtil()

    rng = tqdm(range(len(xyz_list)))
    for idx in rng:
        xyz = np.array(xyz_list[idx])
        xyz_pred = np.array(pred[idx])
        # Not aligned errors
        eval_xyz.feed(
            xyz,
            np.ones_like(xyz[:, 0]),
            xyz_pred
        )
    
        xyz_pred_aligned = align_w_scale(xyz, xyz_pred)
        # Aligned errors
        eval_xyz_aligned.feed(
            xyz,
            np.ones_like(xyz[:, 0]),
            xyz_pred_aligned
        )        

    # Calculate results
    xyz_mean3d, _, xyz_auc3d, pck_xyz, thresh_xyz = eval_xyz.get_measures(0.0, 0.05, 100)
    print('Evaluation 3D KP results:')
    print('auc=%.10f, mean_kp3d_avg=%.10f cm' % (xyz_auc3d, xyz_mean3d * 100.0))

    xyz_al_mean3d, _, xyz_al_auc3d, pck_xyz_al, thresh_xyz_al = eval_xyz_aligned.get_measures(0.0, 0.05, 100)
    print('Evaluation 3D KP ALIGNED results:')
    print('auc=%.10f, mean_kp3d_avg=%.10f cm\n' % (xyz_al_auc3d, xyz_al_mean3d * 100.0))   

    # Dump results
    score_path = os.path.join(output_dir, 'scores.txt')
    with open(score_path, 'w') as fo:
        xyz_mean3d *= 100
        xyz_al_mean3d *= 100
        fo.write('xyz_mean3d: %f\n' % xyz_mean3d)
        fo.write('xyz_auc3d: %f\n' % xyz_auc3d)
        fo.write('xyz_al_mean3d: %f\n' % xyz_al_mean3d)
        fo.write('xyz_al_auc3d: %f\n' % xyz_al_auc3d)

    print('Scores written to: %s' % score_path)
    print('Evaluation complete.')
        

if __name__ == '__main__':
    gt_path = "../data/FreiHand"
    input_dir = "."
    output_dir = "../output/result"
    pred_file_name = "pred.npy"
    
    # TODO: as of now, we are using part of the training set as an evaluation set.
    main(
        os.path.join(gt_path),
        os.path.join(input_dir),
        os.path.join(output_dir),
        pred_file_name,
        set_name='training'
    )