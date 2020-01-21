from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
import argparse
import pprint
import pdb
import time
import cv2
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim

from config import cfg as cfg2
from model_rcnn.utils.config_rcnn import cfg, cfg_from_file, cfg_from_list, get_output_dir
import torchvision.transforms as transforms
import torchvision.datasets as dset
from scipy.misc import imread
from roi_data_layer.roidb import combined_roidb
from roi_data_layer.roibatchLoader import roibatchLoader

from model_rcnn.rpn.bbox_transform import clip_boxes
# from model.nms.nms_wrapper import nms
from model_rcnn.roi_layers import nms
from model_rcnn.rpn.bbox_transform import bbox_transform_inv
from model_rcnn.utils.net_utils import save_net, load_net, vis_detections
from model_rcnn.utils.blob import im_list_to_blob
from model_rcnn.faster_rcnn.vgg16 import vgg16
from model_rcnn.faster_rcnn.resnet import resnet

 
from multiprocessing import set_start_method
try:
    set_start_method('spawn')
except RuntimeError:
    pass

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

class HandDetector:

    def __init__(self, 
                 checksession,
                 checkepoch,
                 checkpoint,
                 dataset = "FreiHand",
                 net = "res101",
                 cuda = True,
                 class_agnostic = False,
                 max_per_image = 100,
                 thresh = 0.05):
        
        np.random.seed(cfg.RNG_SEED)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.load_dir = os.path.join(dir_path, "../output/faster_rcnn_model")
        #self.image_dir = os.path.join(dir_path, "../data/FreiHand/evaluation/rgb")
        self.checksession = checksession
        self.checkepoch = checkepoch
        self.checkpoint = checkpoint
        self.dataset = dataset
        self.net = net 
        self.cuda = cuda
        self.max_per_image = max_per_image
        self.thresh = thresh
        self.class_agnostic = class_agnostic
        self.classes = np.asarray(['__background__',
                             'hand'])
        self.num_classes = len(self.classes)
        self.cfg = cfg
        
    def _get_image_blob(self, im):
      """Converts an image into a network input.
      Arguments:
        im (ndarray): a color image in BGR order
      Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
          in the image pyramid
      """
      im_orig = im.astype(np.float32, copy=True)
      im_orig -= self.cfg.PIXEL_MEANS
    
      im_shape = im_orig.shape
      im_size_min = np.min(im_shape[0:2])
      im_size_max = np.max(im_shape[0:2])
    
      processed_ims = []
      im_scale_factors = []
    
      for target_size in self.cfg.TEST.SCALES:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > self.cfg.TEST.MAX_SIZE:
          im_scale = float(self.cfg.TEST.MAX_SIZE) / float(im_size_max)
        im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                interpolation=cv2.INTER_LINEAR)
        im_scale_factors.append(im_scale)
        processed_ims.append(im)
    
      # Create a blob to hold the input images
      blob = im_list_to_blob(processed_ims)
      return blob, np.array(im_scale_factors)    
      
      
    def load_faster_rcnn_detector(self):
        input_dir = self.load_dir
        if not os.path.exists(input_dir):
          raise Exception('There is no input directory for loading network from ' + input_dir)
        
        load_name = os.path.join(input_dir,
          'faster_rcnn_{}_{}_{}.pth'.format(self.checksession, self.checkepoch, self.checkpoint))
        
        cfg_from_list(['ANCHOR_SCALES', '[4, 8, 16, 32]', 'ANCHOR_RATIOS', '[0.5,1,2]'])
        
        if self.net == 'vgg16':
          fasterRCNN = vgg16(self.classes, pretrained=False, class_agnostic=self.class_agnostic)
        elif self.net == 'res101':
          fasterRCNN = resnet(self.classes, 101, pretrained=False, class_agnostic=self.class_agnostic)
        elif self.net == 'res50':
          fasterRCNN = resnet(self.classes, 50, pretrained=False, class_agnostic=self.class_agnostic)
        elif self.net == 'res152':
          fasterRCNN = resnet(self.classes, 152, pretrained=False, class_agnostic=self.class_agnostic)
        else:
          print("network is not defined")
          pdb.set_trace()
        
        fasterRCNN.create_architecture()
        print("load checkpoint %s" % (load_name))
        if self.cuda > 0:
          checkpoint = torch.load(load_name)
        else:
          checkpoint = torch.load(load_name, map_location=(lambda storage, loc: storage))
        fasterRCNN.load_state_dict(checkpoint['model'])
        if 'pooling_mode' in checkpoint.keys():
          self.cfg.POOLING_MODE = checkpoint['pooling_mode']
        
        if self.cuda:
            self.cfg.CUDA = True
            fasterRCNN.cuda()
        fasterRCNN.eval()
        self._detector =  fasterRCNN
        print('load model successfully!')
    
    def read_image(self, image_path):
        im_in = np.array(imread(image_path))  
        if len(im_in.shape) == 2:
          im_in = im_in[:,:,np.newaxis]
          im_in = np.concatenate((im_in,im_in,im_in), axis=2)
        # rgb -> bgr
        im = im_in[:,:,::-1]
        return im        
    
    def detect(self, image_path):
        im_data = torch.FloatTensor(1)
        im_info = torch.FloatTensor(1)
        num_boxes = torch.LongTensor(1)
        gt_boxes = torch.FloatTensor(1)
        #ship to cuda
        if self.cuda > 0:
          im_data = im_data.cuda()
          im_info = im_info.cuda()
          num_boxes = num_boxes.cuda()
          gt_boxes = gt_boxes.cuda()
        
        # make variable
        im_data = Variable(im_data)
        im_info = Variable(im_info)
        num_boxes = Variable(num_boxes)
        gt_boxes = Variable(gt_boxes)        
        im = self.read_image(image_path)
        blobs, im_scales = self._get_image_blob(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"
        im_blob = blobs
        im_info_np = np.array([[im_blob.shape[1], im_blob.shape[2], im_scales[0]]], dtype=np.float32)
        
        im_data_pt = torch.from_numpy(im_blob)
        im_data_pt = im_data_pt.permute(0, 3, 1, 2)
        im_info_pt = torch.from_numpy(im_info_np)
        
        with torch.no_grad():
                im_data.resize_(im_data_pt.size()).copy_(im_data_pt)
                im_info.resize_(im_info_pt.size()).copy_(im_info_pt)
                gt_boxes.resize_(1, 1, 5).zero_()
                num_boxes.resize_(1).zero_()    
        rois, cls_prob, bbox_pred, \
        rpn_loss_cls, rpn_loss_box, \
        RCNN_loss_cls, RCNN_loss_bbox, \
        rois_label = self._detector(im_data, im_info, gt_boxes, num_boxes)
        scores = cls_prob.data
        boxes = rois.data[:, :, 1:5]    
        
        if self.cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred.data
            if self.cfg.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED:
            # Optionally normalize targets by a precomputed mean and stdev
              if self.class_agnostic:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                             + torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                  box_deltas = box_deltas.view(1, -1, 4)
              else:
                  box_deltas = box_deltas.view(-1, 4) * torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_STDS).cuda() \
                             + torch.FloatTensor(self.cfg.TRAIN.BBOX_NORMALIZE_MEANS).cuda()
                  box_deltas = box_deltas.view(1, -1, 4 * len(self.classes))
        
            pred_boxes = bbox_transform_inv(boxes, box_deltas, 1)
            pred_boxes = clip_boxes(pred_boxes, im_info.data, 1)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))
        
        pred_boxes /= im_scales[0]
        
        scores = scores.squeeze()
        pred_boxes = pred_boxes.squeeze()
        del im_data
        del im_info
        del num_boxes
        del gt_boxes
        #print("=======================")
        for j in xrange(1, len(self.classes)):
            inds = torch.nonzero(scores[:,j]>self.thresh).view(-1)
            # if there is det
            #print(inds.numel())
            if inds.numel() > 0:
              cls_scores = scores[:,j][inds]
              _, order = torch.sort(cls_scores, 0, True)
              if self.class_agnostic:
                cls_boxes = pred_boxes[inds, :]
              else:
                cls_boxes = pred_boxes[inds][:, j * 4:(j + 1) * 4]
              cls_dets = torch.cat((cls_boxes, cls_scores.unsqueeze(1)), 1)
              # cls_dets = torch.cat((cls_boxes, cls_scores), 1)
              cls_dets = cls_dets[order]
              # keep = nms(cls_dets, self.cfg.TEST.NMS, force_cpu=not self.cfg.USE_GPU_NMS)
              keep = nms(cls_boxes[order, :], cls_scores[order], self.cfg.TEST.NMS)
              cls_dets = cls_dets[keep.view(-1).long()]

        return cls_dets
    
    def visualize_and_save(self, cls_dets, save_name, image_path, save_dir):
        img = self.read_image(image_path)
        im2show = np.copy(im)
        for j in xrange(1, len(self.classes)):
            im2show = vis_detections(im2show, self.classes[j], cls_dets.cpu().numpy(), 0.5) 
        #import random
        #nn = str(random.randint(2001,3000))
        cv2.imwrite('{}/{}.jpg'.format(save_dir, save_name), im2show)        