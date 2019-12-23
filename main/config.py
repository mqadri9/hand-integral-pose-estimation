import os
import sys
import numpy as np

class Config:
    trainset = ['FreiHand']
    testset = 'FreiHand'

    ## directory
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.join(cur_dir, '..')
    data_dir = os.path.join(root_dir, 'data')
    output_dir = os.path.join(root_dir, 'output')
    common_dir = os.path.join(root_dir, 'common')
    model_dir = os.path.join(output_dir, 'model_dump')
    vis_dir = os.path.join(output_dir, 'vis')
    log_dir = os.path.join(output_dir, 'log')
    result_dir = os.path.join(output_dir, 'result')
    
    input_shape = (224, 224) 
    output_shape = (input_shape[0]//4, input_shape[1]//4)
    depth_dim = 54
    bbox_3d_shape = (2000, 2000, 2000) # depth, height, width
    # training config
    lr_dec_epoch = [15, 17]
    #end_epoch = 20
    end_epoch = 1
    lr = 1e-3
    lr_dec_factor = 0.1
    optimizer = 'adam'
    weight_decay = 1e-5
    batch_size = 32
    
    ## model setting
    resnet_type = 50 # 18, 34, 50, 101, 152
    patch_width = 224
    patch_height = 224
    batch_size = 32
    loss = "L1"
    num_gpus = 3
    # TODO Need to find the real values for the Freihand dataset
    # TODO move the pixel_mean and pixel_std to the Freihand specific config file: FreiHand_config.py?
    pixel_mean = (0, 0, 0)
    pixel_std = (1, 1, 1)
    
    num_thread = 20
    
    continue_train = True
    
    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))    


cfg = Config()

sys.path.insert(0, os.path.join(cfg.root_dir, 'common'))
from utils.dir_utils import add_pypath, make_folder
add_pypath(os.path.join(cfg.data_dir))
add_pypath(os.path.join(cfg.common_dir))
for i in range(len(cfg.trainset)):
    add_pypath(os.path.join(cfg.data_dir, cfg.trainset[i]))
add_pypath(os.path.join(cfg.data_dir, cfg.testset))
make_folder(cfg.model_dir)
make_folder(cfg.vis_dir)
make_folder(cfg.log_dir)
make_folder(cfg.result_dir)