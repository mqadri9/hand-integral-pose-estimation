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
    eval_result_dir = os.path.join(output_dir, 'result', 'evaluation')
    lib_dir = os.path.join(root_dir, 'lib')
    
    input_shape = (224, 224) 
    output_shape = (input_shape[0]//4, input_shape[1]//4)
    depth_dim = input_shape[0]//4
    bbox_3d_shape = (300, 300, 300) # depth, height, width

    # training config
    lr_dec_epoch = [60, 120]
    #lr_dec_epoch = [2, 3, 4]
    # lr_dec_epoch = [80, 90]
    #end_epoch = 20
    end_epoch = 400
    #lr = 1e-3 
    lr = 1e-3
    #lr_dec_factor = 0.5
    lr_dec_factor = 0.1
    optimizer = 'adam'
    weight_decay = 1e-5
    batch_size = 32
    test_batch_size = 32
    eval_batch_size = 32
    eval_version = 2
    ## model setting
    resnet_type = 50 # 18, 34, 50, 101, 152
    patch_width = 224
    patch_height = 224
    pad_factor = 1.75
    loss = "L1"
    num_gpus = 3
    # TODO move the pixel_mean and pixel_std to the Freihand specific config file: FreiHand_config.py?
    pixel_mean = (0.4559, 0.5142, 0.5148)
    pixel_std = (1, 1, 1) #(0.2736, 0.2474, 0.2523)
    
    num_thread = 1
    
    use_hand_detector = False
    online_hand_detection = False
    checksession = 1
    checkepoch = 6
    checkpoint = 260479
    continue_train = False
    scaling_constant = 100
    
    def set_args(self, gpu_ids, continue_train=False):
        self.gpu_ids = gpu_ids
        self.num_gpus = len(self.gpu_ids.split(','))
        self.continue_train = continue_train
        os.environ["CUDA_VISIBLE_DEVICES"] = self.gpu_ids
        print('>>> Using GPU: {}'.format(self.gpu_ids))    


cfg = Config()

sys.path.insert(0, os.path.join(cfg.root_dir, 'common'))
sys.path.insert(0, os.path.join(cfg.root_dir, 'lib'))
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
make_folder(cfg.eval_result_dir)