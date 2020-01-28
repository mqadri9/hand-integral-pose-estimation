
#######################################
####### PANET Specific  config ########
#######################################

class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)



aug_rotate_val=0.15
augmentation=0
bsize=500
disp_freq=100
encode_with_relu=1
epoch=10
logdir=None
lr=0.0001
r_decay_rate=0.95
lr_decay_step=10000
maxitr=200000
predicted_PANet_path='../logs/hand_pa/pred_PANet.npy'
pretrain_model='/home/mqadri/hand-integral-pose-estimation/output/PANet_model/model_best.pth'
print_freq=10
regressor_output='../local_data/hand_test.npy'
root=None
save_freq=1000
seed_id=0
thread_num=4
trlist=None
validation_freq=1000
which=-2
pts_num=21

args = Namespace(aug_rotate_val=aug_rotate_val,
                 augmentation=augmentation,
                 bsize = bsize,
                 disp_freq = disp_freq,
                 encode_with_relu=encode_with_relu,
                 epoch = epoch,
                 logdir=logdir,
                 lr=lr,
                 r_decay_rate=r_decay_rate,
                 lr_decay_step=lr_decay_step,
                 maxitr=maxitr,
                 predicted_PANet_path=predicted_PANet_path,
                 pretrain_model=pretrain_model,
                 print_freq=print_freq,
                 regressor_output=regressor_output,
                 root=root,
                 save_freq=save_freq,
                 seed_id=seed_id,
                 thread_num=thread_num,
                 trlist=trlist,
                 validation_freq=validation_freq,
                 which=which
                 )