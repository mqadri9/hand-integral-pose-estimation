
#### Required args for data generation ####
datasplit_ratio=0.1
imgdir=/dataset/mosam/FreiHAND_pub_v2
fix_split_flag=1
image_dataset_size=32560
image_sample_version=gs
num_keypoints=21
aspect_ratio=1.0
scaling_constant=100
localdir=../local_data/

#### Required args for PANet ####
ae_config=0
bsize=500
weight=0
seed_id=0
dataset_id=hand
logdir=../logs/hand_pa/
datadir=/dataset/mosam/hand_data/hand.npy
encode_with_relu=1
data_augmentation=1
aug_rotate_val=0.26

nice -10 python3 ../processing/PANet_data_generation.py --imgdir=$imgdir \
										  				--train_split_proportion=$datasplit_ratio \
										  				--fix_split_flag=$fix_split_flag \
										  				--image_dataset_size=$image_dataset_size \
										  				--aspect_ratio=$aspect_ratio \
										  				--num_keypoints=$num_keypoints \
										  				--scaling_constant=$scaling_constant \
										  				--image_sample_version=$image_sample_version \
										  				--localdir=$localdir


CUDA_VISIBLE_DEVICES=$1 nice -10 python3 ../train.py --mode=train \
	--logdir=$logdir \
	--lr=0.0001 \
	--bsize=$bsize \
	--dataset_id=$dataset_id \
	--save_freq=1000 \
	--print_freq=50 \
	--encode_with_relu=$encode_with_relu \
	--augmentation=$data_augmentation \
	--aug_rotate_val=$aug_rotate_val \
	--lr_decay_step=100000 \
	--maxitr=1000000 \
	--seed_id=$seed_id \
	--ae_config=$ae_config \
	--weight=$weight \
	--localdir=$localdir
