
#### Required args for data generation ####
datasplit_ratio=0.2
dataset=Freihand
if [[ $dataset == Freihand ]]
then
	dataset_dir=/dataset/mosam/FreiHAND_pub_v2
elif [[ $dataset == RHD ]]
then
	dataset_dir=/dataset/mosam/RHD/RHD_published_v2/
elif [[ $dataset == LSMV ]]
then
	dataset_dir=/dataset/mosam/LSMV/utils
fi
fix_split_flag=1
Frei_sample_version=gs
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
augment_with_scale=1
aug_scale_sigma_val=0.20


### Generate the training/testing data
nice -10 python3 ../processing/data_main.py --dataset=$dataset \
											--dataset_dir=$dataset_dir \
											--datasplit_ratio=$datasplit_ratio \
											--fix_split_flag=$fix_split_flag \
											--aspect_ratio=$aspect_ratio \
											--num_keypoints=$num_keypoints \
											--scaling_constant=$scaling_constant \
											--Frei_sample_version=$Frei_sample_version \
											--localdir=$localdir


### Start Deep PANet
CUDA_VISIBLE_DEVICES=$1 nice -10 python3 ../train.py --mode=train \
	--logdir=$logdir \
	--lr=0.0001 \
	--bsize=$bsize \
	--save_freq=1000 \
	--print_freq=50 \
	--encode_with_relu=$encode_with_relu \
	--augmentation=$data_augmentation \
	--augment_with_scale=$augment_with_scale \
	--aug_scale_sigma_val=$aug_scale_sigma_val \
	--aug_rotate_val=$aug_rotate_val \
	--lr_decay_step=100000 \
	--maxitr=1000000 \
	--seed_id=$seed_id \
	--ae_config=$ae_config \
	--weight=$weight \
	--localdir=$localdir
