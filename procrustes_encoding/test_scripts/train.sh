ae_config=0
bsize=500
weight=0
seed_id=0

datasplit_ratio=0.1
dataset_id=hand
logdir=../logs/hand_pa/
datadir=/dataset/mosam/hand_data/hand.npy
localdir=../local_data/
fix_split_flag=1


encode_with_relu=1
data_augmentation=1
aug_rotate_val=0.52

python3 ../data_splitting.py --datadir=$datadir \
							 --train_split_proportion=$datasplit_ratio \
							 --fix_split_flag=$fix_split_flag \
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
