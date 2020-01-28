ae_config=0
bsize=500
weight=0
seed_id=0

dataset_id=hand

logdir=../logs/hand_pa/
localdir=../local_data/

encode_with_relu=1
data_augmentation=1

pretrain_model=../logs/hand_pa/model_best.pth

CUDA_VISIBLE_DEVICES=$1 nice -10 python3 ../train.py --mode=test \
	--logdir=$logdir \
	--lr=0.0001 \
	--pretrain_model=$pretrain_model \
	--bsize=$bsize \
	--dataset_id=$dataset_id \
	--save_freq=1000 \
	--print_freq=50 \
	--encode_with_relu=$encode_with_relu \
	--augmentation=$data_augmentation \
	--lr_decay_step=100000 \
	--maxitr=1000000 \
	--seed_id=$seed_id \
	--ae_config=$ae_config \
	--weight=$weight \
	--localdir=$localdir
