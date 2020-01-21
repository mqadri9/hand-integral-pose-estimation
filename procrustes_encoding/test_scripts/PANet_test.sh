bsize=500
pretrain_model=../logs/hand_pa/model_best.pth

CUDA_VISIBLE_DEVICES=$1 nice -10 python3 ../PANet_reconstruction.py \
	--bsize=$bsize \
	--pretrain_model=$pretrain_model \
