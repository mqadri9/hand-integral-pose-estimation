bsize=500
pretrain_model=../logs/hand_pa/model_best.pth
regressor_output=../local_data/hand_test.npy
predicted_PANet_path=../logs/hand_pa/pred_PANet.npy

CUDA_VISIBLE_DEVICES=$1 nice -10 python3 ../PANet_reconstruction.py \
	--bsize=$bsize \
	--pretrain_model=$pretrain_model \
	--regressor_output=$regressor_output \
	--predicted_PANet_path=$predicted_PANet_path
