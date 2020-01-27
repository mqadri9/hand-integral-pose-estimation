#### Required args for data generation ####

datasplit_ratio=0.1

dataset=LSMV

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
dataset_size=32560
Frei_sample_version=gs
num_keypoints=21
aspect_ratio=1.0
scaling_constant=100
localdir=../local_data/


nice -10 python3 data_main.py --dataset=$dataset \
							  --dataset_dir=$dataset_dir \
							  --datasplit_ratio=$datasplit_ratio \
							  --fix_split_flag=$fix_split_flag \
							  --aspect_ratio=$aspect_ratio \
							  --num_keypoints=$num_keypoints \
							  --scaling_constant=$scaling_constant \
							  --Frei_sample_version=$Frei_sample_version \
							  --localdir=$localdir
