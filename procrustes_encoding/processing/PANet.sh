
#### Required args ####
datasplit_ratio=0.1
imgdir=/dataset/mosam/FreiHAND_pub_v2
fix_split_flag=1
localdir=../local_data/

image_dataset_size=32560
image_sample_version=gs
num_keypoints=21
aspect_ratio=1.0
scaling_constant=100

nice -10 python3 PANet_data_generation.py --imgdir=$imgdir \
										  --train_split_proportion=$datasplit_ratio \
										  --fix_split_flag=$fix_split_flag \
										  --image_dataset_size=$image_dataset_size \
										  --aspect_ratio=$aspect_ratio \
										  --num_keypoints=$num_keypoints \
										  --scaling_constant=$scaling_constant \
										  --image_sample_version=$image_sample_version \
										  --localdir=$localdir