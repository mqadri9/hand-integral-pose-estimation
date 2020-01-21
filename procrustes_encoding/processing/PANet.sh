
#### Required args ####
datasplit_ratio=0.1
imgdir=/dataset/mosam/FreiHAND_pub_v2
fix_split_flag=1

image_dataset_size=32560
image_sample_version=gs
num_keypoints=21
root_index=9

nice -10 python3 PANet_data_generation.py --imgdir=$imgdir --train_split_proportion=$datasplit_ratio --fix_split_flag=$fix_split_flag --image_dataset_size=$image_dataset_size --num_keypoints=$num_keypoints --root_index=$root_index --image_sample_version=$image_sample_version