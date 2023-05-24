# Train EfficientNet model
#CUDA_VISIBLE_DEVICES=0 python test.py --name='EfficientNet_faker_data_augmentation' --dataset='clip_cropped_MIDV2020' --model='efficientnet-b3' --type_split kfold --nsplits 1 --static no --pretrained no

# Train Trans FG model
#CUDA_VISIBLE_DEVICES=0 python test.py --name='trans_fg' --dataset='clip_cropped_MIDV2020' --model='trans_fg' --type_split kfold --nsplits 1 --static no --pretrained no

CUDA_VISIBLE_DEVICES=0 python test.py --name='coatten_fcn_model' --dataset='clip_cropped_MIDV2020' --model='coatten_fcn_model' --type_split kfold --nsplits 1 --static no --pretrained no