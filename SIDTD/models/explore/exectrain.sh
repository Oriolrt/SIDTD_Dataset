# Train EfficientNet model
#CUDA_VISIBLE_DEVICES=0 python train.py --name='EfficientNet_faker_data_augmentation' --dataset='clip_cropped_MIDV2020' --model='efficientnet-b3' --type_split kfold --epochs 2 --nsplits 1 --faker_data_augmentation

# Train ResNet50 model
#CUDA_VISIBLE_DEVICES=1 nohup python train.py --name='ResNet50' --dataset='clip_cropped_MIDV2020' --model='resnet50' --type_split kfold --epochs 30  --faker_data_augmentation --nsplits 3 > test_github_ResNet_detailed.log &

# Train ViT model
#CUDA_VISIBLE_DEVICES=0 python train.py --name='vit_large_patch16' --dataset='dataset_raw' --model='vit_large_patch16_224'

# Train Trans FG model
#CUDA_VISIBLE_DEVICES=0 python train.py --name='trans_fg_faker_data_augmentation' --dataset='clip_cropped_MIDV2020' --model='trans_fg' --type_split kfold --epochs 60 --nsplits 1  --faker_data_augmentation

# Train Co-Attention ARC model
#CUDA_VISIBLE_DEVICES=0 python train.py --name='coatten_fcn_model' --dataset 'clip_cropped_MIDV2020' --model='coatten_fcn_model' --type_split kfold --nsplits 1 --n_its 1000 --batchSize 128 #> test_github_coatten_fcn_model_detailed.log &

#CUDA_VISIBLE_DEVICES=0 nohup python train.py --n_its 5000 --name='coatten_fcn_model' --type_split kfold --nsplits 1 --static='yes' --type_data templates  --dataset='SIDTD' --model='coatten_fcn_model' --batchSize 128 > log_info_coatten_fcn_model_templates.log &

#CUDA_VISIBLE_DEVICES=1 nohup python train.py --n_its 5000 --name='coatten_fcn_model' --type_split kfold --nsplits 10 --static='yes' --type_data clips_cropped  --dataset='clip_cropped_SIDTD' --model='coatten_fcn_model' --batchSize 128 > log_info_coatten_fcn_model_clips_cropped.log &

CUDA_VISIBLE_DEVICES=1 python train_fsl.py --model efficientnet-b3 --name few_shot_basic_test --episodes 200 --repetition 1