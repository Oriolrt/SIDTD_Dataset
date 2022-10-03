#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python training_kfold_rectified.py --name='EfficientNet' --dataset='dataset_raw' --model='efficientnet-b3'

CUDA_VISIBLE_DEVICES=0 python training_kfold_rectified.py --name='ResNet50' --dataset='dataset_raw' --model='resnet50'

CUDA_VISIBLE_DEVICES=0 python training_kfold_rectified.py --name='vit_large_patch16' --dataset='dataset_raw' --model='vit_large_patch16_224'

CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python train_kfold_transfg.py --name='trans_fg' --dataset='dataset_raw'

CUDA_VISIBLE_DEVICES=0 python train_coAttn_binary.py --name='coatten_fcn_model' --dataset 'dataset_raw'