#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python test.py --name='EfficientNet' --dataset='dataset_raw' --model='efficientnet-b3'

CUDA_VISIBLE_DEVICES=0 python test.py --name='ResNet50' --dataset='dataset_raw' --model='resnet50'

CUDA_VISIBLE_DEVICES=0 python test.py --name='vit_large_patch16' --dataset='dataset_raw' --model='vit_large_patch16_224'

CUDA_VISIBLE_DEVICES=0 python test.py --name='trans_fg' --dataset='dataset_raw' --model='trans_fg'

CUDA_VISIBLE_DEVICES=0 python test.py --name='coatten_fcn_model' --dataset 'dataset_raw' --model='coatten_fcn_model'