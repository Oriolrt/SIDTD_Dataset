#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python train_coAttn_binary.py --name='coatten_fcn_model' --dataset 'dataset_raw'
