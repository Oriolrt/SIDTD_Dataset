#!/bin/bash
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python train_kfold_transfg.py --name='trans_fg' --dataset='dataset_raw'

