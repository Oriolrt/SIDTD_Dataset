#!/bin/bash
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python train_conditionned_kfold_transfg.py --name='conditioned_trans_fg' --dataset='dataset_raw'