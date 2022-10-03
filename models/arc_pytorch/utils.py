import matplotlib
matplotlib.use('Agg')

import gc
import os
import pickle
import cv2
import sys 
import json 
import time 
import timm 
import torch 
import random
import argparse
import sklearn.metrics 
import matplotlib.pyplot as plt 

from PIL import Image 
from pathlib import Path 
from functools import partial 
from contextlib import contextmanager 
from sklearn.model_selection import KFold, StratifiedKFold

import numpy as np 
import scipy as sp 
import pandas as pd 
import torch.nn as nn 

from torch.optim import Adam, SGD, AdamW 
from torch.optim.lr_scheduler import CosineAnnealingLR 
from torch.utils.data import DataLoader, Dataset 
import torch.nn.functional as F 

from albumentations import Compose, Normalize, Resize 
from albumentations.pytorch import ToTensorV2 
from albumentations import RandomCrop, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, CenterCrop, PadIfNeeded, RandomResizedCrop

import torchvision.models as models 

from torch.optim.lr_scheduler import ReduceLROnPlateau 
from sklearn.metrics import f1_score, accuracy_score, top_k_accuracy_score 

from efficientnet_pytorch import EfficientNet

import tqdm 
import csv 

def plot_loss(opt, training_loss_list, training_acc_list, validation_loss_list, validation_acc_list, training_iteration_list, iteration):

    if not os.path.exists('plots/{}/'.format(opt.dataset)):
        os.makedirs('plots/{}/'.format(opt.dataset))
            
    plt.figure()
    plt.title("Loss")
    plt.plot(training_iteration_list, training_loss_list, label="training")
    plt.plot(training_iteration_list, validation_loss_list, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig('plots/{}/loss_{}_n{}.jpg'.format(opt.dataset, opt.name,iteration))
    plt.close()
    
    plt.figure()
    plt.title("Accuracy")
    plt.plot(training_iteration_list, training_acc_list, label="training")
    plt.plot(training_iteration_list, validation_acc_list, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig('plots/{}/accuracy_{}_n{}.jpg'.format(opt.dataset, opt.name,iteration))
    plt.close()
    
    
def save_results_setup(opt):
    """
    Helper function to create the files to save the final results for each iteration of the kfold
    training as a csv file.

    Parameters
    ----------
    args : Arguments
        args.dataset, args.name : Parameters to decide the name of the output image file

    Returns
    -------
    f_val : file for validation
    f_test : file for test
    writer_val : writer for validation
    writer_test : writer for test

    """
    if not os.path.exists("results_files/{}".format(opt.dataset)):
        os.makedirs("results_files/{}".format(opt.dataset))
        
    print("Results file: ", 'results_files/{}/{}_val_results.csv'.format(opt.dataset, opt.name))

    if os.path.isfile('results_files/{}/{}_val_results.csv'.format(opt.dataset, opt.name)):
        f_val = open('results_files/{}/{}_val_results.csv'.format(opt.dataset, opt.name), 'a')
        writer_val = csv.writer(f_val)
    
    else:
        f_val = open('results_files/{}/{}_val_results.csv'.format(opt.dataset, opt.name), 'w')
        # create the csv writer
        writer_val = csv.writer(f_val)
        header_val = ['training_iteration', 'best_loss', 'best_accuracy', 'best_auc_roc'] 
        writer_val.writerow(header_val)

    if os.path.isfile('results_files/{}/{}_test_results.csv'.format(opt.dataset, opt.name)):
        f_test = open('results_files/{}/{}_test_results.csv'.format(opt.dataset, opt.name), 'a')
        writer_test = csv.writer(f_test)
    else:
        f_test = open('results_files/{}/{}_test_results.csv'.format(opt.dataset, opt.name), 'w')
        # create the csv writer
        writer_test = csv.writer(f_test)
        header_test = ['training_iteration', 'loss', 'accuracy', 'roc_auc_score']
        writer_test.writerow(header_test)
    
    return f_val, f_test, writer_val, writer_test