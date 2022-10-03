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

#custom library for the dataset functions
from dataset.custom_dataset import TrainDataset, get_transforms

@contextmanager 
def timer(name, LOGGER):
    """
    Timer for the log file

    Parameters
    ----------
    name : name of the experiment

    Returns
    -------
    None.

    """
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


    
def init_logger(log_file='train.log'):
    """
    Initialisation of  the log file to save the displayed info.

    Parameters
    ----------
    log_file : Name of the log file. The default is 'train.log'.

    Returns
    -------
    logger : Log file

    """
    from logging import getLogger, DEBUG, FileHandler, Formatter, StreamHandler
        
    log_format = '%(asctime)s %(levelname)s %(message)s'
        
    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(Formatter(log_format))
        
    file_handler = FileHandler(log_file)
    file_handler.setFormatter(Formatter(log_format))
        
    logger = getLogger('Herbarium')
    logger.setLevel(DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
        
    return logger 


def seed_torch(seed=777):
    """
    Seeds the random variables of the different libraries.

    Parameters
    ----------
    seed : int, optional
        Random seed. The default is 777.

    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 


def setup_model(args, N_CLASSES, pretrained=True):
    """
    Pytorch model setup.

    Parameters
    ----------
    args : Arguments.
        args.model : Model to train. Options: ['resnet50', 'vit_large_patch16_224', 'efficientnet-b3']
    N_CLASSES : int
        Number of classes in the dataset.
    pretrained : Bool, optional
        Use pretrained weights on imagenet. The default is True.

    Returns
    -------
    model : pytorch model

    """
    print("MODEL SETUP: ", args.model)
    if args.model == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(model.fc.in_features, N_CLASSES)
        
    elif args.model == 'vit_large_patch16_224':
        model = timm.create_model(args.model, pretrained=pretrained)
        net_cfg = model.default_cfg
        last_layer = net_cfg['classifier']
        num_ftrs = getattr(model, last_layer).in_features
        setattr(model, last_layer, nn.Linear(num_ftrs, N_CLASSES))
        
    elif args.model == 'efficientnet-b3':
        model = EfficientNet.from_pretrained(args.model)
        model._fc = nn.Linear(model._fc.in_features, N_CLASSES)
        
    return model 


def get_mean_std(args, model):
    """
    Computes the mean and std of the model with respect to imagenet.

    Parameters
    ----------
    args : Arguments.
        args.model : Model to train. Options: ['resnet50', 'vit_large_patch16_224', 'efficientnet-b3']
    model : pytorch model

    Returns
    -------
    mean, std : for pretrained models on Imagenet

    """
    if args.model in ['resnet50', 'efficientnet-b3']:
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        
    elif args.model == 'vit_large_patch16_224':
        mean = list(model.default_cfg['mean'])
        std = list(model.default_cfg['std'])    
    return mean, std
    
    

def get_dataset_info(args):
    dataset_path = args.dataset_path
    
    if args.dataset == 'DF20M':
        print(args.dataset)
        train_metadata = pd.read_csv("DF20M-train_metadata_train_id2.csv") 
        print(len(train_metadata)) 
        test_metadata = pd.read_csv("DF20M-public_test_metadata_train_id2.csv")
        print(len(test_metadata))
        train_metadata['image_path'] = train_metadata.apply(lambda x: dataset_path + x['image_path'], axis=1)
        test_metadata['image_path'] = test_metadata.apply(lambda x: dataset_path + x['image_path'], axis=1)
        train_metadata['image_path'] = train_metadata.apply(lambda x: x['image_path'].split('.')[0] + '.JPG', axis=1)
        test_metadata['image_path'] = test_metadata.apply(lambda x: x['image_path'].split('.')[0] + '.JPG', axis=1)
        return train_metadata, test_metadata
        
    elif args.dataset == 'dogs':
        train_metadata = pd.read_csv("train_dogs_dataset.csv") 
        print(len(train_metadata)) 
        test_metadata = pd.read_csv("test_dogs_dataset.csv")
        print(len(test_metadata))
        train_metadata['image_path'] = train_metadata.apply(lambda x: dataset_path + x['image_path'], axis=1)
        test_metadata['image_path'] = test_metadata.apply(lambda x: dataset_path + x['image_path'], axis=1)
        return train_metadata, test_metadata
        
    elif args.dataset == 'findit':
        train_metadata = pd.read_csv("train_findit_dataset.csv") 
        print(len(train_metadata)) 
        test_metadata = pd.read_csv("test_findit_dataset.csv")
        print(len(test_metadata))
        train_metadata['image_path'] = train_metadata.apply(lambda x: dataset_path + x['image_path'], axis=1)
        test_metadata['image_path'] = test_metadata.apply(lambda x: dataset_path + x['image_path'], axis=1)
        return train_metadata, test_metadata

    elif args.dataset == 'findit_crops':
        train_metadata = pd.read_csv("train_findit_crops_dataset.csv") 
        print(len(train_metadata)) 
        test_metadata = pd.read_csv("test_findit_crops_dataset.csv")
        print(len(test_metadata))
        train_metadata['image_path'] = train_metadata.apply(lambda x: dataset_path + x['image_path'], axis=1)
        test_metadata['image_path'] = test_metadata.apply(lambda x: dataset_path + x['image_path'], axis=1)
        return train_metadata, test_metadata
                
        
    elif args.dataset in ['banknotes', 'banknotes_crop']:
        full_metadata = pd.read_csv("full_eur_banknote_dataset.csv") 
        full_metadata['image_path'] = full_metadata.apply(lambda x: dataset_path + x['model'] + '/' + x['file_name'], axis=1)
        return full_metadata
                       


def plot_loss(args, losses, training_iteration):
    """
    Helper function to plot the loss learning curves.

    Parameters
    ----------
    args : Arguments.
        args.dataset, args.name : Parameters to decide the name of the output image file
        
    losses : Dictionary with the training and validation losses.

    Returns
    -------
    None.

    """
    if not os.path.exists('plots/'):
        os.makedirs('plots/')
            
    plt.figure()
    plt.title("Loss")
    plt.plot(losses["train"], label="train")
    plt.plot(losses["val"], label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    #plt.show()
    plt.savefig('plots/{}_{}_n{}.jpg'.format(args.dataset, args.name, training_iteration))
    plt.close()
    
    
def trainer(args, training_iteration, model, optimizer, scheduler, criterion, n_epochs, BATCH_SIZE, N_CLASSES,
            accumulation_steps, train_loader, valid_loader, save_model_path, device, LOGGER):
    """
    Training loop

    Parameters
    ----------
    model : Pytorch model
    optimizer
    scheduler
    criterion
    n_epochs : Number of epochs.
    BATCH_SIZE
    N_CLASSES : Number of classes in the dataset
    accumulation_steps : number of steps to update weights
    train_loader : pytorch dataloader
    valid_loader : pytorch dataloader
    save_model_path : pytorch dataloader
    device : cuda or cpu
    LOGGER : Logger file to write training info
    
    Returns
    -------
    losses : Dictionary with the training and validation losses.
    best_loss
    best_loss_epoch
    best_accuracy
    best_accuracy_epoch
    """
    
    with timer('Train model', LOGGER):
        
        model.to(device)
        best_accuracy = 0.
        best_accuracy_epoch = 0
        best_loss = np.inf
        best_loss_epoch = 0
        best_roc_auc = 0
        losses = {"train": [], "val": []}
        for epoch in range(n_epochs):
        
            start_time = time.time()
            
            model.train()
            avg_loss = 0.

            optimizer.zero_grad()
            
            for i, (images, labels) in tqdm.tqdm(enumerate(train_loader)):
            
                images = images.to(device)
                labels = labels.to(device)
                
                y_preds = model(images)
                loss = criterion(y_preds, labels)
                
                 # Scale the loss to the mean of the accumulated batch size
                loss = loss / accumulation_steps
                loss.backward()
                if (i - 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                    avg_loss += loss.item() / len(train_loader)
                
            #Evaluation
            model.eval()
            avg_val_loss = 0.
            preds = []
            p_preds = []
            reals = np.zeros((len(valid_loader.dataset)))
            
            for i, (images, labels) in enumerate(valid_loader):
             
                images = images.to(device)
                labels = labels.to(device)
                reals[i * BATCH_SIZE: (i+1) * BATCH_SIZE] = labels.to('cpu').numpy()
             
                with torch.no_grad():
                    y_preds = model(images)
               
                preds[i * BATCH_SIZE: (i+1) * BATCH_SIZE] = y_preds.argmax(1).to('cpu').numpy()
                p_pred = F.softmax(y_preds, dim=1)
               
                if args.dataset in ['banknotes', 'findit', 'findit_crops', 'banknotes_crop']:
                    p_preds.extend(p_pred[:,1].to('cpu').numpy())
                else:
                    p_preds.extend(p_pred.to('cpu').numpy())
                    
                loss = criterion(y_preds, labels)
                avg_val_loss += loss.item() / len(valid_loader)

            scheduler.step(avg_val_loss)
            
            score = f1_score(reals, preds, average='macro')
            accuracy = accuracy_score(reals, preds)           
            try:
                if args.dataset in ['banknotes', 'findit', 'findit_crops', 'banknotes_crop']:
                    roc_auc_score = sklearn.metrics.roc_auc_score(reals, p_preds)
                else:
                    roc_auc_score = sklearn.metrics.roc_auc_score(reals, p_preds, labels = np.arange(0,N_CLASSES), multi_class='ovr')
            except:
                roc_auc_score = -1
            
            elapsed = time.time() - start_time
        
            LOGGER.debug(f' Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f} avg_val_loss: {avg_val_loss:.4f} F1: {score:.6f} Accuracy: {accuracy:.6f} Roc AUC: {roc_auc_score:.6f} time: {elapsed:.0f}s')
                     
            
            if accuracy>best_accuracy:
                best_accuracy = accuracy
                best_accuracy_epoch = epoch
                LOGGER.debug(f' Epoch {epoch+1} - Save Best Accuracy: {best_accuracy:.6f} Model')
                torch.save(model.state_dict(), save_model_path + '/{}_{}_best_accuracy_n{}.pth'.format(args.dataset, args.name, training_iteration))
            
            if roc_auc_score>=best_roc_auc:
                best_roc_auc = roc_auc_score
                best_roc_auc_epoch = epoch
                LOGGER.debug(f' Epoch {epoch+1} - Save Best Roc AUC: {best_roc_auc:.6f} Model')
                torch.save(model.state_dict(), save_model_path + '/{}_{}_best_rocAUC_n{}.pth'.format(args.dataset, args.name, training_iteration))
                
            if avg_val_loss<best_loss:
                best_loss = avg_val_loss
                best_loss_epoch = epoch
                LOGGER.debug(f' Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
                torch.save(model.state_dict(), save_model_path + '/{}_{}_best_loss_n{}.pth'.format(args.dataset, args.name, training_iteration))
            losses['train'].append(avg_loss)
            losses['val'].append(avg_val_loss)
            
    torch.save(model.state_dict(), save_model_path + '/{}_{}_best_100E_n{}.pth'.format(args.dataset, args.name, training_iteration))
    
    return losses, best_loss, best_loss_epoch, best_accuracy, best_accuracy_epoch


        
def test(args, model, device, criterion, test_loader, N_CLASSES, BATCH_SIZE, LOGGER):
    """
    Test the best model against the test data

    Parameters
    ----------
    model : Pytorch model.
    device : Cuda or cpu
    criterion 
    test_loader : Pytorch dataloader
    N_CLASSES : Number of classes in the dataset
    BATCH_SIZE
    LOGGER : Logger file to write training info

    Returns
    -------
    avg_val_loss : Average loss for the test set
    accuracy
    score : F1
    roc_auc_score : ROC area under the curve

    """                  
    #Evaluation
    model.eval()
    avg_val_loss = 0.
    preds = np.zeros((len(test_loader.dataset)))
    reals = np.zeros((len(test_loader.dataset)))
    p_preds = []
    
    for i, (images, labels) in enumerate(test_loader):
     
        images = images.to(device)
        labels = labels.to(device)
        reals[i * BATCH_SIZE: (i+1) * BATCH_SIZE] = labels.to('cpu').numpy()     
     
        with torch.no_grad():
            y_preds = model(images)
                      
        preds[i * BATCH_SIZE: (i+1) * BATCH_SIZE] = y_preds.argmax(1).to('cpu').numpy()

        p_pred = F.softmax(y_preds, dim=1)
        if args.dataset in ['banknotes', 'findit', 'findit_crops', 'banknotes_crop']:
            p_preds.extend(p_pred[:,1].to('cpu').numpy())
        else:
            p_preds.extend(p_pred.to('cpu').numpy())
        
        loss = criterion(y_preds, labels)
        avg_val_loss += loss.item() / len(test_loader)
    
    score = f1_score(reals, preds, average='macro')
    accuracy = accuracy_score(reals, preds)
    try: 
        if args.dataset in ['banknotes', 'findit', 'findit_crops', 'banknotes_crop']:
            roc_auc_score = sklearn.metrics.roc_auc_score(reals, p_preds)
        else:
            roc_auc_score = sklearn.metrics.roc_auc_score(reals, p_preds, labels = np.arange(0,N_CLASSES), multi_class='ovr')
    except:
        roc_auc_score = -1

    
    LOGGER.debug(f'TESTING: avg_test_loss: {avg_val_loss:.4f} F1: {score:.6f}  Accuracy: {accuracy:.6f} roc_auc_score: {roc_auc_score:.6f}') 

    return avg_val_loss, accuracy, score, roc_auc_score
            
def save_results_setup(args):
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
    if not os.path.exists("results_files/"):
        os.makedirs("results_files/")
        
    print("Results file: ", 'results_files/{}_{}_val_results.csv'.format(args.dataset, args.name))
    f_val = open('results_files/{}_{}_val_results.csv'.format(args.dataset, args.name), 'w')
    # create the csv writer
    writer_val = csv.writer(f_val)
    header_val = ['iteration', 'best_loss', 'best_loss_epoch', 'best_accuracy', 'best_accuracy_epoch', 'best_AUC', 'best_AUC_epoch'] 
    writer_val.writerow(header_val)

    f_test = open('results_files/{}_{}_test_results.csv'.format(args.dataset, args.name), 'w')
    # create the csv writer
    writer_test = csv.writer(f_test)
    header_test = ['iteration', 'loss', 'accuracy', 'F1_score', 'roc_auc_score']
    writer_test.writerow(header_test)
    return f_val, f_test, writer_val, writer_test
