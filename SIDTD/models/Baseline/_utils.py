import matplotlib
matplotlib.use('Agg')

import os

import time 
import timm 
import torch 
import random
import sklearn.metrics 
import matplotlib.pyplot as plt 
from contextlib import contextmanager 

import numpy as np 
import pandas as pd 
import torch.nn as nn 
import torch.nn.functional as F 


import torchvision.models as models 

from sklearn.metrics import f1_score, accuracy_score 

from efficientnet_pytorch import EfficientNet

import csv 


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

def get_FPR_FNR(actual, pred):

    """
    Compute FPR and FNR metrics.

    Parameters
    ----------
    actual : data label
    pred : predicted label

    Returns
    -------
    False Positive Rate (FPR) and False Negative Rate (FNR).

    """
    
    df = pd.DataFrame({ 'actual': np.array(actual),  
                    'predicted': np.asarray(pred)})

    TP = df[(df['actual'] == 0) & (df['predicted'] == 0)].shape[0]   # compute True Positive number
    TN = df[(df['actual'] == 1) & (df['predicted'] == 1)].shape[0]   # compute True Negative number
    FN = df[(df['actual'] == 0) & (df['predicted'] == 1)].shape[0]   # compute False Negative number
    FP = df[(df['actual'] == 1) & (df['predicted'] == 0)].shape[0]   # compute False Positive number

    try:
        FNR = FN / (TP + FN)
    except: 
        FNR = -1
    try:
        FPR = FP / (FP + TN)
    except: 
        FPR = -1

    return FPR, FNR
    
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

def plot_loss(args, training_loss_list, validation_loss_list, validation_acc_list, training_iteration_list, training_iteration):

    """
    Helper function to plot the loss and accuracy learning curves.

    Parameters
    ----------
    args : Arguments.
        args.dataset, args.model, args.name : Parameters to decide the name of the output image file
        
    training_loss_list : List with training losses.
    validation_loss_list : List with validation losses.
    validation_acc_list : List with the validation accuracies.
    training_iteration_list: list of epochs number
    training_iteration: dataset partition number

    Returns
    -------
    None.

    """

    if not os.path.exists('plots/{}/{}/'.format(args.model, args.dataset)):
        os.makedirs('plots/{}/{}/'.format(args.model, args.dataset))
            
    plt.figure()
    plt.title("Loss")
    plt.plot(training_iteration_list, training_loss_list, label="train")
    plt.plot(training_iteration_list, validation_loss_list, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(args.plot_path + '{}/{}/{}_loss_n{}.jpg'.format(args.model, args.dataset, args.name, training_iteration))
    plt.close()
    
    plt.figure()
    plt.title("Accuracy")
    plt.plot(training_iteration_list, validation_acc_list, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(args.plot_path + '{}/{}/{}_accuracy_n{}.jpg'.format(args.model, args.dataset, args.name, training_iteration))
    plt.close()


        
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
