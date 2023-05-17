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
from models.modeling import VisionTransformer, CONFIGS



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


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count 


def simple_accuracy(preds, labels):
    """ Compute accuracy based on model prediction and label groundtruth"""
    return (preds == labels).mean()


def setup(args, LOGGER):
    # Prepare model
    config = CONFIGS[args.model_type]
    config.split = args.split
    config.slide_step = args.slide_step
    if args.dataset=='dogs':
        num_classes = 120
    elif args.dataset=='DF20M':
        num_classes = 182
    else:
        num_classes = 2
    
    #prerapre model
    model = VisionTransformer(config, args.img_size, zero_head=True, num_classes=num_classes, smoothing_value=args.smoothing_value)
    
    #load the trained weights on imagenet
    model.load_from(np.load(args.pretrained_dir))
    model.to(args.device)
    
    num_params = count_parameters(model)
    LOGGER.info("{}".format(config))
    LOGGER.info("Training parameters %s", args)
    LOGGER.info("Total Parameter: \t%2.1fM" % num_params)
    return args, model, num_classes 



def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000


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

def plot_loss_acc(args, training_loss_list, validation_loss_list, validation_acc_list, training_iteration_list, training_iteration):

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