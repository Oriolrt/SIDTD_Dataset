import matplotlib
matplotlib.use('Agg')

import os
import cv2
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import csv 
import imageio.v2 as imageio
import torch
from torch.autograd import Variable

def plot_loss(opt, training_loss_list, training_acc_list, validation_loss_list, validation_acc_list, training_iteration_list, iteration):

    """
        Helper function to plot the loss and accuracy learning curves.

        Parameters
        ----------
        args : Arguments.
            args.dataset, args.model, args.name : Parameters to decide the name of the output image file
            
        losses : Lists with the training and validation losses.
        accuracies : List with the training and validation accuracies.

        Returns
        -------
        None.

        """

    if not os.path.exists('plots/{}/{}/'.format(opt.model, opt.dataset)):
        os.makedirs('plots/{}/{}/'.format(opt.model, opt.dataset))
            
    plt.figure()
    plt.title("Loss")
    plt.plot(training_iteration_list, training_loss_list, label="train")
    plt.plot(training_iteration_list, validation_loss_list, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(opt.plot_path + '{}/{}/{}_loss_n{}.jpg'.format(opt.model, opt.dataset, opt.name, iteration))
    plt.close()
    
    plt.figure()
    plt.title("Accuracy")
    plt.plot(training_iteration_list, training_acc_list, label="training")
    plt.plot(training_iteration_list, validation_acc_list, label="validation")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.legend()
    plt.savefig(opt.plot_path + '{}/{}/{}_accuracy_n{}.jpg'.format(opt.model, opt.dataset, opt.name, iteration))
    plt.close()
    
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


def get_pct_accuracy(pred: Variable, target) -> int:

    """
    Helper function to compute accuracy metric.

    Parameters
    ----------
    pred : probabilities predicted by model.
    target : images label.

    Returns
    -------
    Accuracy rate in %

    """

    # Classification threshold is set at 0.5. So, if the probibility is lower than 0.5, the image will be classified as 0, 
    hard_pred = (pred > 0.5).int()
    correct = (hard_pred == target).sum().item()
    accuracy = float(correct) / target.size()[0]
    accuracy = int(accuracy * 100)
    return accuracy

def one_shot_eval(pred, truth): 

    """
    Helper function to compute number of data correctly predicted.

    Parameters
    ----------
    pred : probabilities predicted by model.
    truth : images label.

    Returns
    -------
    Number of data correctly predicted

    """

    pred = pred.round()
    corrects = (pred == truth).sum().item()
    return corrects 


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

    TP = df[(df['actual'] == 0) & (df['predicted'] == 0)].shape[0]
    TN = df[(df['actual'] == 1) & (df['predicted'] == 1)].shape[0]
    FN = df[(df['actual'] == 0) & (df['predicted'] == 1)].shape[0]
    FP = df[(df['actual'] == 1) & (df['predicted'] == 0)].shape[0]

    n = len(df['actual'])
    try:
        FNR = FN / (TP + FN)
    except: 
        FNR = -1
    try:
        FPR = FP / (FP + TN)
    except: 
        FPR = -1

    return FPR, FNR

def read_image(image_path, image_size):

    """
    Helper function to read image path and resize/format image.

    Parameters
    ----------
    image_path : image path where is located the chosen image
    image_size : image format to resize the image to.

    Returns
    -------
    f_test : file for test
    writer_test : writer for test

    """

    image = imageio.imread(image_path)
    if image.shape[-1]>=4:
        image = image[...,:-1]
    image = cv2.resize(image, (image_size,image_size))
    
    return np.moveaxis(image, -1, 0) 
    
def save_results_test(opt):
    """
    Helper function to create the files to save the final results for each iteration of the kfold
    training as a csv file.

    Parameters
    ----------
    args : Arguments
        args.dataset, args.name : Parameters to decide the name of the output image file

    Returns
    -------
    f_test : file for test
    writer_test : writer for test

    """
    if not os.path.exists(opt.results_path + '{}/{}/'.format(opt.model, opt.dataset)):
        os.makedirs(opt.results_path + '{}/{}/'.format(opt.model, opt.dataset))
        


    if os.path.isfile(opt.results_path + '{}/{}/{}_test_results.csv'.format(opt.model, opt.dataset, opt.name)):
        f_test = open(opt.results_path + '{}/{}/{}_test_results.csv'.format(opt.model, opt.dataset, opt.name), 'a')
        writer_test = csv.writer(f_test)
    else:
        f_test = open(opt.results_path + '{}/{}/{}_test_results.csv'.format(opt.model, opt.dataset, opt.name), 'w')
        # create the csv writer
        writer_test = csv.writer(f_test)
        header_test = ['training_iteration', 'accuracy', 'roc_auc_score', 'FPR', 'FNR']
        writer_test.writerow(header_test)
    
    return f_test, writer_test


def save_results_train(opt):
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
    writer_val : writer for validation

    """
    if not os.path.exists(opt.results_path + '{}/{}/'.format(opt.model, opt.dataset)):
        os.makedirs(opt.results_path + '{}/{}/'.format(opt.model, opt.dataset))
        
    print("Results file: ", opt.results_path + '{}/{}/{}_val_results.csv'.format(opt.model, opt.dataset, opt.name))

    if os.path.isfile(opt.results_path + '{}/{}/{}_val_results.csv'.format(opt.model, opt.dataset, opt.name)):
        f_val = open(opt.results_path + '{}/{}/{}_val_results.csv'.format(opt.model, opt.dataset, opt.name), 'a')
        writer_val = csv.writer(f_val)
    
    else:
        f_val = open(opt.results_path + '{}/{}/{}_val_results.csv'.format(opt.model, opt.dataset, opt.name), 'w')
        # create the csv writer
        writer_val = csv.writer(f_val)
        header_val = ['training_iteration', 'best_loss', 'best_accuracy', 'best_auc_roc'] 
        writer_val.writerow(header_val)

    
    return f_val, writer_val