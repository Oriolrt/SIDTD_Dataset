# coding=utf-8
from __future__ import absolute_import, division, print_function 
from contextlib import contextmanager
from tqdm import tqdm


import matplotlib 
matplotlib.use('Agg')
import os 
import random
import time
import csv
import os.path
import torch
import sklearn.metrics
import numpy as np
import pandas as pd
import torch.nn.functional as F

from .utils import *
from .models.modeling import VisionTransformer, CONFIGS
from .utils_transfg.data_utils import get_loader_test

def test(args, LOGGER, model, test_loader):

    """
    Inference loop

    Parameters
    ----------
    model : Pytorch model
    test_loader : pytorch dataloader
    LOGGER : Logger file to write training info
    
    Returns
    -------
    eval_losses: inference loss
    val_accuracy: inference accuracy
    roc_auc_score: inference ROC AUC
    FPR: inference False Positive Rate
    FNR: inference False Negative Rate
    """

    # Validation!
    eval_losses = AverageMeter()
    LOGGER.info("***** Running Validation *****")
    LOGGER.info(" Num steps = %d", len(test_loader))
    LOGGER.info(" Batch size = %d", args.eval_batch_size)
    model.eval()
    all_preds, all_label = [], []
    p_preds = []
    loss_fct = torch.nn.CrossEntropyLoss()
    
    for step, batch in tqdm(enumerate(test_loader)):
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        
        with torch.no_grad():
            logits = model(x)
            #predicted index
            preds = torch.argmax(logits, dim=-1)
            #probabilities
            p_pred = F.softmax(logits, dim=-1)
            p_preds.extend(p_pred[:,1].to('cpu').numpy())

            
            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())
            
        # Add predictions and labels groundtruth to list  
        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
    all_preds, all_label = all_preds[0], all_label[0]
    
    # Compute accuracy, ROC AUC, FPR and FNR 
    accuracy = simple_accuracy(all_preds, all_label)
    accuracy = torch.tensor(accuracy).to(args.device)
    val_accuracy = accuracy.detach().cpu().numpy()
    try:
        roc_auc_score = sklearn.metrics.roc_auc_score(all_label, p_preds)
    except:
        roc_auc_score = -1
        
    FPR, FNR = get_FPR_FNR(actual = all_label, pred = all_preds)
        
    LOGGER.info("\n")
    LOGGER.info("Validation Results")
    LOGGER.info("Valid Loss: %2.5f" % eval_losses.avg)
    LOGGER.info("Valid Accuracy: %2.5f" % val_accuracy)
    LOGGER.info("Valid ROC AUC score: %2.5f" % roc_auc_score)
        
    return eval_losses.avg, val_accuracy, roc_auc_score, FPR, FNR



def test_transfg_models(args, LOGGER, iteration=0):
    
    """ Here we set the parameters and features for the inference."""
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE USED: ", device)
    print('path', os.getcwd())
    args.device = device

    # Load csv results file if it exist otherwise, create the results file
    if args.save_results:
        if not os.path.exists(args.results_path + '{}/{}/'.format(args.model, args.dataset)):
            os.makedirs(args.results_path + '{}/{}/'.format(args.model, args.dataset))
        
        print("Results file: ", args.results_path + '{}/{}/{}_test_results.csv'.format(args.model, args.dataset, args.name))
        
        if os.path.isfile(args.results_path + '{}/{}/{}_test_results.csv'.format(args.model, args.dataset, args.name)):
            f_test = open(args.results_path + '{}/{}/{}_test_results.csv'.format(args.model, args.dataset, args.name), 'a')
            writer_test = csv.writer(f_test)
        else:
            f_test = open(args.results_path + '{}/{}/{}_test_results.csv'.format(args.model, args.dataset, args.name), 'w')
            # create the csv writer
            writer_test = csv.writer(f_test)
            header_test = ['iteration', 'loss', 'accuracy', 'roc_auc_score', 'FPR', 'FNR']
            writer_test.writerow(header_test)
    
    seed_torch(seed=777)
    ### start iteration here
    LOGGER.info("----- STARTING NEW ITERATION -----")
    LOGGER.info("Iteration = {}".format(iteration))
    # Model & Tokenizer Setup; prepare the dataset from scratch with imagenet weights at each iteration
    args, model, num_classes = setup(args, LOGGER)
    test_loader = get_loader_test(args, iteration)

    # Load trained models.
    # Choose model depending on if you want to choose a custom trained model or perform inference with our models
    if args.pretrained == 'yes':
        if args.type_data == 'clips_cropped':
            save_model_path = os.getcwd() + "/pretrained_models/unbalanced_clip_cropped_SIDTD/trans_fg_trained_models/"
            model_checkpoint = os.path.join(save_model_path,
                                    'clip_cropped_MIDV2020_trans_fg_best_accuracy_n{}.pth'.format(iteration))
        elif args.type_data == 'clips':
            save_model_path = os.getcwd() + "/pretrained_models/unbalanced_clip_background_SIDTD/trans_fg_trained_models/"
            model_checkpoint = os.path.join(save_model_path,
                                    'clip_background_MIDV2020_trans_fg_best_accuracy_n{}.pth'.format(iteration))
        elif args.type_data == 'templates':
            save_model_path = os.getcwd() + "/pretrained_models/balanced_templates_SIDTD/trans_fg_trained_models/"
            model_checkpoint = os.path.join(save_model_path,
                                    'MIDV2020_trans_fg_best_accuracy_n{}.pth'.format(iteration))
    else:
        save_model_path = args.save_model_path + args.model + "_trained_models/" + args.dataset + "/"
        model_checkpoint = os.path.join(save_model_path,
                                    '{}_{}_best_accuracy_n{}.pth'.format(args.dataset,
                                                              args.name,
                                                              iteration))
    
    model.load_state_dict(torch.load(model_checkpoint))
    
    # Perform inference and get metrics performance
    with torch.no_grad():
        test_loss, test_accuracy, test_roc_auc_score, FPR, FNR = test(args, LOGGER, model, test_loader)


    # Write and save results in csv
    if args.save_results:
        
        test_res = [iteration, test_loss, test_accuracy, test_roc_auc_score, FPR, FNR]
        
        writer_test.writerow(test_res)


    if args.save_results:
        f_test.close() 