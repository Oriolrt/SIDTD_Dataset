# coding=utf-8
from __future__ import absolute_import, division, print_function 

import sys
import os

hard_path = ''
for x in os.getcwd().split('/')[1:-1]: hard_path = hard_path + '/' + x
complete_path = hard_path + '/models/transfg/'
sys.path.insert(1, complete_path)

import matplotlib 
matplotlib.use('Agg') 

import logging 
import argparse 
import random 
import numpy as np 
import time 
import sklearn.metrics 
import csv 
import os.path

from contextlib import contextmanager
import matplotlib.pylab as plt 
import torch 
import torch.nn.functional as F 
from tqdm import tqdm 


#from torch.utils.tensorboard import SummaryWriter 
from models.modeling import VisionTransformer, CONFIGS 
from utils_transfg.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils_transfg.data_utils import get_loader



@contextmanager 
def timer(name, LOGGER):
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')


def seed_torch(seed=777):
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
    return (preds == labels).mean()


def save_model(args, LOGGER, model, best_feature, training_iteration):
    
    save_model_path = args.save_model_path + args.model + "_trained_models/" + args.dataset + "/"
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(save_model_path,
                                    '{}_{}_{}_n{}.pth'.format(args.dataset,
                                                              args.name,
                                                              best_feature,
                                                              training_iteration))
    
    checkpoint = model_to_save.state_dict()
    torch.save(checkpoint, model_checkpoint)
    LOGGER.info("Saved model checkpoint to [DIR: %s]", model_checkpoint)



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

def plot_loss_acc(args, training_loss_list, validation_loss_list, validation_acc_list, training_iteration_list, training_iteration):

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



def valid(args, LOGGER, model, test_loader, num_classes):
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
            #deppending on multiclass or binnary classification add the info to calculate ROC AUC
            p_preds.extend(p_pred[:,1].to('cpu').numpy())

            
            eval_loss = loss_fct(logits, y)
            eval_loss = eval_loss.mean()
            eval_losses.update(eval_loss.item())
            
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
    #print(all_preds, all_label)
    accuracy = simple_accuracy(all_preds, all_label)
    accuracy = torch.tensor(accuracy).to(args.device)
    val_accuracy = accuracy.detach().cpu().numpy()
    try:
        roc_auc_score = sklearn.metrics.roc_auc_score(all_label, p_preds)
    except:
        roc_auc_score = -1
        
        
    LOGGER.info("\n")
    LOGGER.info("Validation Results")
    LOGGER.info("Valid Loss: %2.5f" % eval_losses.avg)
    LOGGER.info("Valid Accuracy: %2.5f" % val_accuracy)
    LOGGER.info("Valid ROC AUC score: %2.5f" % roc_auc_score)
        
    return eval_losses, val_accuracy, roc_auc_score




def train(args, LOGGER, model, num_classes, training_iteration, writer_val):
    """ Train the model """
     
    args.train_batch_size_total = args.train_batch_size * args.gradient_accumulation_steps
    
    #dataloaders according to the iteration split
    train_loader, val_loader = get_loader(args, training_iteration)
    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate_sgd,
                                momentum=0.9,
                                weight_decay=args.weight_decay)
    t_total = int(args.epochs * len(train_loader)/args.train_batch_size_total)
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    # Train!
    LOGGER.info("***** Running training *****")
    LOGGER.info(" Instantaneous batch size per GPU = %d", args.train_batch_size)
    LOGGER.info(" Total train batch size = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    LOGGER.info(" Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    model.zero_grad()
    
    losses_dict = {"train": [], "val": []}
    losses = AverageMeter()
    global_step, best_acc = 0, 0
    best_loss = 9999999
    best_roc_AUC = 0
    start_time = time.time()
    best_loss_epoch = 0
    best_acc_epoch = 0
    best_roc_AUC_epoch = 0

    training_iteration_list = []
    validation_loss_list = []
    training_loss_list = []
    validation_acc_list = []
    training_acc_list = []
    
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        
        
        for step, batch in tqdm(enumerate(train_loader)):
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            x, y = batch
            loss, logits = model(x, y)
            loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                
            loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                
                losses.update(loss.item()*args.gradient_accumulation_steps)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                

                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
        #EVALUATION on validation dataset
        with torch.no_grad():
            eval_losses, val_accuracy, roc_auc_score = valid(args, LOGGER, model, val_loader, num_classes)

        
        training_iteration_list = training_iteration_list + [epoch]
        validation_loss_list = validation_loss_list + [eval_losses.avg]
        training_loss_list = training_loss_list + [losses.avg]
        validation_acc_list = validation_acc_list + [val_accuracy]
        
        plot_loss_acc(args, training_loss_list, validation_loss_list, validation_acc_list, training_iteration_list, training_iteration)
        
        LOGGER.info(f' Epoch {epoch+1} - avg_train_loss: {losses.avg:.4f} avg_val_loss: {eval_losses.avg:.4f} Accuracy: {val_accuracy:.6f} Roc AUC: {roc_auc_score:.6f}')
        if best_acc <= val_accuracy:
            best_feature = 'best_accuracy'
            save_model(args, LOGGER, model, best_feature, training_iteration)
            best_acc = val_accuracy
            best_acc_epoch = epoch
            LOGGER.info("best accuracy: %f" % best_acc)
        if best_roc_AUC <= roc_auc_score:
            best_feature = 'best_ROC_AUC'
            best_roc_AUC = roc_auc_score
            best_roc_AUC_epoch = epoch
            LOGGER.info("best ROC AUC: %f" % best_roc_AUC)
        if best_loss >= eval_losses.avg:
            best_feature = 'best_loss'
            best_loss = eval_losses.avg
            best_loss_epoch = epoch
            LOGGER.info("best loss: %f" % best_loss)
            
        losses_dict['train'].append(losses.avg)
        losses_dict['val'].append(eval_losses.avg)
            
        losses.reset()
    LOGGER.info("End Training!")
    end_time = time.time()
    LOGGER.info("Total Training Time: \t%f" % ((end_time - start_time) / 3600))

    if args.save_results:
        val_res = [training_iteration, best_loss, best_loss_epoch, best_acc,
                   best_acc_epoch, best_roc_AUC, best_roc_AUC_epoch]
                
        writer_val.writerow(val_res)



def train_transfg_models(args, LOGGER, iteration):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE USED: ", device)
    print('path', os.getcwd())
    args.device = device
    if args.save_results:
        if not os.path.exists(args.results_path + '{}/{}/'.format(args.model, args.dataset)):
            os.makedirs(args.results_path + '{}/{}/'.format(args.model, args.dataset))
        
        print("Results file: ", args.results_path + '{}/{}/{}_val_results.csv'.format(args.model, args.dataset, args.name))
        
        if os.path.isfile(args.results_path + '{}/{}/{}_val_results.csv'.format(args.model, args.dataset, args.name)):
            f_val = open(args.results_path + '{}/{}/{}_val_results.csv'.format(args.model, args.dataset, args.name), 'a')
            writer_val = csv.writer(f_val)
        else:
            f_val = open(args.results_path + '{}/{}/{}_val_results.csv'.format(args.model, args.dataset, args.name), 'w')
            # create the csv writer
            writer_val = csv.writer(f_val)
            header_val = ['iteration', 'best_loss', 'best_loss_epoch', 'best_accuracy', 'best_accuracy_epoch', 'best_AUC', 'best_AUC_epoch']
            writer_val.writerow(header_val)
        
    
    seed_torch(seed=777)
### start iteration here
    LOGGER.info("----- STARTING NEW ITERATION -----")
    LOGGER.info("Iteration = {}".format(iteration))
    # Model & Tokenizer Setup; prepare the dataset from scratch with imagenet weights at each iteration
    args, model, num_classes = setup(args, LOGGER)
    
    train(args, LOGGER, model, num_classes, iteration, writer_val)

    
    if args.save_results:
        f_val.close()
