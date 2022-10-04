# coding=utf-8
from __future__ import absolute_import, division, print_function 

import matplotlib 
matplotlib.use('Agg') 
import sys
import os

hard_path = ''
for x in os.getcwd().split('/')[1:-1]: hard_path = hard_path + '/' + x
complete_path = hard_path + '/models/transfg/'
sys.path.insert(1, complete_path)

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
from utils_transfg.data_utils import get_loader_test



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



def test(args, LOGGER, model, test_loader):
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
        
    return eval_losses.avg, val_accuracy, roc_auc_score





def test_transfg_models(args, LOGGER, iteration):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE USED: ", device)
    print('path', os.getcwd())
    args.device = device
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
            header_test = ['iteration', 'loss', 'accuracy', 'roc_auc_score']
            writer_test.writerow(header_test)
    
    seed_torch(seed=777)
    ### start iteration here
    LOGGER.info("----- STARTING NEW ITERATION -----")
    LOGGER.info("Iteration = {}".format(iteration))
    # Model & Tokenizer Setup; prepare the dataset from scratch with imagenet weights at each iteration
    args, model, num_classes = setup(args, LOGGER)
    test_loader = get_loader_test(args, iteration)

    save_model_path = args.save_model_path + args.model + "_trained_models/" + args.dataset + "/"
    model_checkpoint = os.path.join(save_model_path,
                                    '{}_{}_best_accuracy_n{}.pth'.format(args.dataset,
                                                              args.name,
                                                              iteration))
    
    model.load_state_dict(torch.load(model_checkpoint))
    with torch.no_grad():
        test_loss, test_accuracy, test_roc_auc_score = test(args, model, test_loader, num_classes)


    if args.save_results:
        
        test_res = [iteration, test_loss, test_accuracy, test_roc_auc_score]
        
        writer_test.writerow(test_res)


    if args.save_results:
        f_test.close() 