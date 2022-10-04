import sys
import os

hard_path = ''
for x in os.getcwd().split('/')[1:-1]: hard_path = hard_path + '/' + x
complete_path = hard_path + '/models/arc_pytorch/'
sys.path.insert(1, complete_path)

import random
import numpy as np
import sklearn
import torch
from torch.autograd import Variable
import batcher_kfold_binary as batcher
from batcher_kfold_binary import Batcher
from utils import *

import models_binary
from models_binary import ArcBinaryClassifier, CustomResNet50, CoAttn


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
    hard_pred = (pred > 0.5).int()
    correct = (hard_pred == target).sum().item()
    accuracy = float(correct) / target.size()[0]
    accuracy = int(accuracy * 100)
    return accuracy

def one_shot_eval(pred, truth): 
    pred = pred.round()
    corrects = (pred == truth).sum().item()
    return corrects 


def train(opt, save_model_path, iteration):

    #Define the ResNet50 NN
    if opt.apply_fcn:
        print('Use ResNet50')
        resNet = CustomResNet50()
        
    #Define the CoAttn
    if opt.use_coAttn:
        print('Use Co Attention Model')
        coAtten = CoAttn()
             
    # initialise the model
    discriminator = ArcBinaryClassifier(num_glimpses=opt.numGlimpses,
                                        glimpse_h=opt.glimpseSize,
                                        glimpse_w=opt.glimpseSize,
                                        channels = 1024,
                                        controller_out=opt.numStates)

    if opt.cuda:
        print('Use GPU')
        discriminator.cuda()
        if opt.apply_fcn:
            resNet.cuda()
        if opt.use_coAttn:
            coAtten.cuda()

    # set up the optimizer.
    bce = torch.nn.BCELoss()
    if opt.cuda:
        bce = bce.cuda()
    #ce_loss = torch.nn.CrossEntropyLoss()

    optim_params = []
    optim_params.append(list(discriminator.parameters()))
    
    if opt.apply_fcn:
        optim_params.append(list(resNet.parameters()))
    if opt.use_coAttn:
        optim_params.append(list(coAtten.parameters()))
        
    flat_params = [item for sublist in optim_params for item in sublist]
    
    optimizer = torch.optim.Adam(params=flat_params, lr=opt.lr)

    # load the dataset in memory.
    paths_splits = {'train':{}, 'val' :{}}
    n_val = 0
    for d_set in ['train', 'val']:
        for key in ['reals','fakes']:
            path = opt.npy_dataset_path +  opt.dataset + '/'+ d_set +'_split_' + key  + '_it_' + str(iteration) + '.npy'
            data = np.load(path)
            if d_set == 'val':
                n_val = n_val + len(data)
            paths_splits[d_set][key] = list(data)
    loader = Batcher(paths_splits= paths_splits, batch_size=opt.batchSize, image_size=opt.imageSize)

    # ready to train ...
    best_validation_loss = None
    best_validation_auc = None
    best_validation_acc = None
    
    saving_threshold = 1.02
    
    training_iteration_list = []
    validation_loss_list = []
    training_loss_list = []
    
    validation_acc_list = []
    training_acc_list = []
    window = opt.batchSize
    
    for training_iteration in range(0,opt.n_its):
        
        discriminator.train(mode=True)
        

        X, Y = loader.fetch_batch("train", batch_size=opt.batchSize)
        if opt.cuda:
            X = X.cuda()
            Y = Y.cuda()
        B,P,C,W,H=X.size()
        
        if opt.apply_fcn:
            resNet.train()
            X = resNet(X.view(B*P,C,W,H))
            _,C,W,H = X.size()
            X =X.view(B,P,C,W,H)
            
        # CoAttention Module 
        if opt.use_coAttn:
            coAtten.train()
            X = coAtten(X)
            
        pred = discriminator(X)
        loss = bce(pred, Y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if training_iteration % 50 == 0:
        
            discriminator.eval()
            if opt.apply_fcn:
                resNet.eval()
            if opt.use_coAttn:
                coAtten.eval()
            
            # validate your model
            acc_val = 0
            loss_val = 0
            auc_val = 0
            nloop = n_val // window
            for i in range(nloop):
                X_val, Y_val = loader.fetch_batch(part = "val", batch_size = opt.batchSize)
                if opt.cuda:
                    X_val = X_val.cuda()
                    Y_val = Y_val.cuda()
                
                B,P,C,W,H=X_val.size()
                
                # if we apply the FCN
                if opt.apply_fcn:
                    with torch.no_grad():
                        X_val = resNet(X_val.view(B*P,C,W,H))
                    _,C,W,H = X_val.size()
                    X_val = X_val.view(B,P,C,W,H)

                # CoAttention Module 
                if opt.use_coAttn:
                    with torch.no_grad():
                        X_val = coAtten(X_val)   
                

                with torch.no_grad():
                    pred_val = discriminator(X_val)
                
                pred_val = torch.reshape(pred_val, (-1,))
                Y_val = torch.reshape(Y_val, (-1,))
                
                acc = one_shot_eval(pred_val.cpu().detach().numpy(), Y_val.cpu().detach().numpy())
                auc = sklearn.metrics.roc_auc_score(Y_val.cpu().detach().numpy(), pred_val.cpu().detach().numpy())
                
                auc_val = auc_val + auc
                acc_val = acc_val + (acc/window)
                loss_val = loss_val + bce(pred_val, Y_val.float())
            
            acc_val = (acc_val / nloop)*100
            loss_val = loss_val / nloop
            validation_auc = auc_val / nloop


            training_loss = loss.item()
            validation_loss = loss_val.item()
            
            training_iteration_list = training_iteration_list + [training_iteration]
            validation_loss_list = validation_loss_list + [validation_loss]
            training_loss_list = training_loss_list + [training_loss]
                            
            training_acc = get_pct_accuracy(pred, Y)
            
            training_acc_list = training_acc_list + [training_acc]
            validation_acc_list = validation_acc_list + [acc_val]
            plot_loss(opt, training_loss_list, training_acc_list, validation_loss_list, validation_acc_list, training_iteration_list, iteration)
            print("kfold number {} Iteration: {} \t Train: Acc={}%, Loss={} \t\t Validation: Acc={}%, Loss={}, ROC AUC={}".format(
                iteration, training_iteration, training_acc, training_loss, acc_val, validation_loss, validation_auc))

            if best_validation_loss is None:
                best_validation_loss = validation_loss
                
            if best_validation_auc is None:
                best_validation_auc = validation_auc
            
            if best_validation_acc is None:
                best_validation_acc = acc_val

            if not os.path.exists(save_model_path):
                os.makedirs(save_model_path)

            if (best_validation_auc*saving_threshold) < validation_auc:
                print("Significantly improved validation ROC AUC from {} --> {}. Saving...".format(
                    best_validation_auc, validation_auc
                ))
                best_validation_auc = validation_auc
                
            if best_validation_loss > (saving_threshold * validation_loss):
                print("Significantly improved validation loss from {} --> {}. Saving...".format(
                    best_validation_loss, validation_loss
                ))
                best_validation_loss = validation_loss
                
            if acc_val > (saving_threshold * best_validation_acc):
                print("Significantly improved validation accuracy from {} --> {}. Saving...".format(
                    best_validation_acc, acc_val
                ))
                torch.save(discriminator.state_dict(),save_model_path + '/{}_{}_best_accuracy_n{}.pth'.format(opt.dataset, opt.name, iteration))
                if opt.apply_fcn :
                    torch.save(resNet.state_dict(),save_model_path + '/{}_{}_fcn_best_accuracy_n{}.pth'.format(opt.dataset, opt.name, iteration))
                if opt.use_coAttn :
                    torch.save(coAtten.state_dict(),save_model_path + '/{}_{}_coatten_best_accuracy_n{}.pth'.format(opt.dataset, opt.name, iteration))
                best_validation_acc = acc_val

    
    print('****** TRAINING COMPLETED ******')
    print('Kfold number:',iteration)
    
    return best_validation_loss, best_validation_acc, best_validation_auc

def train_coAttn_models(opt, iteration) -> None:
    
    if opt.cuda:
        batcher.use_cuda = True
        models_binary.use_cuda = True  
    
    SEED = 777 
    seed_torch(SEED)

    if opt.name is None:
        # if no name is given, we generate a name from the parameters.
        # only those parameters are taken, which if changed break torch.load compatibility.
        opt.name = "{}_{}_{}_{}".format(opt.numGlimpses, opt.glimpseSize, opt.numStates)

    print("Will start training {} with parameters:\n{}\n\n".format(opt.name, opt))

    
    #writers to write the results obtained for each split
    f_val, writer_val = save_results_train(opt)
    
    save_model_path = opt.save_model_path + opt.model + "_trained_models/" + opt.dataset + "/"
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
        
    best_validation_loss, best_validation_acc, best_validation_auc = train(opt, save_model_path, iteration)
    
    #save results on the output cvs file
    val_res = [iteration, best_validation_loss, best_validation_acc, best_validation_auc]
    writer_val.writerow(val_res)

    if opt.save_results:
        f_val.close()

