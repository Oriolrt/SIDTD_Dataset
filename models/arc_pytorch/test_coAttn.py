import sys
import os

hard_path = ''
for x in os.getcwd().split('/')[1:-1]: hard_path = hard_path + '/' + x
complete_path = hard_path + '/models/arc_pytorch/'
sys.path.insert(1, complete_path)

import random
import pandas as pd
import numpy as np
import torch
from torch.autograd import Variable
import batcher_kfold_binary as batcher
from batcher_kfold_binary import Batcher
from utils import *
from sklearn.metrics import roc_auc_score, accuracy_score

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

def test_one_batch(opt, discriminator, resNet, coAtten, loader, labels, images, paths, prob, prediction, y_true):

    X_test, Y_test = loader.fetch_batch(part = "test", labels = labels, image_paths = images, batch_size = opt.batchSize)
    if opt.cuda:
        X_test = X_test.cuda()
        Y_test = Y_test.cuda()
    
    if len(X_test.size())<5:
        X_test = X_test.unsqueeze(2)
        
    if opt.apply_fcn:
        if X_test.size()[2] == 1:
            # since the omiglot data is grayscale we need to transform it to 3 channels in order to fit through resnet
            X_test = X_test.repeat(1,1,3,1,1)
        B,P,C,W,H = X_test.size()

        with torch.no_grad():
            X_test = resNet(X_test.view(B*P,C,W,H))
        _, C, W, H = X_test.size()
        X_test = X_test.view(B, P, C, W, H)
        
    if opt.use_coAttn:
        with torch.no_grad():
            X_test = coAtten(X_test)
    
    with torch.no_grad():
        pred_test = discriminator(X_test)
    
    pred_test = torch.reshape(pred_test, (-1,))
    Y_test = torch.reshape(Y_test, (-1,))
    
    paths_list = paths + list(images)
    prob_list = prob + list(pred_test.to('cpu').numpy())
    prediction_list = prediction + list(pred_test.to('cpu').numpy().round())
    y_true_list = y_true + list(Y_test.to('cpu').numpy())

    return paths_list, prob_list, prediction_list, y_true_list


def test(opt, save_model_path, iteration):

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
        discriminator.cuda()
        if opt.apply_fcn:
            resNet.cuda()
        if opt.use_coAttn:
            coAtten.cuda()


    # set up the optimizer.
    bce = torch.nn.BCELoss()
    if opt.cuda:
        bce = bce.cuda()


    # load the dataset in memory.
    paths_splits = {'test':{}}
    d_set = 'test'
    for key in ['reals','fakes']:
        path = opt.npy_dataset_path +  opt.dataset + '/' + d_set + '_split_' + key  + '_it_' + str(iteration) + '.npy'
        data = np.load(path)
        paths_splits[d_set][key] = list(data)
    loader = Batcher(paths_splits= paths_splits, batch_size=opt.batchSize, image_size=opt.imageSize)
    window = opt.batchSize

    # Test model
    discriminator.load_state_dict(torch.load(save_model_path + '/{}_{}_best_accuracy_n{}.pth'.format(opt.dataset, opt.name, iteration)))
    discriminator.eval()
    if opt.apply_fcn:
        resNet.load_state_dict(torch.load(save_model_path + '/{}_{}_fcn_best_accuracy_n{}.pth'.format(opt.dataset, opt.name, iteration)))
        resNet.eval()
    if opt.use_coAttn:
        coAtten.load_state_dict(torch.load(save_model_path + '/{}_{}_coatten_best_accuracy_n{}.pth'.format(opt.dataset, opt.name, iteration)))
        coAtten.eval()
    
    path_test = opt.csv_dataset_path + "{}/test_split_{}_it_{}.csv".format(opt.dataset, opt.dataset, iteration)
    df_test = pd.read_csv(path_test)
    image_paths = df_test.image_path.values
    label_name = df_test.label_name.values
    p_preds = []
    preds = []
    reals = []
    test_paths = []
    i = 0
    while window*(i+1) < len(df_test):
        labels = label_name[window*i]
        images = image_paths[window*(i+1)]
        test_paths, p_preds, preds, reals = test_one_batch(opt, discriminator, resNet, coAtten, loader, labels, images, test_paths, p_preds, preds, reals)
        i = i + 1  
        
    labels = label_name[-window:]
    images = image_paths[-window:]
    test_paths, p_preds, preds, reals = test_one_batch(opt, discriminator, resNet, coAtten, loader, labels, images, test_paths, p_preds, preds, reals)    

    my_array = np.asarray([ test_paths, reals, preds, p_preds]).T
    df = pd.DataFrame(my_array, columns = ['Path_image','Label_image','Pred_label_image','Proba_label_image'])
    df.drop_duplicates(subset=['Path_image'], inplace=True)

    test_auc = roc_auc_score(df['Label_image'].astype(int),df['Proba_label_image'])
    acc_test = accuracy_score(df['Label_image'],df['Pred_label_image'])
    
    print('****** TEST COMPLETED ******')
    print('Kfold number:',iteration)
    print('Final accuracy:', acc_test, 'test AUC:', test_auc)
    
    return acc_test, test_auc


def test_coAttn_models(opt, iteration) -> None:
    
    if opt.cuda:
        batcher.use_cuda = True
        models_binary.use_cuda = True  
    
    SEED = 777 
    seed_torch(SEED)
    
    #writers to write the results obtained for each split
    f_test, writer_test = save_results_test(opt)
    save_model_path = opt.save_model_path + opt.model + "_trained_models/" + opt.dataset + "/"
    
        
    acc_test, test_auc = test(opt, save_model_path, iteration)
    
    #save results on the output cvs file
    test_res = [iteration, acc_test, test_auc]
    writer_test.writerow(test_res)

    if opt.save_results:
        f_test.close()

