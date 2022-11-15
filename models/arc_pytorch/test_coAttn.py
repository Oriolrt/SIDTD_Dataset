import sys
import os

hard_path = ''
for x in os.getcwd().split('/')[1:-1]: hard_path = hard_path + '/' + x
complete_path = hard_path + '/models/arc_pytorch/'
sys.path.insert(1, complete_path)

import random
import pandas as pd
import numpy as np
import imageio.v2 as imageio
import cv2
import torch
from torch.autograd import Variable
import batcher_kfold_binary as batcher
from batcher_kfold_binary import Batcher
from sklearn.metrics import roc_auc_score, accuracy_score

import models_binary
from models_binary import ArcBinaryClassifier, CustomResNet50, CoAttn
import csv 

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

def read_image(image_path, image_size):
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
        header_test = ['training_iteration', 'accuracy', 'roc_auc_score']
        writer_test.writerow(header_test)
    
    return f_test, writer_test

def test_one_batch(opt, discriminator, resNet, coAtten, loader, labels, images, paths, prob, prediction, y_true):

    X_test, Y_test = loader.fetch_batch(part = "test", labels = labels, image_paths = images, batch_size = opt.batchSize)
    if opt.device=='cuda':
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

    if opt.device=='cuda':
        discriminator.cuda()
        if opt.apply_fcn:
            resNet.cuda()
        if opt.use_coAttn:
            coAtten.cuda()


    # set up the optimizer.
    bce = torch.nn.BCELoss()
    if opt.device=='cuda':
        bce = bce.cuda()

    # csv path for train and validation
    if opt.type_split =='kfold':
        path_test = os.getcwd() + "/split_kfold/{}/test_split_{}_it_{}.csv".format(opt.dataset, opt.dataset, iteration)
    elif opt.type_split =='cross':
        path_test = os.getcwd() + "/split_normal/{}/test_split_{}.csv".format(opt.dataset, opt.dataset)
    else:
        path_test = os.getcwd() + "/static_cross_val/{}/test_split_{}.csv".format(opt.dataset, opt.dataset)

    # load the dataset in memory.
    paths_splits = {'test' :{}}
    d_set = 'test'
    df = pd.read_csv(path_test)
    for key in ['reals','fakes']:
        imgs_path = df[df['label_name']==key].image_path.values
        array_data = []
        for path in imgs_path:
            img = read_image(path, image_size=opt.imageSize)  # read and resize image
            array_data.append(img)
        paths_splits[d_set][key] = list(array_data)
    loader = Batcher(paths_splits = paths_splits, batch_size=opt.batchSize, image_size=opt.imageSize)
    window = opt.batchSize

    # Test model
    if opt.pretrained == 'no':
        discriminator.load_state_dict(torch.load(save_model_path + '/{}_{}_best_accuracy_n{}.pth'.format(opt.dataset, opt.name, iteration)))
        discriminator.eval()
        if opt.apply_fcn:
            resNet.load_state_dict(torch.load(save_model_path + '/{}_{}_fcn_best_accuracy_n{}.pth'.format(opt.dataset, opt.name, iteration)))
            resNet.eval()
        if opt.use_coAttn:
            coAtten.load_state_dict(torch.load(save_model_path + '/{}_{}_coatten_best_accuracy_n{}.pth'.format(opt.dataset, opt.name, iteration)))
            coAtten.eval()
    
    if opt.pretrained == 'yes':
        discriminator.load_state_dict(torch.load(save_model_path + '/MIDV2020_coatten_fcn_model_best_accuracy_n{}.pth'.format(iteration)))
        discriminator.eval()
        if opt.apply_fcn:
            resNet.load_state_dict(torch.load(save_model_path + '/MIDV2020_coatten_fcn_model_fcn_best_accuracy_n{}.pth'.format(iteration)))
            resNet.eval()
        if opt.use_coAttn:
            coAtten.load_state_dict(torch.load(save_model_path + '/MIDV2020_coatten_fcn_model_coatten_best_accuracy_n{}.pth'.format(iteration)))
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
        labels = label_name[window*i:window*(i+1)]
        images = image_paths[window*i:window*(i+1)]
        test_paths, p_preds, preds, reals = test_one_batch(opt, discriminator, resNet, coAtten, loader, labels, images, test_paths, p_preds, preds, reals)
        i = i + 1  
        
    labels = label_name[-window:]
    images = image_paths[-window:]
    test_paths, p_preds, preds, reals = test_one_batch(opt, discriminator, resNet, coAtten, loader, labels, images, test_paths, p_preds, preds, reals)    

    my_array = np.asarray([ test_paths, reals, preds, p_preds]).T
    df = pd.DataFrame(my_array, columns = ['Path_image','Label_image','Pred_label_image','Proba_label_image'])
    df.drop_duplicates(subset=['Path_image'], inplace=True, ignore_index=True)
    df.Label_image = df.Label_image.astype(int)
    df.Proba_label_image = df.Proba_label_image.astype(float)
    df.Pred_label_image = df.Pred_label_image.astype(float).astype(int)

    test_auc = roc_auc_score(df['Label_image'],df['Proba_label_image'])
    acc_test = accuracy_score(df['Label_image'], df['Pred_label_image'])
    
    print('****** TEST COMPLETED ******')
    print('Kfold number:',iteration)
    print('Final accuracy:', acc_test, 'test AUC:', test_auc)
    
    return acc_test, test_auc


def test_coAttn_models(opt, iteration=0) -> None:
    
    if opt.device=='cuda':
        models_binary.use_cuda = True  
    if opt.device=='cpu':
        models_binary.use_cuda = False 
    
    SEED = 777 
    seed_torch(SEED)
    
    #writers to write the results obtained for each split
    f_test, writer_test = save_results_test(opt)
    if opt.pretrained == 'yes':
        save_model_path = os.getcwd() + "/pretrained_models/coatten_fcn_model_trained_models/"
    if opt.pretrained == 'no':
        save_model_path = opt.save_model_path + opt.model + "_trained_models/" + opt.dataset + "/"
    
        
    acc_test, test_auc = test(opt, save_model_path, iteration)
    
    #save results on the output cvs file
    test_res = [iteration, acc_test, test_auc]
    writer_test.writerow(test_res)

    if opt.save_results:
        f_test.close()

