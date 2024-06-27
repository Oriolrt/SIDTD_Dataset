# Local import
from .batcher_kfold_binary import Batcher
from .models_binary import *
from .utils import *
import SIDTD.models.arc_pytorch.models_binary as models_binary

# Import package
from sklearn.metrics import roc_auc_score, accuracy_score
import os
import pandas as pd
import numpy as np
import torch
import csv


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

def test_one_batch(opt, discriminator, resNet, coAtten, loader, labels, images, paths, prob, prediction, y_true):

    """
    Helper function to perform inference on one batch of images.

    Parameters
    ----------
    opt : Arguments
        opt.cuda : Parameters to decide to load data on GPU or CPU device

    discriminator, resNet, coAtten : pytorch models
    loader: batch generator
    labels
    images: image paths of the current batch
    paths: list of accumulated image paths. It will be used to generate results csv file image by image.
    prob: list of accumulated probability. It will be used to generate results csv file image by image and compute metrics over the test set.
    prediction: list of accumulated label prediction. It will be used to generate results csv file image by image and compute metrics over the test set.
    y_true: list of accumulated label groundtruth. It will be used to generate results csv file image by image and compute metrics over the test set.
    

    Returns
    -------
    paths_list: list of accumulated image paths. It will be used to generate results csv file image by image.
    prob_list: list of accumulated probability. It will be used to generate results csv file image by image and compute metrics over the test set.
    prediction_list: list of accumulated label prediction. It will be used to generate results csv file image by image and compute metrics over the test set.
    y_true_list: list of accumulated label groundtruth. It will be used to generate results csv file image by image and compute metrics over the test set.

    """

    X_test, Y_test = loader.fetch_batch(part = "test", labels = labels, image_paths = images)
    if opt.device=='cuda':
        X_test = X_test.cuda()
        Y_test = Y_test.cuda()
    
    if len(X_test.size())<5:
        X_test = X_test.unsqueeze(2)
        
    if X_test.size()[2] == 1:
        # since the omiglot data is grayscale we need to transform it to 3 channels in order to fit through resnet
        X_test = X_test.repeat(1,1,3,1,1)
    B,P,C,W,H = X_test.size()

    with torch.no_grad():
        X_test = resNet(X_test.view(B*P,C,W,H))
    _, C, W, H = X_test.size()
    X_test = X_test.view(B, P, C, W, H)
        
    with torch.no_grad():
        X_test = coAtten(X_test)
    
    with torch.no_grad():
        pred_test = discriminator(X_test)
    
    pred_test = torch.reshape(pred_test, (-1,))
    Y_test = torch.reshape(Y_test, (-1,))
    
    # Save image paths, probability, prediction and groundtruth label into lists
    paths_list = paths + list(images)
    prob_list = prob + list(pred_test.to('cpu').numpy())
    prediction_list = prediction + list(pred_test.to('cpu').numpy().round())
    y_true_list = y_true + list(Y_test.to('cpu').numpy())

    return paths_list, prob_list, prediction_list, y_true_list

def test(opt, save_model_path, iteration):

    #Define the ResNet50 NN
    print('Use ResNet50')
    resNet = CustomResNet50()
        
    #Define the CoAttn
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
        resNet.cuda()
        coAtten.cuda()


    # Load dataset paths that will be used by the batch generator
    # You can use static path csv to replicate results or custom/random partitionning
    # You can use different type of partitionning: train validation test split or kfold cross-validation 
    # You can use different type of data: templates, clips or cropped clips.
    if opt.static == 'no':
        if opt.type_split == 'kfold':
            if opt.inf_domain_change == 'yes':
                path_test = os.getcwd() + "/split_kfold/{}/test_split_{}_it_0.csv".format(opt.dataset, opt.dataset)
            else:
                path_test = os.getcwd() + "/split_kfold/{}/test_split_{}_it_{}.csv".format(opt.dataset, opt.dataset, iteration)
        elif opt.type_split =='cross':
            path_test = os.getcwd() + "/split_normal/{}/test_split_{}.csv".format(opt.dataset, opt.dataset)
    else:
        if opt.type_split =='kfold':
            if opt.type_data == 'templates':
                path_test = os.getcwd() + "/static/split_kfold/test_split_SIDTD_it_{}.csv".format(iteration)
            elif opt.type_data == 'clips':
                path_test = os.getcwd() + "/static/split_kfold_unbalanced/test_split_clip_background_SIDTD_it_{}.csv".format(iteration)
            elif opt.type_data == 'clips_cropped':
                path_test = os.getcwd() + "/static/split_kfold_cropped_unbalanced/test_split_clip_cropped_SIDTD_it_{}.csv".format(iteration)
        elif opt.type_split =='cross':
            if opt.type_data == 'templates':
                path_test = os.getcwd() + "/static/split_normal/test_split_SIDTD.csv"
            elif opt.type_data == 'clips':
                path_test = os.getcwd() + "/static/cross_val_unbalanced/test_split_SIDTD.csv"
            elif opt.type_data == 'clips_cropped':
                path_test = os.getcwd() + "/static/cross_val_cropped_unbalanced/test_split_clip_cropped_SIDTD.csv"


    # preload the dataset in python dictionnary to make the training faster.
    paths_splits = {'test':{'reals':{}, 'fakes':{}}}
    d_set = 'test'
    df = pd.read_csv(path_test)   # load validation set csv file with image paths and labels
    path_images = list(df.image_path.values)
    for key in ['reals','fakes']:
        imgs_path = list(df[df['label_name']==key].image_path.values)   # save image path from the same label
        paths_splits[d_set][key]['path'] = list(imgs_path)   # save image PATH in dictionnary along the corresponding label and data set
    loader = Batcher(opt = opt, paths_splits = paths_splits, path_img = path_images)   # load batch generator
    
    window = opt.batchSize   # define batch size for inference

    # Load trained models.
    # Choose model depending on if you want to choose a custom trained model or perform inference with our models
    if opt.pretrained == 'no':
        if opt.inf_domain_change == 'yes':
            discriminator.load_state_dict(torch.load(save_model_path + '/{}_{}_best_accuracy_n{}.pth'.format(opt.dataset_source, opt.name, iteration)))
            resNet.load_state_dict(torch.load(save_model_path + '/{}_{}_fcn_best_accuracy_n{}.pth'.format(opt.dataset_source, opt.name, iteration)))
            coAtten.load_state_dict(torch.load(save_model_path + '/{}_{}_coatten_best_accuracy_n{}.pth'.format(opt.dataset_source, opt.name, iteration)))
        else:
            discriminator.load_state_dict(torch.load(save_model_path + '/{}_{}_best_accuracy_n{}.pth'.format(opt.dataset, opt.name, iteration)))
            resNet.load_state_dict(torch.load(save_model_path + '/{}_{}_fcn_best_accuracy_n{}.pth'.format(opt.dataset, opt.name, iteration)))
            coAtten.load_state_dict(torch.load(save_model_path + '/{}_{}_coatten_best_accuracy_n{}.pth'.format(opt.dataset, opt.name, iteration)))
        
        discriminator.eval()
        resNet.eval()
        coAtten.eval()
    
    elif opt.pretrained == 'yes':
        if opt.type_data in ['clips', 'clips_cropped']:
            discriminator.load_state_dict(torch.load(save_model_path + '/coatten_fcn_model_best_accuracy_n{}.pth'.format(iteration)))
            resNet.load_state_dict(torch.load(save_model_path + '/coatten_fcn_model_fcn_best_accuracy_n{}.pth'.format(iteration)))
            coAtten.load_state_dict(torch.load(save_model_path + '/coatten_fcn_model_coatten_best_accuracy_n{}.pth'.format(iteration)))
        elif opt.type_data == 'templates':
            discriminator.load_state_dict(torch.load(save_model_path + '/MIDV2020_coatten_fcn_model_best_accuracy_n{}.pth'.format(iteration)))
            resNet.load_state_dict(torch.load(save_model_path + '/MIDV2020_coatten_fcn_model_fcn_best_accuracy_n{}.pth'.format(iteration)))
            coAtten.load_state_dict(torch.load(save_model_path + '/MIDV2020_coatten_fcn_model_coatten_best_accuracy_n{}.pth'.format(iteration)))
        discriminator.eval()
        resNet.eval()
        coAtten.eval()
    
    # Start inference...
    #path_test = opt.csv_dataset_path + "{}/test_split_{}_it_{}.csv".format(opt.dataset, opt.dataset, iteration)
    df_test = pd.read_csv(path_test)
    image_paths = df_test.image_path.values
    label_name = df_test.label_name.values
    p_preds = []
    preds = []
    reals = []
    test_paths = []
    i = 0
    # Loop over the test set, batch by batch.
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
    df.drop_duplicates(subset=['Path_image'], inplace=True, ignore_index=True)   # remove the prediction duplicate, generated in the overlap of the last step
    df.Label_image = df.Label_image.astype(int)
    df.Proba_label_image = df.Proba_label_image.astype(float)
    df.Pred_label_image = df.Pred_label_image.astype(float).astype(int)
    path_save_img_results = f'{opt.results_path}/{opt.model}/{opt.dataset}/{opt.model}_{str(iteration)}fold_stats_per_image.csv'
    df.to_csv(path_save_img_results, index=False)

    # Compute metrics: ROC AUC, accuracy, FPR and FNR
    test_auc = roc_auc_score(df['Label_image'],df['Proba_label_image'])
    acc_test = accuracy_score(df['Label_image'], df['Pred_label_image'])
    FPR, FNR = get_FPR_FNR(actual = df['Label_image'].values, pred = df['Pred_label_image'].values)
    
    print('****** TEST COMPLETED ******')
    print('Kfold number:',iteration)
    print('Final accuracy:', acc_test, 'test AUC:', test_auc)
    
    return acc_test, test_auc, FPR, FNR


def test_coAttn_models(opt, iteration=0) -> None:
    print('opt.static', opt.static)
    
    if opt.device=='cuda':
        models_binary.use_cuda = True  
    if opt.device=='cpu':
        models_binary.use_cuda = False 
    
    SEED = 777 
    seed_torch(SEED)
    
    #writers to write the results obtained for each split
    f_test, writer_test = save_results_test(opt)
    if opt.pretrained == 'yes':
        if opt.type_data == 'templates':
            save_model_path = os.getcwd() + "/pretrained_models/balanced_templates_SIDTD/coatten_fcn_model_trained_models/"
        elif opt.type_data == 'clips_cropped':
            save_model_path = os.getcwd() + "/pretrained_models/unbalanced_clip_cropped_SIDTD/coattention_trained_models/"
        elif opt.type_data == 'clips':
            save_model_path = os.getcwd() + "/pretrained_models/unbalanced_clip_background_SIDTD/coattention_trained_models/"
    else:
        if opt.inf_domain_change == 'yes':
            save_model_path = opt.save_model_path + opt.model + "_trained_models/" + opt.dataset_source + "/"
        else:
            save_model_path = opt.save_model_path + opt.model + "_trained_models/" + opt.dataset + "/"
    

    acc_test, test_auc, FPR, FNR = test(opt, save_model_path, iteration)
    
    #save results on the output cvs file
    test_res = [iteration, acc_test, test_auc, FPR, FNR]
    writer_test.writerow(test_res)

    if opt.save_results:
        f_test.close()