from .batcher_kfold_binary  import Batcher
from .models_binary import *
from .utils import *
import SIDTD.models.arc_pytorch.models_binary as models_binary


import os
import pandas as pd
import cv2
import sklearn
import torch


def train(opt, save_model_path, iteration):

    #Define the ResNet50 NN
    print('Use ResNet50')
    resNet = CustomResNet50()
        
    #Define the CoAttn
    print('Use Co Attention Model')
    coAtten = CoAttn()
             
    # initialise the ARC (Attentive Recurrent Comparators) model
    discriminator = ArcBinaryClassifier(num_glimpses=opt.numGlimpses,
                                        glimpse_h=opt.glimpseSize,
                                        glimpse_w=opt.glimpseSize,
                                        channels = 1024,
                                        controller_out=opt.numStates)

    # If chosen, use models on GPU device
    if opt.device=='cuda':
        print('Use GPU')
        discriminator.cuda()
        resNet.cuda()
        coAtten.cuda()

    # set up the optimizer.
    bce = torch.nn.BCELoss()
    if opt.device=='cuda':
        bce = bce.cuda()

    print('Optimizer before')

    optim_params = []
    optim_params.append(list(discriminator.parameters()))
    optim_params.append(list(resNet.parameters()))
    optim_params.append(list(coAtten.parameters()))

    print('Optimizer append')
        
    flat_params = [item for sublist in optim_params for item in sublist]

    print('Optimizer flat')
    
    optimizer = torch.optim.Adam(params=flat_params, lr=opt.lr)

    print('Adam')

    # Load dataset paths that will be used by the batch generator
    # You can use static path csv to replicate results or choose your own random partitionning
    # You can use different type of partitionning: train validation split or kfold cross-validation 
    # You can use different type of data: templates, clips or cropped clips.
    if opt.static == 'no':
        if opt.type_split =='kfold':
            path_train = os.getcwd() + "/split_kfold/{}/train_split_{}_it_{}.csv".format(opt.dataset, opt.dataset, iteration)
            path_val = os.getcwd() + "/split_kfold/{}/val_split_{}_it_{}.csv".format(opt.dataset, opt.dataset, iteration)
            print('LOAD PATH CORRECTLY')
            print(os.path.exists(path_train))
            print(os.path.exists(path_val))
        elif opt.type_split =='cross':
            path_train = os.getcwd() + "/split_normal/{}/train_split_{}.csv".format(opt.dataset, opt.dataset)
            path_val = os.getcwd() + "/split_normal/{}/val_split_{}.csv".format(opt.dataset, opt.dataset)
    else:
        if opt.type_split =='kfold':
            if opt.type_data == 'templates':
                path_train = os.getcwd() + "/static/split_kfold/train_split_SIDTD_it_{}.csv".format(iteration)
                path_val = os.getcwd() + "/static/split_kfold/val_split_SIDTD_it_{}.csv".format(iteration)
            elif opt.type_data == 'clips':
                path_train = os.getcwd() + "/static/split_kfold_unbalanced/train_split_clip_background_SIDTD_it_{}.csv".format(iteration)
                path_val = os.getcwd() + "/static/split_kfold_unbalanced/val_split_clip_background_SIDTD_it_{}.csv".format(iteration)
            elif opt.type_data == 'clips_cropped':
                path_train = os.getcwd() + "/static/split_kfold_cropped_unbalanced/train_split_clip_cropped_SIDTD_it_{}.csv".format(iteration)
                path_val = os.getcwd() + "/static/split_kfold_cropped_unbalanced/val_split_clip_cropped_SIDTD_it_{}.csv".format(iteration)
        elif opt.type_split =='cross':
            if opt.type_data == 'templates':
                path_train = os.getcwd() + "/static/split_normal/train_split_SIDTD.csv"
                path_val = os.getcwd() + "/static/split_normal/val_split_SIDTD.csv"
            elif opt.type_data == 'clips':
                path_train = os.getcwd() + "/static/cross_val_unbalanced/train_split_SIDTD.csv"
                path_val = os.getcwd() + "/static/cross_val_unbalanced/val_split_SIDTD.csv"
            elif opt.type_data == 'clips_cropped':
                path_train = os.getcwd() + "/static/cross_val_cropped_unbalanced/train_split_clip_cropped_SIDTD.csv"
                path_val = os.getcwd() + "/static/cross_val_cropped_unbalanced/val_split_clip_cropped_SIDTD.csv"

    # preload the dataset in python dictionnary to make the training faster.
    paths_splits = {'train':{'reals':{}, 'fakes':{}}, 'val' :{'reals':{}, 'fakes':{}}}
    for d_set in ['train', 'val']:
        if d_set == 'val':
            df = pd.read_csv(path_val)   # load validation set csv file with image paths and labels
            n_val = len(df)
        else:
            df = pd.read_csv(path_train)   # load train set csv file with image paths and labels
            path_images = list(df.image_path.values)
        for key in ['reals','fakes']:
            print(d_set, key)
            imgs_path = list(df[df['label_name']==key].image_path.values)   # save image path from the same label
            # Loop over all image from the same label, read image and convert it to RGB image
            paths_splits[d_set][key]['path'] = list(imgs_path)   # save image PATH in dictionnary along the corresponding label and data set
    print('LOAD PATH COMPLETED')
    loader = Batcher(opt = opt, paths_splits = paths_splits, path_img = path_images)   # load batch generator
    print('LOAD BATCHER')

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
    
    print('Start Training...')
    # One epoch = training_iteration = one batch
    for training_iteration in range(0,opt.n_its):
        
        discriminator.train(mode=True)

        X, Y = loader.fetch_batch("train")
        if opt.device=='cuda':
            X = X.cuda()
            Y = Y.cuda()
        B,P,C,W,H=X.size()
        
        resNet.train()
        X = resNet(X.view(B*P,C,W,H))
        _,C,W,H = X.size()
        X =X.view(B,P,C,W,H)
            
        # CoAttention Module 
        coAtten.train()
        X = coAtten(X)
            
        pred = discriminator(X)
        loss = bce(pred, Y.float())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print('Start Evaluation after 50epochs...')
        
        # Evaluation on validation set every 50 epochs
        if training_iteration % 50 == 0:
        
            discriminator.eval()
            resNet.eval()
            coAtten.eval()
            
            # validate your model
            acc_val = 0
            loss_val = 0
            auc_val = 0
            nloop = n_val // window # approximate number of loop to do in order to validate over the whole validation set
            for i in range(nloop):
                X_val, Y_val = loader.fetch_batch(part = "val")
                if opt.device=='cuda':
                    X_val = X_val.cuda()
                    Y_val = Y_val.cuda()
                
                B,P,C,W,H=X_val.size()
                
                # if we apply the FCN
                with torch.no_grad():
                    X_val = resNet(X_val.view(B*P,C,W,H))
                _,C,W,H = X_val.size()
                X_val = X_val.view(B,P,C,W,H)

                # CoAttention Module 
                with torch.no_grad():
                    X_val = coAtten(X_val)   
                

                with torch.no_grad():
                    pred_val = discriminator(X_val)
                
                pred_val = torch.reshape(pred_val, (-1,))
                Y_val = torch.reshape(Y_val, (-1,))
                
                # Compute accuracy and ROC AUC metrics
                acc = one_shot_eval(pred_val.cpu().detach().numpy(), Y_val.cpu().detach().numpy())
                auc = sklearn.metrics.roc_auc_score(Y_val.cpu().detach().numpy(), pred_val.cpu().detach().numpy())
                
                auc_val = auc_val + auc
                acc_val = acc_val + (acc/window)
                loss_val = loss_val + bce(pred_val, Y_val.float())
            
            # store loss, accuracy and ROC AUC of validation set
            acc_val = (acc_val / nloop)*100
            loss_val = loss_val / nloop
            validation_auc = auc_val / nloop

            training_acc = get_pct_accuracy(pred, Y) # accuracy of training set

            training_loss = loss.item()
            validation_loss = loss_val.item()                      
            
            # Add model performance in list and save image plot of loss and accuracy in function of epoch number
            training_iteration_list = training_iteration_list + [training_iteration]
            validation_loss_list = validation_loss_list + [validation_loss]
            training_loss_list = training_loss_list + [training_loss]
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

            # Save best roc auc in memory
            if (best_validation_auc*saving_threshold) < validation_auc:
                print("Significantly improved validation ROC AUC from {} --> {}. Saving...".format(
                    best_validation_auc, validation_auc
                ))
                best_validation_auc = validation_auc
                
            # Save best loss in memory    
            if best_validation_loss > (saving_threshold * validation_loss):
                print("Significantly improved validation loss from {} --> {}. Saving...".format(
                    best_validation_loss, validation_loss
                ))
                best_validation_loss = validation_loss
                
            # Save model if higher than previously saved best accuracy
            if acc_val > (saving_threshold * best_validation_acc):
                print("Significantly improved validation accuracy from {} --> {}. Saving...".format(
                    best_validation_acc, acc_val
                ))
                torch.save(discriminator.state_dict(),save_model_path + '/{}_{}_best_accuracy_n{}.pth'.format(opt.dataset, opt.name, iteration))
                torch.save(resNet.state_dict(),save_model_path + '/{}_{}_fcn_best_accuracy_n{}.pth'.format(opt.dataset, opt.name, iteration))
                torch.save(coAtten.state_dict(),save_model_path + '/{}_{}_coatten_best_accuracy_n{}.pth'.format(opt.dataset, opt.name, iteration))
                best_validation_acc = acc_val

    
    print('****** TRAINING COMPLETED ******')
    print('Kfold number:',iteration)
    
    return best_validation_loss, best_validation_acc, best_validation_auc

def train_coAttn_models(opt, iteration=0) -> None:
    
    if opt.device=='cuda':
        models_binary.use_cuda = True  
    if opt.device=='cpu':
        models_binary.use_cuda = False
    
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
        
    # Start training step and returns loss, accuracy and ROC AUC
    best_validation_loss, best_validation_acc, best_validation_auc = train(opt, save_model_path, iteration)
    
    #save results on the output cvs file
    val_res = [iteration, best_validation_loss, best_validation_acc, best_validation_auc]
    writer_val.writerow(val_res)

    if opt.save_results:
        f_val.close()