# local import
from .datasets import *
from ._utils import *
from SIDTD.utils.batch_generator import *

# package import
import matplotlib
matplotlib.use('Agg')
import os
import cv2
import time 
import torch 
import sklearn.metrics
import tqdm
import csv

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import torch.nn as nn 
import torch.nn.functional as F


from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset 
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from sklearn.metrics import f1_score, accuracy_score
                       
    
    
def trainer(args, LOGGER, training_iteration, model, optimizer, scheduler, criterion, n_epochs, BATCH_SIZE,
            accumulation_steps, train_loader, valid_loader, save_model_path, device):
    
    """
    Training loop

    Parameters
    ----------
    model : Pytorch model
    training_iteration: number of training partition
    optimizer
    scheduler
    criterion
    n_epochs : Number of epochs.
    BATCH_SIZE
    accumulation_steps : number of steps to update weights
    train_loader : pytorch dataloader
    valid_loader : pytorch dataloader
    save_model_path : pytorch dataloader
    device : cuda or cpu
    LOGGER : Logger file to write training info
    
    Returns
    -------
    losses : Dictionary with the training and validation losses.
    best_loss
    best_loss_epoch
    best_accuracy
    best_accuracy_epoch
    """

    with timer(LOGGER, 'Train model'):
        
        model.to(device)
        best_accuracy = 0.
        best_accuracy_epoch = 0
        best_loss = np.inf
        best_loss_epoch = 0
        best_roc_auc = 0
        losses = {"train": [], "val": []}
        
        training_iteration_list = []
        validation_loss_list = []
        training_loss_list = []
        validation_acc_list = []
        for epoch in range(n_epochs):
        
            start_time = time.time()
            
            model.train()
            avg_loss = 0.

            optimizer.zero_grad()
            
            # Model training
            for i, (images, labels) in tqdm.tqdm(enumerate(train_loader)):
            
                images = images.to(device)
                labels = labels.to(device)
                
                y_preds = model(images)
                loss = criterion(y_preds, labels)
                
                 # Scale the loss to the mean of the accumulated batch size
                loss = loss / accumulation_steps
                loss.backward()
                if (i - 1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                    avg_loss += loss.item() / len(train_loader)
                
            #Evaluation
            model.eval()
            avg_val_loss = 0.
            preds = []
            p_preds = []
            reals = np.zeros((len(valid_loader.dataset)))
            
            for i, (images, labels) in enumerate(valid_loader):
             
                images = images.to(device)
                labels = labels.to(device)
                reals[i * BATCH_SIZE: (i+1) * BATCH_SIZE] = labels.to('cpu').numpy()
             
                with torch.no_grad():
                    y_preds = model(images)
               
                preds[i * BATCH_SIZE: (i+1) * BATCH_SIZE] = y_preds.argmax(1).to('cpu').numpy()    # Label predicted by the model 
                p_pred = F.softmax(y_preds, dim=1)   # Probability predicted for each label
               
                p_preds.extend(p_pred[:,1].to('cpu').numpy())   # Probability predicted for fake label
                    
                loss = criterion(y_preds, labels)
                avg_val_loss += loss.item() / len(valid_loader)

            scheduler.step(avg_val_loss)   # initiate scheduler depending on the loss
            
            # Compute metrics
            score = f1_score(reals, preds, average='macro')
            accuracy = accuracy_score(reals, preds)           
            try:
                roc_auc_score = sklearn.metrics.roc_auc_score(reals, p_preds)
            except:
                roc_auc_score = -1
            
            elapsed = time.time() - start_time
        
            LOGGER.debug(f' Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f} avg_val_loss: {avg_val_loss:.4f} F1: {score:.6f} Accuracy: {accuracy:.6f} Roc AUC: {roc_auc_score:.6f} time: {elapsed:.0f}s')

            # Add model performance in list and save image plot of loss and accuracy in function of epoch number
            training_iteration_list = training_iteration_list + [epoch]
            validation_loss_list = validation_loss_list + [avg_val_loss]
            training_loss_list = training_loss_list + [avg_loss]
            validation_acc_list = validation_acc_list + [accuracy]
            plot_loss(args, training_loss_list, validation_loss_list, validation_acc_list, training_iteration_list, training_iteration)
        
            # Save model if higher than previously saved best accuracy
            if accuracy>best_accuracy:
                best_accuracy = accuracy
                best_accuracy_epoch = epoch
                LOGGER.debug(f' Epoch {epoch+1} - Save Best Accuracy: {best_accuracy:.6f} Model')
                torch.save(model.state_dict(), save_model_path + '/{}_{}_best_accuracy_n{}.pth'.format(args.dataset, args.name, training_iteration))
            
            # Save best roc auc in memory
            if roc_auc_score>=best_roc_auc:
                best_roc_auc = roc_auc_score
                best_roc_auc_epoch = epoch
                LOGGER.debug(f' Epoch {epoch+1} - Save Best Roc AUC: {best_roc_auc:.6f} Model')
                
            # Save best loss in memory
            if avg_val_loss<best_loss:
                best_loss = avg_val_loss
                best_loss_epoch = epoch
                LOGGER.debug(f' Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            losses['train'].append(avg_loss)
            losses['val'].append(avg_val_loss)
                
    return best_loss, best_loss_epoch, best_accuracy, best_accuracy_epoch, best_roc_auc, best_roc_auc_epoch
        

def train_baseline_models(args, LOGGER, iteration = 0):
    
    """ Here we set the parameters and features for the training."""

    if args.device=='cuda':
        # Set all data on GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(torch.cuda.is_available())
        print(device)
    else:
        device = 'cpu'
    print(device)     
    SEED = 777 
    seed_torch(SEED)

    # Set model path
    save_model_path = args.save_model_path + args.model + "_trained_models/" + args.dataset + "/"
    if not os.path.exists(save_model_path):
        os.makedirs(save_model_path)
    print("Models will be saved at: ", save_model_path)

    if not os.path.exists(args.results_path + '{}/{}/'.format(args.model, args.dataset)):
        os.makedirs(args.results_path + '{}/{}/'.format(args.model, args.dataset))
    
    # Load csv results file if it exist otherwise, create the results file
    if args.save_results:
        print("Results file: ", args.results_path + '{}/{}/{}_val_results.csv'.format(args.model, args.dataset, args.name))
        if os.path.isfile(args.results_path + '{}/{}/{}_val_results.csv'.format(args.model, args.dataset, args.name)):
            # Load csv results file
            f_val = open(args.results_path + '{}/{}/{}_val_results.csv'.format(args.model, args.dataset, args.name), 'a')
            writer_val = csv.writer(f_val)
        else:
            f_val = open(args.results_path + '{}/{}/{}_val_results.csv'.format(args.model, args.dataset, args.name), 'w')
            # create the csv writer
            writer_val = csv.writer(f_val)
            header_val = ['iteration', 'best_loss', 'best_loss_epoch', 'best_accuracy', 'best_accuracy_epoch', 'best_AUC', 'best_AUC_epoch']
            writer_val.writerow(header_val)
        
    print("DATASET: ", args.dataset)


    # Parameters for model training. They can be chosen with the flags
    BATCH_SIZE = args.batch_size
    ACCUMULATION_STEPS = args.accumulation_steps
    EPOCHS = args.epochs
    WORKERS = args.workers
    lr = args.learning_rate
    
    if args.model in ['vit_large_patch16_224', 'vit_small_patch16_224']:
        WIDTH, HEIGHT = 224, 224
    else: 
        WIDTH, HEIGHT = 299, 299
        
    dataset_crop = False
    N_CLASSES = args.nclasses
    
    #load model weights
    model = setup_model(args, N_CLASSES)
    mean, std = get_mean_std(args, model)

    LOGGER.debug("New iteration {}".format(iteration))
    LOGGER.debug("------------------------------------------------")
    
    # Load dataset paths that will be used by the batch generator
    # You can use static path csv to replicate results or choose your own random partitionning
    # You can use different type of partitionning: train validation split or kfold cross-validation 
    # You can use different type of data: templates, clips or cropped clips.
    if args.static == 'no':
        if args.type_split =='kfold':

            if os.path.exists(os.getcwd() + "/split_kfold/{}/train_split_{}_it_{}.csv".format(args.dataset, args.dataset, iteration)) and \
                os.path.exists(os.getcwd() + "/split_kfold/{}/val_split_{}_it_{}.csv".format(args.dataset, args.dataset, iteration)):
                print("Loading existing partition: ", "split_{}_it_{}".format(args.dataset, iteration))
                train_metadata_split = pd.read_csv(os.getcwd() + "/split_kfold/{}/train_split_{}_it_{}.csv".format(args.dataset, args.dataset, iteration))
                val_metadata_split = pd.read_csv(os.getcwd() + "/split_kfold/{}/val_split_{}_it_{}.csv".format(args.dataset, args.dataset, iteration))
                
            else:
                print('ERROR : WRONG PATH')
        
        elif args.type_split =='cross':

            if os.path.exists(os.getcwd() + "/split_normal/{}/train_split_{}.csv".format(args.dataset, args.dataset)) and \
                os.path.exists(os.getcwd() + "/split_normal/{}/val_split_{}.csv".format(args.dataset, args.dataset)):
                train_metadata_split = pd.read_csv(os.getcwd() + "/split_normal/{}/train_split_{}.csv".format(args.dataset, args.dataset))
                val_metadata_split = pd.read_csv(os.getcwd() + "/split_normal/{}/val_split_{}.csv".format(args.dataset, args.dataset))
                
            else:
                print('ERROR : WRONG PATH')

    else:
        if args.type_split =='kfold':
            if args.type_data =='templates':
                if (os.path.exists(os.getcwd() + "/static/split_kfold/train_split_SIDTD_it_{}.csv".format(iteration))) and (os.path.exists(os.getcwd() + "/static/split_kfold/val_split_SIDTD_it_{}.csv".format(iteration))):
                    print("Loading static train and val partition")
                    train_metadata_split = pd.read_csv(os.getcwd() + "/static/split_kfold/train_split_SIDTD_it_{}.csv".format(iteration)) 
                    val_metadata_split = pd.read_csv(os.getcwd() + "/static/split_kfold/val_split_SIDTD_it_{}.csv".format(iteration)) 
                else:
                    print('ERROR : WRONG PATH')
                
            elif args.type_data == 'clips':
                if (os.path.exists(os.getcwd() + "/static/split_kfold_unbalanced/train_split_clip_background_SIDTD_it_{}.csv".format(iteration))) and (os.path.exists(os.getcwd() + "/static/split_kfold_unbalanced/val_split_clip_background_SIDTD_it_{}.csv".format(iteration))):
                    print("Loading static train and val partition")
                    train_metadata_split = pd.read_csv(os.getcwd() + "/static/split_kfold_unbalanced/train_split_clip_background_SIDTD_it_{}.csv".format(iteration))
                    val_metadata_split = pd.read_csv(os.getcwd() + "/static/split_kfold_unbalanced/val_split_clip_background_SIDTD_it_{}.csv".format(iteration))
                else:
                    print('ERROR : WRONG PATH')

            else:
                if (os.path.exists(os.getcwd() + "/static/split_kfold_cropped_unbalanced/train_split_clip_cropped_SIDTD_it_{}.csv".format(iteration))) and (os.path.exists(os.getcwd() + "/static/split_kfold_cropped_unbalanced/val_split_clip_cropped_SIDTD_it_{}.csv".format(iteration))):
                    print("Loading static train and val partition")
                    train_metadata_split = pd.read_csv(os.getcwd() + "/static/split_kfold_cropped_unbalanced/train_split_clip_cropped_SIDTD_it_{}.csv".format(iteration))
                    val_metadata_split = pd.read_csv(os.getcwd() + "/static/split_kfold_cropped_unbalanced/val_split_clip_cropped_SIDTD_it_{}.csv".format(iteration))
                else:
                    print('ERROR : WRONG PATH')

        else:
            if args.type_data =='templates':
                if (os.path.exists(os.getcwd() + "/static/split_normal/train_split_SIDTD.csv")) and (os.path.exists(os.getcwd() + "/static/split_normal/val_split_SIDTD.csv")):
                    print("Loading static train and val partition")
                    train_metadata_split = pd.read_csv(os.getcwd() + "/static/split_normal/train_split_SIDTD.csv")
                    val_metadata_split = pd.read_csv(os.getcwd() + "/static/split_normal/val_split_SIDTD.csv")
                else:
                        print('ERROR : WRONG PATH')

            elif args.type_data == 'clips':
                if (os.path.exists(os.getcwd() + "/static/cross_val_unbalanced/train_split_SIDTD.csv")) and (os.path.exists(os.getcwd() + "/static/cross_val_unbalanced/val_split_SIDTD.csv")):
                    print("Loading static train and val partition")
                    train_metadata_split = pd.read_csv(os.getcwd() + "/static/cross_val_unbalanced/train_split_SIDTD.csv")
                    val_metadata_split = pd.read_csv(os.getcwd() + "/static/cross_val_unbalanced/val_split_SIDTD.csv")
                else:
                    print('ERROR : WRONG PATH')

            elif args.type_data == 'clips_cropped':
                if (os.path.exists(os.getcwd() + "/static/cross_val_cropped_unbalanced/train_split_clip_cropped_SIDTD.csv")) and (os.path.exists(os.getcwd() + "/static/cross_val_cropped_unbalanced/val_split_clip_cropped_SIDTD.csv")):
                    print("Loading static train and val partition")
                    train_metadata_split = pd.read_csv(os.getcwd() + "/static/cross_val_cropped_unbalanced/train_split_clip_cropped_SIDTD.csv")
                    val_metadata_split = pd.read_csv(os.getcwd() + "/static/cross_val_cropped_unbalanced/val_split_clip_cropped_SIDTD.csv")
                else:
                    print('ERROR : WRONG PATH')


    train_paths = train_metadata_split['image_path'].values.tolist()  # load train image path

    # Load train label. If you choose to perform forgery augmentation, the images path and labels must be loaded in a dictionnary.
    if not args.faker_data_augmentation:
        train_ids = train_metadata_split['label'].values.tolist()
    else:
        dataset_dict = {'train':{}}
        for label in [0,1]:
            dataset_dict['train'][label] = train_metadata_split[train_metadata_split['label'] == label]['image_path'].values.tolist()
    
    # Load image path and label for validation set
    val_paths = val_metadata_split['image_path'].values.tolist()
    val_ids = val_metadata_split['label'].values.tolist()
    
    # Load Batch Generator function
    if not args.faker_data_augmentation:
        train_dataset = TrainDataset(train_paths, train_ids, transform=get_transforms(WIDTH, HEIGHT, mean, std, data='train'), dataset_crop = dataset_crop)
    else:
        train_dataset = TrainDatasets_augmented(args, dataset_dict['train'], train_paths, transform=get_transforms(WIDTH, HEIGHT, mean, std, data='train'))

    val_dataset = TrainDataset(val_paths, val_ids, transform=get_transforms(WIDTH, HEIGHT, mean, std, data='train'), dataset_crop = dataset_crop)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS)
    
    # Set training parameters function for: Optimizer, scheduler and criterion
    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.9, patience=1, verbose=True, eps=1e-6)
    criterion = nn.CrossEntropyLoss()
    
    # Train model
    best_loss, best_loss_epoch, best_score, best_score_epoch, best_roc_auc, best_roc_auc_epoch = trainer(args, LOGGER, iteration,
                        model, optimizer,
                        scheduler, criterion,
                        EPOCHS, BATCH_SIZE, 
                        ACCUMULATION_STEPS,
                        train_loader,
                        val_loader,
                        save_model_path,
                        device)
    
    
    # write and save training results in csv
    if args.save_results:
        val_res = [iteration, best_loss, best_loss_epoch, best_score, best_score_epoch, best_roc_auc, best_roc_auc_epoch]
        
        writer_val.writerow(val_res)
        

    if args.save_results:
        f_val.close()
