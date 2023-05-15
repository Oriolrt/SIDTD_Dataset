from .datasets import *

import matplotlib
matplotlib.use('Agg')
import os
import cv2
import time 
import timm
import torch 
import random
import sklearn.metrics
import tqdm
import csv

import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
import torch.nn as nn 
import torch.nn.functional as F
import torchvision.models as models


from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset 
from contextlib import contextmanager
from albumentations import Compose, Normalize, Resize 
from albumentations.pytorch import ToTensorV2 
from albumentations import HorizontalFlip, VerticalFlip, RandomBrightnessContrast, RandomResizedCrop
from torch.optim.lr_scheduler import ReduceLROnPlateau 
from sklearn.metrics import f1_score, accuracy_score
from efficientnet_pytorch import EfficientNet




@contextmanager 
def timer(LOGGER, name):
    """ Returns time duration """
    t0 = time.time()
    LOGGER.info(f'[{name}] start')
    yield
    LOGGER.info(f'[{name}] done in {time.time() - t0:.0f} s.')

def seed_torch(seed=777):
    """ Set torch model seed to fix randomness """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 


class TrainDataset(Dataset):
    """ Classic Batch Data Generator. 
    Input:  Image path in csv
    Output: image in array format + label
    You can choose to perform data augmentation by setting a function with the transform argument.
    """
    def __init__(self, paths, ids, transform=None, dataset_crop=False):
        self.paths = paths
        self.ids = ids
        self.transform = transform
        self.dataset_crop = dataset_crop
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        file_path = self.paths[idx]
        label = self.ids[idx]
        image = cv2.imread(file_path)   # read img path
    
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # read image as RGB
        except:
            print(file_path)
        if self.dataset_crop:
            sx, sy, _ = image.shape
            sx2 = int(sx/2)
            sy2 = int(sy/2)
            image = image[sx2-250:sx2+250, sy2-250:sy2+250, :]   # Crop image
            
        if self.transform:
            augmented = self.transform(image=image)   # perform data augmentation
            image = augmented['image']
    
        return image, label

    
def get_transforms(WIDTH, HEIGHT, mean, std, data):
    """ Function that returns augmented image based on albumentations functions. Images are also reized and normalized. """
    assert data in ('train', 'valid')
    
    if data == 'train':
        # Data Augmentation performed on train set
        return Compose([
        RandomResizedCrop(WIDTH, HEIGHT, scale=(0.8, 1.0)),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomBrightnessContrast(p=0.2),
        Normalize(mean=mean, std=std),
        ToTensorV2(),
        ])
    
    elif data == 'valid':
        # Data Augmentation performed on validation and test set
        return Compose([
        Resize(WIDTH, HEIGHT),
        Normalize(mean=mean, std=std),
        ToTensorV2(),
        ]) 


def setup_model(args, N_CLASSES, pretrained=True):
    """ Set type of model chosen: Efficientnet-b3, Resnet50 or ViT."""
    print("MODEL SETUP: ", args.model)
    if args.model == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(model.fc.in_features, N_CLASSES)    # Change last layer to fit the number of class needed
        
    elif args.model in ['vit_large_patch16_224', 'vit_small_patch16_224']:
        model = timm.create_model(args.model, pretrained=pretrained)
        net_cfg = model.default_cfg
        last_layer = net_cfg['classifier']
        num_ftrs = getattr(model, last_layer).in_features
        setattr(model, last_layer, nn.Linear(num_ftrs, N_CLASSES))    # Change last layer to fit the number of class needed
        
    elif args.model == 'efficientnet-b3':
        model = EfficientNet.from_pretrained(args.model)
        model._fc = nn.Linear(model._fc.in_features, N_CLASSES)      # Change last layer to fit the number of class needed
        
    return model 


def get_mean_std(args, model):
    if args.model in ['resnet50', 'efficientnet-b3']:
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        
    elif args.model in ['vit_large_patch16_224', 'vit_small_patch16_224']:
        mean = list(model.default_cfg['mean'])
        std = list(model.default_cfg['std'])    
    return mean, std
                       


def plot_loss(args, training_loss_list, validation_loss_list, validation_acc_list, training_iteration_list, training_iteration):

    """ Returns plot image of loss and accuracy for validation and train set in function of number of epochs """

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
    
    
def trainer(args, LOGGER, training_iteration, model, optimizer, scheduler, criterion, n_epochs, BATCH_SIZE,
            accumulation_steps, train_loader, valid_loader, save_model_path, device):
    
    """ Trains the model chosen depending on the features chosen previously (epochs, batch size, optimizer etc...) """

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

            # Save plot of accuracy and loss
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
            
            # Save best roc auc
            if roc_auc_score>=best_roc_auc:
                best_roc_auc = roc_auc_score
                best_roc_auc_epoch = epoch
                LOGGER.debug(f' Epoch {epoch+1} - Save Best Roc AUC: {best_roc_auc:.6f} Model')
                
            # Save best loss
            if avg_val_loss<best_loss:
                best_loss = avg_val_loss
                best_loss_epoch = epoch
                LOGGER.debug(f' Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            losses['train'].append(avg_loss)
            losses['val'].append(avg_val_loss)
                
    return best_loss, best_loss_epoch, best_accuracy, best_accuracy_epoch, best_roc_auc, best_roc_auc_epoch
        

def train_baseline_models(args, LOGGER, iteration = 0):
    
    """ Here we set the features of the experiment. For instance we set the dataloaders, csv results files and model training features etc..."""

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
    
    # Load existing csv results file if it exist otherwise, create the results file
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
    # You can use static path csv to replicate results or custom/random partitionning
    # You can use different type of partitionning: train validation test split or kfold cross-validation 
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

    # Load train label. If you choose to perform forgery augmentation, the images path and labels need to be load into a dictionnary.
    if not args.faker_data_augmentation:
        train_ids = train_metadata_split['label'].values.tolist()
    else:
        dataset_dict = {'train':{}}
        for label in [0,1]:
            dataset_dict['train'][label] = train_metadata_split[train_metadata_split['label'] == label]['image_path'].values.tolist()
    
    # Load image path and label for validation set
    val_paths = val_metadata_split['image_path'].values.tolist()
    val_ids = val_metadata_split['label'].values.tolist()

    
    print("Training images: ", len(list(set(train_ids))))
    print("Validation images: ", len(list(set(val_ids))))
    
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
