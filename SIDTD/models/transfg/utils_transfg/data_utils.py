import logging

import torch
import pandas as pd
import os


from torch.utils.data import DataLoader

from albumentations import Compose, Normalize, Resize 
from albumentations.pytorch import ToTensorV2 
from albumentations import HorizontalFlip, VerticalFlip, RandomBrightnessContrast, RandomResizedCrop
from SIDTD.utils.batch_generator import *


logger = logging.getLogger(__name__)

def get_transforms(WIDTH, HEIGHT, mean, std, data):

    """ Function that returns augmented image based on albumentations functions. Images are also reized and normalized. """

    assert data in ('train', 'valid')
    
    if data == 'train':
        return Compose([
        RandomResizedCrop(WIDTH, HEIGHT, scale=(0.8, 1.0)),
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomBrightnessContrast(p=0.2),
        Normalize(mean=mean, std=std),
        ToTensorV2(),
        ])
    
    elif data == 'valid':
        return Compose([
        Resize(WIDTH, HEIGHT),
        Normalize(mean=mean, std=std),
        ToTensorV2(),
        ]) 

def get_loader(args, training_iteration):

    """ Load dataset paths that will be used by the batch generator for training Trans FG model.
        You can use static path csv to replicate results or choose your own random partitionning
        You can use different type of partitionning: train validation split or kfold cross-validation 
        You can use different type of data: templates, clips or cropped clips.  """

    # Dimension to resize images to fit model dimension
    WIDTH = args.img_size
    HEIGHT = args.img_size

    # Load Data Augmentation function
    train_transforms = get_transforms(WIDTH, HEIGHT, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], data='train')
    test_transforms = get_transforms(WIDTH, HEIGHT, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], data='valid')

    # function to load csv with image paths and labels groundtruths of train and validation set in memory
    if args.static == 'no':
        if args.type_split == 'kfold':
            train_metadata_split = pd.read_csv(os.getcwd() + "/split_kfold/{}/train_split_{}_it_{}.csv".format(args.dataset, args.dataset, training_iteration))
            val_metadata_split = pd.read_csv(os.getcwd() + "/split_kfold/{}/val_split_{}_it_{}.csv".format(args.dataset, args.dataset, training_iteration))
        elif args.type_split =='cross':
            train_metadata_split = pd.read_csv(os.getcwd() + "/split_normal/{}/train_split_{}.csv".format(args.dataset, args.dataset))
            val_metadata_split = pd.read_csv(os.getcwd() + "/split_normal/{}/val_split_{}.csv".format(args.dataset, args.dataset))
    else:
        if args.type_split =='kfold':
            if args.type_data == 'templates':
                train_metadata_split = pd.read_csv(os.getcwd() + "/static/split_kfold/train_split_SIDTD_it_{}.csv".format(training_iteration))
                val_metadata_split = pd.read_csv(os.getcwd() + "/static/split_kfold/val_split_SIDTD_it_{}.csv".format(training_iteration))
            elif args.type_split =='clips_cropped':
                train_metadata_split = pd.read_csv(os.getcwd() + "/static/split_kfold_cropped_unbalanced/train_split_clip_cropped_SIDTD_it_{}.csv".format(training_iteration))
                val_metadata_split = pd.read_csv(os.getcwd() + "/static/split_kfold_cropped_unbalanced/val_split_clip_cropped_SIDTD_it_{}.csv".format(training_iteration))
            elif args.type_split =='clips':
                train_metadata_split = pd.read_csv(os.getcwd() + "/static/split_kfold_unbalanced/train_split_clip_background_SIDTD_it_{}.csv".format(training_iteration))
                val_metadata_split = pd.read_csv(os.getcwd() + "/static/split_kfold_unbalanced/val_split_clip_background_SIDTD_it_{}.csv".format(training_iteration))
        elif args.type_split =='cross':
            if args.type_data =='templates':
                train_metadata_split = pd.read_csv(os.getcwd() + "/static/split_normal/train_split_SIDTD.csv")
                val_metadata_split = pd.read_csv(os.getcwd() + "/static/split_normal/val_split_SIDTD.csv")
            elif args.type_data == 'clips':
                train_metadata_split = pd.read_csv(os.getcwd() + "/static/cross_val_unbalanced/train_split_SIDTD.csv")
                val_metadata_split = pd.read_csv(os.getcwd() + "/static/cross_val_unbalanced/val_split_SIDTD.csv")
            elif args.type_data == 'clips_cropped':
                train_metadata_split = pd.read_csv(os.getcwd() + "/static/cross_val_cropped_unbalanced/train_split_clip_cropped_SIDTD.csv")
                val_metadata_split = pd.read_csv(os.getcwd() + "/static/cross_val_cropped_unbalanced/val_split_clip_cropped_SIDTD.csv")

    
    # Load train label. If you choose to perform forgery augmentation, the images path and labels must be loaded in a dictionnary.
    train_paths = train_metadata_split['image_path'].values.tolist()
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
        trainset = TrainDataset(train_paths, train_ids, transform=train_transforms)
    else:
        trainset = TrainDatasets_augmented(args, dataset_dict['train'], train_paths, transform = train_transforms)
    valset = TrainDataset(val_paths, val_ids, transform=test_transforms)
    

    train_loader = DataLoader(trainset,
                              shuffle=True,
                              batch_size=args.train_batch_size,
                              num_workers=0,
                              drop_last=True,
                              pin_memory=True)
    val_loader = DataLoader(valset,
                            shuffle=True,
                            batch_size=args.eval_batch_size,
                            num_workers=0,
                            pin_memory=True) if valset is not None else None
    

    return train_loader, val_loader

def get_loader_test(args, training_iteration):

    """ Load dataset paths that will be used by the batch generator for inference of trained Trans FG model.
        You can use static path csv to replicate results or choose your own random partitionning
        You can use different type of partitionning: train validation split or kfold cross-validation 
        You can use different type of data: templates, clips or cropped clips.  """

    # Dimension to resize images to fit model dimension
    WIDTH = args.img_size
    HEIGHT = args.img_size

    # Load Data Augmentation function
    test_transforms = get_transforms(WIDTH, HEIGHT, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], data='valid')
    

    # function to load csv with image paths and labels groundtruths of test set in memory
    if args.static == 'no':
        if args.type_split == 'kfold':
            if args.inf_domain_change == 'yes':
                test_metadata_split = pd.read_csv(os.getcwd() + "/split_kfold/{}/test_split_{}_it_0.csv".format(args.dataset, args.dataset))
            else:
                test_metadata_split = pd.read_csv(os.getcwd() + "/split_kfold/{}/test_split_{}_it_{}.csv".format(args.dataset, args.dataset, training_iteration))
        elif args.type_split =='cross':
            test_metadata_split = pd.read_csv(os.getcwd() + "/split_normal/{}/test_split_{}.csv".format(args.dataset, args.dataset))
    else:
        if args.type_split =='kfold':
            if args.type_data == 'templates':
                test_metadata_split = pd.read_csv(os.getcwd() + "/static/split_kfold/test_split_SIDTD_it_{}.csv".format(training_iteration))
            elif args.type_data =='clips_cropped':
                test_metadata_split = pd.read_csv(os.getcwd() + "/static/split_kfold_cropped_unbalanced/test_split_clip_cropped_SIDTD_it_{}.csv".format(training_iteration))
            elif args.type_data =='clips':
                test_metadata_split = pd.read_csv(os.getcwd() + "/static/split_kfold_unbalanced/test_split_clip_background_SIDTD_it_{}.csv".format(training_iteration))
        elif args.type_split =='cross':
            if args.type_data =='templates':
                test_metadata_split = pd.read_csv(os.getcwd() + "/static/split_normal/test_split_SIDTD.csv")
            elif args.type_data == 'clips':
                test_metadata_split = pd.read_csv(os.getcwd() + "/static/cross_val_unbalanced/test_split_SIDTD.csv")
            elif args.type_data == 'clips_cropped':
                test_metadata_split = pd.read_csv(os.getcwd() + "/static/cross_val_cropped_unbalanced/test_split_clip_cropped_SIDTD.csv")

    
    # Load image path and label for validation set
    test_paths = test_metadata_split['image_path'].values.tolist()
    test_ids = test_metadata_split['label'].values.tolist()
    
    print("Test images: ", len(test_paths), 'N test classes: ', len(list(set(test_ids))))
    
    # Load Batch Generator function
    testset = TrainDataset(test_paths, test_ids, transform=test_transforms)   
    test_loader = DataLoader(testset,
                             shuffle=True,
                             batch_size=args.eval_batch_size,
                             num_workers=0,
                             pin_memory=True) if testset is not None else None

    return test_loader