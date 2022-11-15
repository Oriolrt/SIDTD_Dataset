import logging

import torch
import pandas as pd
import os


from torch.utils.data import DataLoader

from .dataset import TrainDataset


from albumentations import Compose, Normalize, Resize 
from albumentations.pytorch import ToTensorV2 
from albumentations import HorizontalFlip, VerticalFlip, RandomBrightnessContrast, RandomResizedCrop



logger = logging.getLogger(__name__)

def get_transforms(WIDTH, HEIGHT, mean, std, data):
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
    WIDTH = args.img_size
    HEIGHT = args.img_size

    train_transforms = get_transforms(WIDTH, HEIGHT, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], data='train')
    test_transforms = get_transforms(WIDTH, HEIGHT, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], data='valid')
    
    if args.type_split =='kfold':
        train_metadata_split = pd.read_csv(os.getcwd() + "/split_kfold/{}/train_split_{}_it_{}.csv".format(args.dataset, args.dataset, training_iteration))
        val_metadata_split = pd.read_csv(os.getcwd() + "/split_kfold/{}/val_split_{}_it_{}.csv".format(args.dataset, args.dataset, training_iteration))
    elif args.type_split =='cross':
        train_metadata_split = pd.read_csv(os.getcwd() + "/split_normal/{}/train_split_{}.csv".format(args.dataset, args.dataset))
        val_metadata_split = pd.read_csv(os.getcwd() + "/split_normal/{}/val_split_{}.csv".format(args.dataset, args.dataset))
    else:
        train_metadata_split = pd.read_csv(os.getcwd() + "/static_cross_val/{}/train_split_{}.csv".format(args.dataset, args.dataset))
        val_metadata_split = pd.read_csv(os.getcwd() + "/static_cross_val/{}/val_split_{}.csv".format(args.dataset, args.dataset))
    
    train_paths = train_metadata_split['image_path'].values.tolist()
    train_ids = train_metadata_split['label'].values.tolist()
    
    val_paths = val_metadata_split['image_path'].values.tolist()
    val_ids = val_metadata_split['label'].values.tolist()

    
    print("Training images: ", len(train_paths),'N training classes:', len(list(set(train_ids))))
    print("Validation images: ", len(val_paths), 'N val classes: ', len(list(set(val_ids))))
    
    trainset = TrainDataset(train_paths, train_ids, transform=train_transforms)
    valset = TrainDataset(val_paths, val_ids, transform=test_transforms)
    

    train_loader = DataLoader(trainset,
                              shuffle=True,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              drop_last=True,
                              pin_memory=True)
    val_loader = DataLoader(valset,
                            shuffle=True,
                            batch_size=args.eval_batch_size,
                            num_workers=4,
                            pin_memory=True) if valset is not None else None
    

    return train_loader, val_loader

def get_loader_test(args, training_iteration):
    WIDTH = args.img_size
    HEIGHT = args.img_size

    test_transforms = get_transforms(WIDTH, HEIGHT, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], data='valid')
    
    test_metadata_split = pd.read_csv(args.csv_dataset_path + "/{}/test_split_{}_it_{}.csv".format(args.dataset, args.dataset, training_iteration))

    if args.type_split =='kfold':
        test_metadata_split = pd.read_csv(os.getcwd() + "/split_kfold/{}/test_split_{}_it_{}.csv".format(args.dataset, args.dataset, training_iteration))
    elif args.type_split =='cross':
        test_metadata_split = pd.read_csv(os.getcwd() + "/split_normal/{}/test_split_{}.csv".format(args.dataset, args.dataset))
    else:
        test_metadata_split = pd.read_csv(os.getcwd() + "/static_cross_val/{}/test_split_{}.csv".format(args.dataset, args.dataset))
    
    test_paths = test_metadata_split['image_path'].values.tolist()
    test_ids = test_metadata_split['label'].values.tolist()
    
    print("Test images: ", len(test_paths), 'N test classes: ', len(list(set(test_ids))))
    
    testset = TrainDataset(test_paths, test_ids, transform=test_transforms)


    
    test_loader = DataLoader(testset,
                             shuffle=True,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return test_loader