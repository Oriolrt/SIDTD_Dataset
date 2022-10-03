import logging
from PIL import Image
import os

import torch
import pandas as pd


from torchvision import transforms
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler

from .dataset import TrainDataset, TrainDatasetConditionned
from .autoaugment import AutoAugImageNetPolicy


from albumentations import Compose, Normalize, Resize 
from albumentations.pytorch import ToTensorV2 
from albumentations import RandomCrop, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, CenterCrop, PadIfNeeded, RandomResizedCrop



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
    
    
    
def get_loader_conditionned(args, training_iteration):
    WIDTH = args.img_size
    HEIGHT = args.img_size

    train_transforms = get_transforms(WIDTH, HEIGHT, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], data='train')
    test_transforms = get_transforms(WIDTH, HEIGHT, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], data='valid')
    
    train_metadata_split = pd.read_csv(args.dataset_csv_path + "/{}/train_split_{}_it_{}.csv".format(args.dataset, args.dataset, training_iteration))
    
    val_metadata_split = pd.read_csv(args.dataset_csv_path + "/{}/val_split_{}_it_{}.csv".format(args.dataset, args.dataset, training_iteration))
    
    test_metadata_split = pd.read_csv(args.dataset_csv_path + "/{}/test_split_{}_it_{}.csv".format(args.dataset, args.dataset, training_iteration))

    train_paths = train_metadata_split['image_path'].values.tolist()
    train_ids = train_metadata_split['label'].values.tolist()
    train_class = train_metadata_split['class'].values.tolist()
    
    val_paths = val_metadata_split['image_path'].values.tolist()
    val_ids = val_metadata_split['label'].values.tolist()
    val_class = val_metadata_split['class'].values.tolist()
    
    test_paths = test_metadata_split['image_path'].values.tolist()
    test_ids = test_metadata_split['label'].values.tolist()
    test_class = test_metadata_split['class'].values.tolist()
    
    print("Training images: ", len(train_paths),'N training classes:', len(list(set(train_ids))))
    print("Validation images: ", len(val_paths), 'N val classes: ', len(list(set(val_ids))))
    print("Test images: ", len(test_paths), 'N test classes: ', len(list(set(test_ids))))
    
    trainset = TrainDatasetConditionned(train_paths, train_ids, train_class, transform=train_transforms)
    valset = TrainDatasetConditionned(val_paths, val_ids, val_class, transform=test_transforms)
    testset = TrainDatasetConditionned(test_paths, test_ids, test_class, transform=test_transforms)


    train_loader = DataLoader(trainset,
                              shuffle=True,
#                             sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              drop_last=True,
                              pin_memory=True)
    val_loader = DataLoader(valset,
#                            sampler=val_sampler,
                            shuffle=True,
                            batch_size=args.eval_batch_size,
                            num_workers=4,
                            pin_memory=True) if valset is not None else None
    
    test_loader = DataLoader(testset,
#                             sampler=test_sampler,
                             shuffle=True,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, val_loader, test_loader

def get_loader(args, training_iteration):
    WIDTH = args.img_size
    HEIGHT = args.img_size
        
#    if args.local_rank not in [-1, 0]:
#        torch.distributed.barrier()

    train_transforms = get_transforms(WIDTH, HEIGHT, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], data='train')
    test_transforms = get_transforms(WIDTH, HEIGHT, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], data='valid')
    
    train_metadata_split = pd.read_csv(args.dataset_csv_path + "/{}/train_split_{}_it_{}.csv".format(args.dataset, args.dataset, training_iteration))
    
    val_metadata_split = pd.read_csv(args.dataset_csv_path + "/{}/val_split_{}_it_{}.csv".format(args.dataset, args.dataset, training_iteration))
    
    test_metadata_split = pd.read_csv(args.dataset_csv_path + "/{}/test_split_{}_it_{}.csv".format(args.dataset, args.dataset, training_iteration))
            
    train_paths = train_metadata_split['image_path'].values.tolist()
    train_ids = train_metadata_split['label'].values.tolist()
    
    val_paths = val_metadata_split['image_path'].values.tolist()
    val_ids = val_metadata_split['label'].values.tolist()
    
    test_paths = test_metadata_split['image_path'].values.tolist()
    test_ids = test_metadata_split['label'].values.tolist()
    
    print("Training images: ", len(train_paths),'N training classes:', len(list(set(train_ids))))
    print("Validation images: ", len(val_paths), 'N val classes: ', len(list(set(val_ids))))
    print("Test images: ", len(test_paths), 'N test classes: ', len(list(set(test_ids))))
    
    trainset = TrainDataset(train_paths, train_ids, transform=train_transforms)
    valset = TrainDataset(val_paths, val_ids, transform=test_transforms)
    testset = TrainDataset(test_paths, test_ids, transform=test_transforms)


#    if args.local_rank == 0:
#        torch.distributed.barrier()

#    train_sampler = RandomSampler(trainset) if args.local_rank == -1 else DistributedSampler(trainset)
#    val_sampler = SequentialSampler(valset) if args.local_rank == -1 else DistributedSampler(valset)
#    test_sampler = SequentialSampler(testset) if args.local_rank == -1 else DistributedSampler(testset)
    

    train_loader = DataLoader(trainset,
                              shuffle=True,
#                             sampler=train_sampler,
                              batch_size=args.train_batch_size,
                              num_workers=4,
                              drop_last=True,
                              pin_memory=True)
    val_loader = DataLoader(valset,
#                            sampler=val_sampler,
                            shuffle=True,
                            batch_size=args.eval_batch_size,
                            num_workers=4,
                            pin_memory=True) if valset is not None else None
    
    test_loader = DataLoader(testset,
#                             sampler=test_sampler,
                             shuffle=True,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return train_loader, val_loader, test_loader

def get_test_img(args, training_iteration):
    WIDTH = args.img_size
    HEIGHT = args.img_size

    test_transforms = get_transforms(WIDTH, HEIGHT, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], data='valid')
 
    test_metadata_split = pd.read_csv(args.dataset_csv_path + "/split_kfold/{}/test_split_{}_it_{}.csv".format(args.dataset, args.dataset, training_iteration))
    
    test_paths = test_metadata_split['image_path'].values.tolist()
    test_ids = test_metadata_split['label'].values.tolist()

    print("Test images: ", len(test_paths), 'N test classes: ', len(list(set(test_ids))))

    testset = TrainDataset(test_paths, test_ids, transform=test_transforms)
    
    
    test_loader = DataLoader(testset,
#                             sampler=test_sampler,
                             shuffle=False,
                             batch_size=args.eval_batch_size,
                             num_workers=4,
                             pin_memory=True) if testset is not None else None

    return test_loader, test_paths