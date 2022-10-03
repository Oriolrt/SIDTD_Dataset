import os
import json
from os.path import join

import numpy as np
import scipy
from scipy import io
import scipy.misc
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import torch
from torch.utils.data import Dataset
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url, list_dir, check_integrity, extract_archive, verify_str_arg


class TrainDataset(Dataset):
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
        image = cv2.imread(file_path)
    
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print(file_path)
        if self.dataset_crop:
            sx, sy, _ = image.shape
            sx2 = int(sx/2)
            sy2 = int(sy/2)
            image = image[sx2-250:sx2+250, sy2-250:sy2+250, :]
            
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
    
        return image, label



class TrainDatasetConditionned(Dataset):
    def __init__(self, paths, ids, meta_class, transform=None, dataset_crop=False):
        self.paths = paths
        self.ids = ids
        self.meta_class = meta_class
        self.transform = transform
        self.dataset_crop = dataset_crop
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx):
        file_path = self.paths[idx]
        id_meta_class = self.meta_class[idx]
        label = self.ids[idx]
        image = cv2.imread(file_path)
    
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print(file_path)
        if self.dataset_crop:
            sx, sy, _ = image.shape
            sx2 = int(sx/2)
            sy2 = int(sy/2)
            image = image[sx2-250:sx2+250, sy2-250:sy2+250, :]
            
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        img_meta_class = np.ones((image.shape[0], image.shape[1],1)) * id_meta_class
        image = np.concatenate((image, img_meta_class), axis=2)
    
        return image, label