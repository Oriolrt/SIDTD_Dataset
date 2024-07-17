import json
from pathlib import Path
from typing import List, Union
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
import torch

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import random
import glob
NORMALIZE_DEFAULT = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
from albumentations import Compose, Normalize, Resize 
from albumentations.pytorch import ToTensorV2 
from albumentations import RandomCrop, HorizontalFlip, VerticalFlip, RandomBrightnessContrast, CenterCrop, PadIfNeeded, RandomResizedCrop, RandomResizedCrop, RandomBrightnessContrast, RandomShadow, RandomFog, RandomSunFlare

class EasySetCoAARCRandom(Dataset):


    def __init__(self, dataset_name, list_metaclass, image_size=224, training=False):
        """
        Args:
            specs_file: path to the JSON file
            image_size: images returned by the dataset will be square images of the given size
            training: preprocessing is slightly different for a training set, adding a random
                cropping and a random horizontal flip.
        """

        self.training = training
        self.class_names = ["reals", "fakes"]
        self.images, self.images_real, self.labels = self.list_data_instances(dataset_name, list_metaclass)
        print('self.images',len(self.images))
        print('self.images_real', len(self.images_real))
        
        #self.transform = self.compose_transforms(image_size, training)
        if self.training:
            data = 'train'
        else:
            data = 'valid'
        
        self.transform = self.get_transforms(image_size, data)

    @staticmethod
    def get_transforms(image_size, data):
        assert data in ('train', 'valid')
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        WIDTH, HEIGHT = image_size, image_size
        
        if data == 'train':
            return Compose([
            RandomResizedCrop(WIDTH, HEIGHT, scale=(0.8, 1.0)),
            #GridMask(prob=0.3, min_h=10, max_h=50, min_w=10, max_w=50),
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


    @staticmethod
    def list_data_instances(dataset_name: str, list_metaclass: List[str]):
        """
        Explore the directories specified in class_roots to find all data instances.
        Args:
            class_roots: each element is the path to the directory containing the elements
                of one class

        Returns:
            list of paths to the images, and a list of same length containing the integer label
                of each image
        """
        images = []
        images_real = []
        labels = []
        for country in list_metaclass:
            for label in ['reals', 'fakes']:
                if label == 'reals':
                    class_id = 0
                else:
                    class_id = 1
                path_img = glob.glob(f'/data/users/soteria/{dataset_name}/{label}/{country}*')
                images += path_img
                if class_id == 0:
                    images_real = path_img*8
                labels += len(path_img) * [class_id]

        return images, images_real, labels

    def __getitem__(self, item: int):
        """
        Get a data sample from its integer id.
        Args:
            item: sample's integer id

        Returns:
            data sample in the form of a tuple (image, label), where label is an integer.
            The type of the image object depends of the output type of self.transform. By default
            it's a torch.Tensor, however you are free to define any function as self.transform, and
            therefore any type for the output image. For instance, if self.transform = lambda x: x,
            then the output image will be of type PIL.Image.Image.
        """
        # Some images of ILSVRC2015 are grayscale, so we convert everything to RGB for consistence.
        # If you want to work on grayscale images, use torch.transforms.Grayscale in your
        # transformation pipeline.
        img = self.transform(image=np.asarray(Image.open(self.images[item]).convert("RGB")))['image']
        img_real = self.transform(image=np.asarray(Image.open(self.images_real[item]).convert("RGB")))['image']
        label = self.labels[item]

        return img, img_real, label

    def __len__(self) -> int:
        return len(self.labels)


