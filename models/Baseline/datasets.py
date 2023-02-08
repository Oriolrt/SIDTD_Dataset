# repo originally forked from https://github.com/Confusezius/Deep-Metric-Learning-Baselines and from https://github.com/Andrew-Brown1/Smooth_AP/blob/master/src/datasets.py

################# LIBRARIES ###############################
import warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, copy, torch, random, os

from torch.utils.data import Dataset
from PIL import Image
import cv2



######################## dataset for SmoothAP regular training ##################################

flatten = lambda l: [item for sublist in l for item in sublist]

class TrainDatasetsmoothap(Dataset):
    """
    This dataset class allows mini-batch formation pre-epoch, for greater speed
    """
    def __init__(self, image_dict, args, N_CLASSES=2, transform=None):
        """
        Args:
            image_dict: two-level dict, `super_dict[super_class_id][class_id]` gives the list of
                        image paths having the same super-label and class label
        """
        self.image_dict = image_dict
        self.dataset_name = args.dataset
        self.batch_size = args.batch_size
        self.nclasses = N_CLASSES
        self.samples_per_class = args.batch_size // N_CLASSES
        
        for sub in self.image_dict:
            newsub = []
            for instance in self.image_dict[sub]:
                newsub.append((sub, instance))
            self.image_dict[sub] = newsub
        
        # checks
        # provide avail_classes
        self.avail_classes = [*self.image_dict]
        # Data augmentation/processing methods.
        self.transform = transform

        self.reshuffle()


    def ensure_3dim(self, img):
        if len(img.size) == 2:
            img = img.convert('RGB')
        return img


    def reshuffle(self):

        image_dict = copy.deepcopy(self.image_dict) 
        print('shuffling data')
        for sub in image_dict:
            random.shuffle(image_dict[sub])
        
        classes = [*image_dict]
        random.shuffle(classes)
        total_batches = []
        batch = []
        finished = 0
        while finished == 0:
            for sub_class in classes:
                if (len(image_dict[sub_class]) >= self.samples_per_class) and (len(batch) < self.nclasses) :
                    batch.append(image_dict[sub_class][:self.samples_per_class])
                    image_dict[sub_class] = image_dict[sub_class][self.samples_per_class:] 

            if len(batch) == self.nclasses:
                total_batches.append(batch)
                batch = []
            else:
                finished = 1

        
        random.shuffle(total_batches)
        self.dataset = flatten(flatten(total_batches))

    def __getitem__(self, idx):
        # we use SequentialSampler together with SuperLabelTrainDataset,
        # so idx==0 indicates the start of a new epoch
        batch_item = self.dataset[idx]
        cls = batch_item[0]
        image = cv2.imread(batch_item[1])
        try:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except:
            print('ERROR')
        
        augmented = self.transform(image=image)
        image = augmented['image']

        return  image, cls


    def __len__(self):
        return len(self.dataset)