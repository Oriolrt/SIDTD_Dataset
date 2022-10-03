# repo originally forked from https://github.com/Confusezius/Deep-Metric-Learning-Baselines and from https://github.com/Andrew-Brown1/Smooth_AP/blob/master/src/datasets.py

################# LIBRARIES ###############################
import warnings
warnings.filterwarnings("ignore")

import numpy as np, pandas as pd, copy, torch, random, os

from torch.utils.data import Dataset
from PIL import Image
import cv2


################## BASIC PYTORCH DATASET USED FOR ALL DATASETS ##################################
class BaseTripletDataset(Dataset):
    """
    Dataset class to provide (augmented) correctly prepared training samples corresponding to standard DML literature.
    This includes normalizing to ImageNet-standards, and Random & Resized cropping of shapes 224 for ResNet50 and 227 for
    GoogLeNet during Training. During validation, only resizing to 256 or center cropping to 224/227 is performed.
    """
    def __init__(self, image_dict, args, is_validation=False, transform=None):
        """
        Dataset Init-Function.
        Args:
            image_dict:         dict, Dictionary of shape {class_idx:[list of paths to images belong to this class] ...} providing all the training paths and classes.
            args:                argparse.Namespace, contains all training-specific parameters.
            samples_per_class:  Number of samples to draw from one class before moving to the next when filling the batch.
            is_validation:      If is true, dataset properties for validation/testing are used instead of ones for training.
        Returns:
            Nothing!
        """
        #Define length of dataset
        self.n_files     = np.sum([len(image_dict[key]) for key in image_dict.keys()])

        self.is_validation = is_validation

        self.pars        = args
        self.image_dict  = image_dict

        self.avail_classes    = sorted(list(self.image_dict.keys()))

        # Convert image dictionary from classname:content to class_idx:content, because the initial indices are not necessarily from 0 - <n_classes>.
        self.image_dict    = {i:self.image_dict[key] for i,key in enumerate(self.avail_classes)}
        self.avail_classes = sorted(list(self.image_dict.keys()))

        # Init. properties that are used when filling up batches.
        if not self.is_validation:
            self.samples_per_class = args.batch_size // len(self.avail_classes)
            # Select current class to sample images from up to <samples_per_class>
            self.current_class   = np.random.randint(len(self.avail_classes))
            self.classes_visited = [self.current_class, self.current_class]
            self.n_samples_drawn = 0

        # Data augmentation/processing methods.
        self.transform = transform

        # Convert Image-Dict to list of (image_path, image_class). Allows for easier direct sampling.
        self.image_list = [[(x,key) for x in self.image_dict[key]] for key in self.image_dict.keys()]
        self.image_list = [x for y in self.image_list for x in y]

        # Flag that denotes if dataset is called for the first time.
        self.is_init = True


    def augment_data(self, path):
        """
        Function that read image path and performs data augmentation.
        Args:
            path: file to read the image
        Returns:
            Augmented image
        """
        img =  cv2.imread(path)
        try:
            image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            print(path)
        augmented = self.transform(image = image)
        image = augmented['image']
        
        return image


    def __getitem__(self, idx):
        """
        Args:
            idx: Sample idx for training sample
        Returns:
            tuple of form (sample_class, torch.Tensor() of input image)
        """
        if self.pars.loss == 'smoothap' or self.pars.loss == 'smoothap_element':
            if self.is_init:
                #self.current_class = self.avail_classes[idx%len(self.avail_classes)]
                self.is_init = False
        
            if not self.is_validation:
                if self.samples_per_class==1:
                    image =  self.augment_data(self,self.image_list[idx][0])
                    return image, self.image_list[idx][-1]

                if self.n_samples_drawn==self.samples_per_class:
                    #Once enough samples per class have been drawn, we choose another class to draw samples from.
                    #Note that we ensure with self.classes_visited that no class is chosen if it had been chosen
                    #previously or one before that.
                    counter = copy.deepcopy(self.avail_classes)
                    for prev_class in self.classes_visited:
                        if prev_class in counter: counter.remove(prev_class)

                    self.current_class   = counter[idx%len(counter)]
                    #self.classes_visited = self.classes_visited[1:]+[self.current_class]
                    # EDIT -> there can be no class repeats
                    self.classes_visited = self.classes_visited+[self.current_class]
                    self.n_samples_drawn = 0

                class_sample_idx = idx%len(self.image_dict[self.current_class])
                self.n_samples_drawn += 1

                image =  self.augment_data(self.image_dict[self.current_class][class_sample_idx])
                
                return image, self.current_class
            else:
                image =  self.augment_data(self.image_list[idx][0])
                return image, self.image_list[idx][-1]
        else:
            if self.is_init:
                self.current_class = self.avail_classes[idx%len(self.avail_classes)]
                self.is_init = False
            if not self.is_validation:
                if self.samples_per_class==1:

                    image =  self.augment_data(self.image_list[idx][0])

                    return image, self.image_list[idx][-1]

                if self.n_samples_drawn==self.samples_per_class:
                    #Once enough samples per class have been drawn, we choose another class to draw samples from.
                    #Note that we ensure with self.classes_visited that no class is chosen if it had been chosen
                    #previously or one before that.
                    counter = copy.deepcopy(self.avail_classes)
                    for prev_class in self.classes_visited:
                        if prev_class in counter: counter.remove(prev_class)

                    self.current_class   = counter[idx%len(counter)]
                    self.classes_visited = self.classes_visited[1:]+[self.current_class]
                    self.n_samples_drawn = 0

                class_sample_idx = idx%len(self.image_dict[self.current_class])
                self.n_samples_drawn += 1

                image =  self.augment_data(self.image_dict[self.current_class][class_sample_idx])

                return image, self.current_class
            else:

                image =  self.augment_data(self.image_list[idx][0])

                return image, self.image_list[idx][-1]

    def __len__(self):
        return self.n_files

flatten = lambda l: [item for sublist in l for item in sublist]

######################## dataset for SmoothAP regular training ##################################



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
            print(file_path)
        
        augmented = self.transform(image=image)
        image = augmented['image']

        return  image, cls


    def __len__(self):
        return len(self.dataset)