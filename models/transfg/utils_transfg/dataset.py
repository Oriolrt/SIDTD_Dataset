import cv2

import torch
from torch.utils.data import Dataset

import numpy as np, pandas as pd, copy, torch, random, os

from torch.utils.data import DataLoader, Dataset
from albumentations import Resize 
from PIL import Image
import names
from faker import Faker
import glob
from data_augm_utils import *


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
    


######################## dataloader for forgery data augmentation ##################################

flatten = lambda l: [item for sublist in l for item in sublist]

class TrainDatasets_augmented(Dataset):
    """
    This dataset class allows mini-batch formation pre-epoch, for greater speed
    """
    def __init__(self, args, image_dict, path_img, transform=None):
        """
        Args:
            image_dict: two-level dict, `super_dict[super_class_id][class_id]` gives the list of
                        image paths having the same super-label and class label
        """
        self.image_dict = image_dict
        self.batch_size = args.batch_size
        self.faker_data_augmentation = args.faker_data_augmentation
        self.shift_crop = args.shift_crop
        self.shift_copy = args.shift_copy
        self.samples_per_class = self.batch_size // 4
        self.path_img = path_img
        
        # checks
        # provide avail_classes
        self.avail_classes = [*self.image_dict]
        # Data augmentation/processing methods.
        self.transform = transform

        self.reshuffle()


    def reshuffle(self):

        image_dict = copy.deepcopy(self.image_dict) 
        print('shuffling data')
        for sub in image_dict:
            random.shuffle(image_dict[sub])

        total_batches = []
        batch = []
        finished = 0
        if len(image_dict[0]) >= 2*len(image_dict[1]):
            while finished == 0:
                if (len(image_dict[0]) >= 2*self.samples_per_class):
                    l_clips = []
                    list_fakes_img = list(np.random.choice(image_dict[1],self.samples_per_class))
                    l_clips = l_clips + [(1, x) for x in list_fakes_img]
                
                    list_reals_img = list(image_dict[0][:2*self.samples_per_class])
                    l_clips = l_clips + [(0, x) for x in list_reals_img]
                    image_dict[0] = image_dict[0][2*self.samples_per_class:] 

                    list_forgery_augm_img = list(np.random.choice(list_reals_img,self.samples_per_class))
                    l_clips = l_clips + [(2, x) for x in list_forgery_augm_img]                
                    batch.append(l_clips)

                if len(batch) == 1:
                    total_batches.append(batch)
                    batch = []
                else:
                    finished = 1
        else:
            while finished == 0:
                if (len(image_dict[1]) >= self.samples_per_class):
                    l_clips = []
                    list_reals_img = list(np.random.choice(image_dict[0],2*self.samples_per_class))
                    l_clips = l_clips + [(0, x) for x in list_reals_img]
                
                    list_fakes_img = list(image_dict[1][:self.samples_per_class])
                    l_clips = l_clips + [(1, x) for x in list_fakes_img]
                    image_dict[1] = image_dict[1][self.samples_per_class:] 

                    list_forgery_augm_img = list(np.random.choice(list_reals_img,self.samples_per_class))
                    l_clips = l_clips + [(2, x) for x in list_forgery_augm_img]                
                    batch.append(l_clips)
                
                if len(batch) == 1:
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
        
        l_fake_type = ['crop', 'inpainting', 'copy']
        if self.faker_data_augmentation:
            if batch_item[0] == 2:
                cls = 1
                fake_type = random.choice(l_fake_type)
                id_img = batch_item[1].split('/')[-1]
                id_country = id_img[:3]
                path = 'split_kfold/clip_cropped_MIDV2020/annotations/annotation_' + id_country + '.json' 
                annotations = read_json(path)
                    
                if fake_type == 'copy':
                    image = CopyPaste(image, annotations, self.shift_copy)

                if fake_type == 'inpainting':
                    image = Inpainting(image, annotations, id_country)

                if fake_type == 'crop':

                    if id_country in ['rus', 'grc']:
                        list_image_field = ['image']
                    else:
                        list_image_field = ['image', 'signature']
                        
                    dim_issue = True
                    while dim_issue:
                        img_path_clips_target = random.choice(self.path_img)
                        id_country_target = img_path_clips_target.split('/')[-1][:3]
                        if id_country_target in ['rus', 'grc']:
                            if 'signature' in list_image_field:
                                list_image_field.remove('signature')

                        image_target = cv2.imread(img_path_clips_target)

                        path = 'split_kfold/clip_cropped_MIDV2020/annotations/annotation_' + id_country_target + '.json'
                        annotations_target = read_json(path)
                            
                        image, dim_issue = CropReplace_v2(image, annotations, image_target, annotations_target, list_image_field, self.shift_crop)

        augmented = self.transform(image=image)
        image = augmented['image']

        return  image, cls


    def __len__(self):
        return len(self.dataset)


def CopyPaste(images, annotations, shift_copy):
    """Copy a text randomly chosen among the field available and Paste in a random text field area.

    Args:
        prob (float): probability to perform CopyPaste. Must be between 0 and 1.
    """

    list_text_field = list(annotations.keys())
    if 'image' in list_text_field:
        list_text_field.remove('image')
    if 'photo' in list_text_field:
        list_text_field.remove('photo')
    if 'signature' in list_text_field:
        list_text_field.remove('signature')
    if 'face' in list_text_field:
        list_text_field.remove('face')
    
    dim_issue = True
    while dim_issue:
        source_field_to_change_txt = random.choice(list_text_field)
        target_field_to_change_txt = random.choice(list_text_field)
        source_info_txt = annotations[source_field_to_change_txt]
        target_info_txt = annotations[target_field_to_change_txt]
        img_tr, dim_issue = copy_paste_on_document(images, source_info_txt, target_info_txt, shift_copy)
    
    return img_tr


def CropReplace_v2(image, annotations, image_target, annotations_target, list_image_field, shift_crop):
    """Copy a text randomly chosen among the field available and Paste in a random text field area.

    Args:
        prob (float): probability to perform CopyPaste. Must be between 0 and 1.
    """


    field_to_change = random.choice(list_image_field)
    info_source = annotations[field_to_change]
    if field_to_change == 'photo':
        field_to_change = 'image'
    info_target = annotations_target[field_to_change]
    img_tr, dim_issue = copy_paste_on_two_documents(image, info_source, image_target, info_target, shift_crop)
    return img_tr, dim_issue


def Inpainting(image, annotations, id_country):
    """Copy a text randomly chosen among the field available and Paste in a random text field area.

    Args:
        prob (float): probability to perform CopyPaste. Must be between 0 and 1.
    """
    list_text_field = list(annotations.keys())
    if 'image' in list_text_field:
        list_text_field.remove('image')
    if 'photo' in list_text_field:
        list_text_field.remove('photo')
    if 'signature' in list_text_field:
        list_text_field.remove('signature')
    if 'face' in list_text_field:
        list_text_field.remove('face')
    field_to_change = random.choice(list_text_field)

    if field_to_change == 'name':
        text_str = names.get_first_name()
    elif field_to_change == 'surname':
        text_str = names.get_last_name()
    elif field_to_change == 'sex':
        if id_country in ['esp', 'alb', 'fin', 'grc', 'svk']:
            text_str = random.choice(['F','M'])
        else:
            text_str = random.choice(['K/M','N/F'])
    elif field_to_change == 'nationality':
        if id_country == 'esp':
            text_str = 'ESP'
        elif id_country == 'alb':
            text_str = 'Shqiptare/Albanian'
        elif id_country == 'aze':
            text_str = 'AZORBAYCA/AZERBAIJAN'
        elif id_country == 'est':
            text_str = 'EST'
        elif id_country == 'fin':
            text_str = 'FIN'
        elif id_country == 'grc':
            text_str = 'EAAHNIKH/HELLENIC'
        elif id_country == 'lva':
            text_str = 'LVA'
        elif id_country == 'rus':
            text_str = 'AOMNHNKA'
        elif id_country == 'srb':
            text_str = 'SRPSKO'
        elif id_country == 'svk':
            text_str = 'SVK'
    elif field_to_change == 'birthdate':
        fake = Faker()
        t = fake.date_time_between(start_date='-60y', end_date='-18y')
        text_str = t.strftime('%d %m %Y')
    
    swap_info = annotations[field_to_change]
    img_tr = inpaint_image(img=image, swap_info=swap_info, text_str=text_str)
    return img_tr