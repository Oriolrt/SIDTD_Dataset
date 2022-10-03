"""
taken and modified from https://github.com/pranv/ARC
"""
import os
from tqdm import tqdm
import imageio
import numpy as np
from random import random

from numpy.random import choice
import torch
from torch.autograd import Variable
import pandas as pd

#from scipy.misc import imresize as resize
import cv2

from image_augmenter import ImageAugmenter

use_cuda = True


class Binary(object):
    def __init__(self, batch_size=128, image_size=224):
        """
        path: dataset path folder
        batch_size: the output is (2 * batch size, 1, image_size, image_size)
                    X[i] & X[i + batch_size] are the pair
        image_size: size of the image
        load_file: json file dict with split data information
        ---------------------
        Data Augmentation Parameters:
            flip: here flipping both the images in a pair
            scale: x would scale image by + or - x%
            rotation_deg
            shear_deg
            translation_px: in both x and y directions
        """
       
            

                
        self.image_size = image_size
        self.batch_size = batch_size

        self.mean_pixel = 0.5 #self.compute_mean()# used later for mean subtraction
        
        
        flip = True
        scale = 0.2
        rotation_deg = 20
        shear_deg = 10
        translation_px = 5
        self.augmentor = ImageAugmenter(image_size, image_size, channel_is_first_axis=True,
                                        hflip=flip, vflip=flip,
                                        scale_to_percent=1.0 + scale, rotation_deg=rotation_deg, shear_deg=shear_deg,
                                        translation_x_px=translation_px, translation_y_px=translation_px)


                
        
    def compute_mean(self):
        """
        Time consuming, ToDo: load precalculated if it exists.
        """
        count_images = 0
        # Read al the banknote images to comput the mean 
        for folder in tqdm(self.folders):
            for real_or_fake in self.images_dict[folder]:
                for image_path in self.images_dict[folder][real_or_fake]:
                
                    image = cv2.resize(imageio.imread(image_path), (self.image_size, self.image_size))
                    #initialize variable to compute the mean
                    if count_images == 0:
                        if image.shape[-1] == 3:
                            mean_image = np.zeros_like(image, dtype = np.float32())
                        else:
                            mean_image = np.zeros_like(image[...,:-1])
                    if image.shape[-1] == 3:
                        mean_image += image
                    else:
                        mean_image += image[...,:-1]
                    
                    count_images += 1
        mean_image = mean_image/count_images
        return np.moveaxis(mean_image, -1, 0)/255.0
    
    


class Batcher(Binary):
    def __init__(self, paths_splits, batch_size=32, image_size=224):
        Binary.__init__(self, image_size)

        image_size = self.image_size
        self.paths_splits = paths_splits
        


    def fetch_batch(self, part, labels: str = None, image_paths: str = None, batch_size: int = 32):

        if batch_size is None:
            batch_size = self.batch_size

        if part == 'test':
            X, Y = self._fetch_eval(part, labels, image_paths, batch_size)
        else:
            X, Y = self._fetch_batch(part, batch_size)
        X = Variable(torch.from_numpy(X))#.view(2*batch_size, self.image_size, self.image_size)

        X1 = X[:batch_size]  # (B, c, h, w)
        X2 = X[batch_size:]  # (B, c, h, w)

        X = torch.stack([X1, X2], dim=1)  # (B, 2, c, h, w)

        Y = Variable(torch.from_numpy(Y))

        if use_cuda:
            X, Y = X.cuda(), Y.cuda()

        return X, Y

    def _fetch_batch(self, part, batch_size: int = None):
        if batch_size is None:
            batch_size = self.batch_size


        paths_splits = self.paths_splits[part]
        
        image_size = self.image_size

        #build the input pairs
        X = np.zeros((2 * batch_size, 3, image_size, image_size), dtype='uint8')
        y = np.zeros((batch_size, 1), dtype='int32')
        
        fakes_or_reals = np.asarray([list(np.zeros(batch_size//2)),list(np.ones(batch_size//2))]).reshape(batch_size)
        draw_f_or_r = choice(fakes_or_reals,batch_size, replace=False)
        
        idx_support = choice(len(paths_splits['reals']), batch_size, replace=False)
        idx_test_reals = choice(len(paths_splits['reals']), batch_size//2, replace=False)
        idx_test_fakes = choice(len(paths_splits['fakes']), batch_size//2, replace=False)
        
        r = 0
        f = 0
        for i in range(batch_size):
            fake_or_true = draw_f_or_r[i]
            idx1 = idx_support[i]
            X[i] = paths_splits['reals'][idx1]
            if fake_or_true == 0: 
                # choose one real image
                idx2 = idx_test_reals[r]
                r = r + 1
                X[i + batch_size] = paths_splits['reals'][idx2]
                y[i] = 0
            else:
                # choose one fake image
                idx2 = idx_test_fakes[f]
                f = f + 1
                X[i + batch_size] = paths_splits['fakes'][idx2]
                y[i] = 1

        if part == 'train':
            X = self.augmentor.augment_batch(X)
        if part == 'val':
            X = X / 255.0
        X = X.astype("float32")

        return X, y

    def _fetch_eval(self, part, labels, image_paths, batch_size: int = 32):
        ''' 
            To load a batch of test data into the model so that 2-way one-shot classification 
            can be conducted, match each test image with every image in support set:
            
            Test     Support Set     Labels
            Img1  |  True image 1    1 if Img fake, else 0

            Img n |  True image n    1 if Img fake, else 0
            The test and support sets are outputted from  _fetch_eval() 
            in a single column, then matched horizontally in fetch_batch() (like above  ) 
        '''
        if batch_size is None:
            batch_size = self.batch_size

        paths_splits = self.paths_splits[part]
        
        image_size = self.image_size

        #build the input pairs
        X = np.zeros((2 * batch_size, 3, image_size, image_size), dtype='uint8')
        y = np.zeros((batch_size, 1), dtype='int32')
        i = 0 
        for lbl, img in zip(labels, image_paths):
            idx = choice(len(paths_splits['reals']))
            X[i] = paths_splits['reals'][idx]
            X[i + batch_size] = self.read_image(img, image_size)
            if lbl == 'reals':
                y[i] = 0
            if lbl == 'fakes':
                y[i] = 1
            i = i + 1

        
        X = X / 255.0
        X = X.astype("float32")

        return X, y
    
    def read_image(self, image_path, image_size):
        image = imageio.imread(image_path)
        if image.shape[-1]>=4:
            image = image[...,:-1]
        image = cv2.resize(image, (image_size,image_size))
        
        return np.moveaxis(image, -1, 0) 


    def _fetch_test_all_class(self, part, labels, image_paths, batch_size: int = 32):
        ''' 
            To load a batch of test data into the model so that 2-way one-shot classification 
            can be conducted, match each test image with every image in support set:
            
            Test     Support Set     Labels
            Img1  |  True image 1    1 if Img fake, else 0

            Img n |  True image n    1 if Img fake, else 0
            The test and support sets are outputted from  _fetch_eval() 
            in a single column, then matched horizontally in fetch_batch() (like above  ) 
        '''

        paths_splits = self.paths_splits[part]
        n_class = 2
        image_size = self.image_size
        X = np.zeros((2 * batch_size, 3, self.image_size, self.image_size), dtype='uint8')
        i = 0
        label = []
        fakes_reals = ['fakes','reals']
        for lbl, img in zip(labels, image_paths):
            label = label + list(np.tile(lbl,n_class))
            for j in range(n_class):
                diff_path = choice(paths_splits[fakes_reals[j]])
                X[i + j] = self.read_image(diff_path, image_size)
                X[ batch_size + i + j] = self.read_image(img, image_size)
            i = i + 2
        fakes_reals = np.tile(fakes_reals, batch_size//2)
        y = np.array(np.array(label) == fakes_reals).astype('int32')
        X = X / 255.0
        X = X.astype("float32")

        return X, y   