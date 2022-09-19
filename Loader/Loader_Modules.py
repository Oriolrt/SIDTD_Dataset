from ast import arg, parse
import numpy as np
import pandas as pd
from Datasets import *
from abc import ABC, abstractmethod
import time
import os
import sys
import logging
import argparse
import glob
from sklearn.utils import shuffle
from distutils.dir_util import copy_tree
import shutil
import imageio
from typing import *
from PIL.Image import Image

class DataLoader(object):


    ##################################################################
    class _Data(object):

        __slots__ = ["_sample", "_gt", "_class", "_metaclass"]
        def __init__(self, sample, gt, clas, metaclass= None):
            self._sample = sample
            self._gt = gt
            self._clas = clas
            self._metaclass = metaclass


        @property
        def sample(self):
            return self._sample

        @property
        def gt(self):
            return self._gt

        @property
        def clas(self):
            return self._class

        @property
        def metaclass(self):
            return self._metaclass

        def __getitem__(self, item):
            pass




    def __init__(self, dataset:str="Midv", type_split:str = "normal", batch_size: int = 1,kfold_split:int=10, normal_split:list=[0.8,0.1,0.1]
, few_shot_split:str=None, conditioned:bool = True):

        """
            Input of the class:         dataset --> Define what kind of the different datasets do you want to download [Midv, Dogs, Fungus, Findit, Banknotes]
                                                    This datasets have been changed in order to the different approach we are working on
                                        Type_split --> Diferent kind of split for train the models. The diferents splits are [kfold, normal or few_shot]

                                        batch_size --> define the batch of the training set

                                        kfold_, normal, few_shot_ (split) --> define the behaviour of the split based on what kind of split did you put

                                        conditioned --> flag to define if you want to train with the metaclasses inside the dataset thath downloaded 
        
        """

        ### ASSERTS AND ERROR CONTROL ###

        if type_split == "normal": assert (type(normal_split) == list and np.sum(normal_split) == 1)
        elif type_split == "kfold": assert (kfold_split > 0 and type(kfold_split) == int)
        elif type_split == "shot": pass #TODO veure com es genera el few shot partition

        assert (batch_size > 0 and type(batch_size) == int)

        assert dataset in ["Midv", "Dogs", "Fungus", "Findit", "Banknotes"]


        ### PLACEHOLDERS  ###
        self._dataset = dataset
        self._type_split = type_split
        self._batch_size = batch_size
        self._kfold_split = kfold_split
        self._few_shot_split = few_shot_split
        self._normal_split = normal_split
        self._conditioned = conditioned

        self._datasets = [Midv, Dogs, Fungus, Findit, Banknotes]

        self._dt = list(filter(lambda dts : dts.__name__ == self._dataset, self._datasets))[0](self._conditioned)

        ### DOWNLOADING THE DATASET TO MAKE THE EXPERIMENTS ###
        logging.info("Searching for the dataset in the current working directory")
        flag, self._dataset_path = self.control_download()
        if flag is False:
            logging.warning("The dataset hasnt been found, starting to download")
            time.sleep(1)
            self._dt.dowload_dataset()
            logging.info("Dataset Download in {}".format(os.path.join(self._dt._uri.split("/")[-2], self._dt._uri.split("/")[-1])))

        new_df =  self._prepare_csv()
        if len(new_df)  == 0:
            logging.error("Some error occurred and the data couldnt been downloaded")
            sys.exit()

        if type_split == "kfold":
            train, val, test = self._kfold_partition(new_df)

        elif type_split == "shot":
            train, val, test = self._shot_partition(new_df)

        else:
            train, val, test = self._train_val_test_split(new_df)

        (partitions_control, size_partition) = (len(train.keys()) == len(val.keys()) == len(test.keys())),len(train.keys())
        assert partitions_control == True


        train_array, val_array, test_array =  np.array(train), np.array(val), np.array()


        ### defining the batch size ###
        self._batch = self.make_batches(train_array)





    def _kfold_partition(self, new_df) -> Tuple[List[Image], List[Image], List[Image]]:

        structure_train = []
        structure_val = []
        structure_test = []
        # Window to split the dataset in training/validation/testing set for the 10-fold
        if not os.path.exists('split_kfold/'):
            os.makedirs('split_kfold/')

        print('Splitting dataset into {}-folds partition with train/validation/test sets...'.format(self._kfold_split))
        k = len(new_df) / 10
        shuffled_df = shuffle(new_df)
        for iteration in range(self._kfold_split):
            b_low = int(iteration * k)
            b_high = int((1 + iteration) * k)
            df_test = shuffled_df[b_low:b_high]
            df_test.to_csv('split_kfold/test_split_' + self._dataset + '_it_' + str(iteration) + '.csv', index=False)
            df_val_train = shuffled_df.drop(shuffled_df.index[b_low:b_high])
            if iteration == 9:
                b_high = int(k)
                df_val = df_val_train[:b_high]
                df_val.to_csv('split_kfold/val_split_' + self._dataset + '_it_' + str(iteration) + '.csv', index=False)
                df_train = df_val_train.drop(df_val_train.index[:b_high])
                df_train.to_csv('split_kfold/train_split_' + self._dataset + '_it_' + str(iteration) + '.csv', index=False)
            else:
                df_val = df_val_train[b_low:b_high]
                df_val.to_csv('split_kfold/val_split_' + self._dataset + '_it_' + str(iteration) + '.csv', index=False)
                df_train = df_val_train.drop(df_val_train.index[b_low:b_high])
                df_train.to_csv('split_kfold/train_split_' + self._dataset + '_it_' + str(iteration) + '.csv', index=False)


            self.load_dataset(df_train, structure_train)
            self.load_dataset(df_val, structure_val)
            self.load_dataset(df_test, structure_test)


        print('Directory split_kfold created.')

        # Copy split_kfold directory to models' directory
        print('Copying split_kfold to Baseline directory...')
        shutil.copytree("split_kfold", "Baseline/split_kfold")
        print('Done.')

        print('Copying split_kfold to transfg directory...')
        shutil.copytree("split_kfold", "transfg/split_kfold")
        print('Done.')

        print('Copying split_kfold to arc_pytorch directory...')
        shutil.copytree("split_kfold", "arc_pytorch/split_kfold")
        print('Done.')

        return structure_train, structure_val, structure_test


    def _shot_partition(self,new_df):
        pass

    def _train_val_test_split(self, new_df) -> Tuple[List[Image], List[Image], List[Image]]:


        structure_train = []
        structure_val = []
        structure_test = []
        # Window to split the dataset in training/validation/testing set for the 10-fold
        if not os.path.exists('split_normal/'):
            os.makedirs('split_normal/')

        print('Splitting dataset into {}-normal partition with train/validation/test sets...'.format(self._normal_split))
        shuffled_df = shuffle(new_df)
        train_sec = int(len(shuffled_df) * self._normal_split[0])
        val_sec =  int(len(shuffled_df) * self._normal_split[1])
        test_sec = len(shuffled_df) - (train_sec+val_sec)
        ### TEST ####
        df_test = shuffled_df[:test_sec]
        df_test.to_csv('split_normal/test_split_' + self._dataset + '.csv', index=False)
        ### VALIDATION ###
        df_val_train = shuffled_df.drop(shuffled_df.index[:test_sec])
        df_val = df_val_train[:val_sec]
        df_val.to_csv('split_normal/val_split_' + self._dataset + '.csv', index=False)
        ### TRAIN ###
        df_train = df_val_train.drop(df_val_train.index[:val_sec])
        df_train.to_csv('split_normal/train_split_' + self._dataset + '.csv', index=False)

        self.load_dataset(df_train, structure_train)
        self.load_dataset(df_val, structure_val)
        self.load_dataset(df_test, structure_test)


        print('Directory split_normal created.')

        # Copy split_noraml directory to models' directory
        print('Copying split_normal to Baseline directory...')
        shutil.copytree("split_normal", "Baseline/split_normal")
        print('Done.')

        print('Copying split_normal to transfg directory...')
        shutil.copytree("split_normal", "transfg/split_normal")
        print('Done.')

        print('Copying split_normal to arc_pytorch directory...')
        shutil.copytree("split_normal", "arc_pytorch/split_normal")
        print('Done.')

        return structure_train, structure_val, structure_test


    def _prepare_csv(self):
        l_label = []
        l_img = []
        l_conditioned = []
        for file in glob.glob('{}/*/*'.format("dataset/"+self._dataset)):
            path = file.replace('\\', '/')
            path_decompose = path.split('/')
            if path_decompose[-1].startswith("index"):continue
            path_img = os.getcwd() + '/' + path
            label = list(filter(lambda x: x in ["reals", "fakes"], path_decompose))[0]
            l_label.append(label)
            l_img.append(path_img)
            if self._conditioned is True:
                clas_to_ap = self._dt._map_classes[label]
                l_conditioned.append(clas_to_ap.get(file, -1))
            else:
                l_conditioned.append(-1)

        columns = ["label_name", "label", "image_path", "class"]
        data = np.array([l_label, l_label, l_img, l_conditioned]).T
        new_df = pd.DataFrame(data=data, columns=columns)

        new_df['label'] = new_df['label'].map({'reals': 0,
                                               'fakes': 1},
                                              na_action=None)

        return new_df

    def control_download(self):
        result = []

        # Wlaking top-down from the Working directory
        for root, dir, files in os.walk(os.getcwd()):
            if self._dataset in dir:
                result.append(os.path.join(root, "/".join(dir)))

        return (True and len(result) != 0), result


    def read_img(path: str):

        img = np.array(imageio.imread(path))
        if img.shape[-1] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            return img


    def load_dataset(self, df, structure):

        information = zip(df['label'].to_numpy() + df['image_path'].to_numpy(), df['class'].to_numpy())
        for label, path, clas in information:

            structure.append(self._Data(label, self.read_img(path), clas))

    def make_batches(self, structure):
        """
        :param structure: list with the images or nested list with the images for every fold
        :return: a list with tuples in case basic split or a nested list with the fold len with tuples inside
         this tuples are the index of the batch size defined by the parameter self._batch_size
        """
        if len(structure) == 1:
            size = len(structure[0])
            nb_batch = int(np.ceil(size / float(self._batch_size)))
            res = [(i * self._batch_size, min(size, (i + 1) * self._batch_size)) for i in range(0, nb_batch)]

        else:
            res = []
            for fold in structure:
                size = len(fold)
                nb_batch = int(np.ceil(size / float(self._batch_size)))
                tmp = [(i * self._batch_size, min(size, (i + 1) * self._batch_size)) for i in range(0, nb_batch)]

                res.append(fold)
        return res


if __name__ == "__main__":


    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",default="Midv",nargs="?", required=True, type=str, choices=["Midv", "Dogs", "Fungus", "Findit", "Banknotes"],help="Define what kind of the different datasets do you want to download")
    parser.add_argument("--batch_size", default=1, type=int, nargs="?", help="Define the batch of the training set")
    parser.add_argument("-ts","--type_split",default="normal",nargs="?", choices=["normal", "kfold", "few_shot"], help="Diferent kind of split for train the models.")
    parser.add_argument("--conditioned", default=1 ,nargs="?",type=int, help="Flag to define if you want to train with the metaclasses inside the dataset thath downloaded ")

    opts, rem_args = parser.parse_known_args()

    conditioned = False if opts.conditioned == 0 else True

    print(conditioned)

    print(opts.type_split)
    if opts.type_split != "kfold" and opts.type_split != "few_shot":
        parser.add_argument("--normal_split", default=[0.8,0.1,0.1], nargs="+",help="define the behaviour of the split" )
        op = parser.parse_args()

        t = DataLoader(dataset=op.dataset,conditioned=conditioned,batch_size=op.batch_size,type_split=op.type_split, normal_split=op.normal_split)

    elif opts.type_split != "kfold" and opts.type_split != "normal":
        parser.add_argument("--few_shot_split", nargs="+",default=["random", 0.75, 0.25], help="define the behaviour of the split, the first value must be between [random, ranked] and the other values must be the proportion example(0.75,0.25)")
        op = parser.parse_args()
        print(op.few_shot_split)
        t = DataLoader(dataset=op.dataset,conditioned=conditioned,batch_size=op.batch_size,type_split=op.type_split, few_shot_split=op.few_shot_split)

    else:
        parser.add_argument("--kfold_split", default=10, type=int, nargs="?",help="define the number of folds")
        op = parser.parse_args()
        print(op.kfold_split)
        t = DataLoader(dataset=op.dataset,conditioned=conditioned,batch_size=op.batch_size,type_split=op.type_split, kfold_split=op.kfold_split)









