from ast import arg, parse
import numpy as np
import pandas as pd
from torch import split
from Datasets import *
from abc import ABC, abstractmethod
import time
import os
import os.path
import json
import cv2
import sys
import argparse
import glob
from sklearn.utils import shuffle
from distutils.dir_util import copy_tree
import shutil
import imageio
from typing import *
from PIL.Image import Image
from math import *
from pathlib import Path
import random

import logging
logging.basicConfig(format='%(asctime)s %(message)s',filename='runtime.log', level=logging.DEBUG)

class DataLoader(object):


    ##################################################################
    class _Data(object):

        __slots__ = ["_sample", "_gt", "_class", "_metaclass"]
        def __init__(self, sample, gt, clas, metaclass= None):
            self._sample = sample
            self._gt = gt
            self._class = clas
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




    def __init__(self, dataset:str="SIDTD",kind:str="templates", kind_models:str="transfg_img_net", download_static:bool= False,type_split:str = "cross", batch_size: int = 1,kfold_split:int=10, cross_split:list=[0.8,0.1,0.1]
, few_shot_split:Optional[str]=None, metaclasses:Optional[list] = None, conditioned:bool = True, unbalanced:bool=False, cropped:bool=True):

        """
            Input of the class:         dataset --> Define what kind of the different datasets do you want to download [SIDTD, Dogs, Fungus, Findit, Banknotes]
                                                    This datasets have been changed in order to the different approach we are working on
                                        Type_split --> Diferent kind of split for train the models. The diferents splits are [kfold, normal or few_shot]

                                        batch_size --> define the batch of the training set

                                        kfold_, normal, few_shot_ (split) --> define the behaviour of the split based on what kind of split did you put

                                        conditioned --> flag to define if you want to train with the metaclasses inside the dataset thath downloaded 
        
        """
        #["all_trained_models", "effnet", "resnet", "vit", "transfg", "arc", "transfg_img_net","no"]
        models = {
            "effnet":"efficientnet-b3_trained_models",
            "resnet": "resnet50_trained_models",
            "vit": "vit_large_patch16_224_trained_models",
            "transfg": "trans_fg_trained_models",
            "transfg_img_net": "transfg_pretrained",
            "arc": "coatten_fcn_model_trained_models",
            "no": "no",
            "all_trained_models":"all_trained_models"
        }
        
        static_csv = {
            ("cross", False): "split_normal",
            ("kfold", False): "split_kfold",
            ("few_shot", False):"split_shot",
            ("cross", True) : "cross_val_unbalanced",
            ("kfold", True) : "split_kfold_unbalanced",
            ("few_shot", True): "split_shot_unbalanced"
        }
        
        

        ### ASSERTS AND ERROR CONTROL ###
        cross_split = list(float(x) for x in cross_split)
        if type_split == "cross": assert (type(cross_split) == list and int(np.sum(cross_split)) == 1)
        elif type_split == "kfold": assert (kfold_split > 0 and type(kfold_split) == int)
        elif type_split == "shot": assert (isinstance(cross_split, list) and len(cross_split) == 3 and isinstance(cross_split[0], str)
                                           and isinstance(cross_split[1], float) and isinstance(cross_split[2], float)), "Some parameter have been badly declared"

        assert (batch_size > 0 and type(batch_size) == int)

        assert dataset in ["SIDTD", "Dogs", "Fungus", "Findit", "Banknotes"]

        ### PLACEHOLDERS  ###
        self._dataset = dataset
        self._dataset_type = kind
        self._model_folder_name = models[kind_models]
        self._model_name = kind_models
        self._type_split = type_split
        self._batch_size = batch_size
        self._kfold_split = kfold_split
        self._few_shot_split = few_shot_split
        self._normal_split = cross_split
        self._conditioned = conditioned
        self._unbalanced = unbalanced
        self._cropped = cropped
        
        
        current_path = os.getcwd()
        self._save_dir = os.path.join(current_path,"code_examples")
        
        self._datasets = [SIDTD, Dogs, Fungus, Findit, Banknotes]

        self._dt = list(filter(lambda dts : dts.__name__ == self._dataset, self._datasets))[0](conditioned=conditioned)

        ### DOWNLOADING THE DATASET TO MAKE THE EXPERIMENTS ###
        
        logging.info("Searching for the dataset in the current working directory")
        if kind != "no":

            flag, self._dataset_path = self.control_download(to_search=dataset,search="dataset", root="datasets")

            if flag is False:
                logging.warning("The dataset hasnt been found, starting to download")
                time.sleep(1)
                if self._dataset == "SIDTD":
                    self._dt.download_dataset(type_download=kind, cropped=cropped)
                else:
                    self._dt.download_dataset()
                
                logging.info("Dataset Download in {}".format(os.path.join(self._dt._uri.split("/")[-2], "code_examples")))
            else:
                kinds = os.listdir(self._dataset_path)
                if kind not in kinds:
                    self._dt.download_dataset(type_download=kind, cropped=cropped)

                else:   
                    empty = (len(glob.glob(os.path.join(self._dataset_path, kind, "Images","fakes", "*"))) == 0) and (len(glob.glob(os.path.join(self._dataset_path, kind, "Images", "reals", "*"))) == 0)
                    if empty:
                        os.rmdir(os.path.join(self._dataset_path, kind))
                        self._dt.download_dataset(type_download=kind, cropped=cropped)
                    else:
                        logging.info(f"Dataset found in {self._dataset_path}, check if it is empty")
    
        else:logging.info("No dataset is set to download, not searching for")


        #### MODELS
            
        if kind_models != "no":

            if kind_models == "all_trained_models":

                for model in list(models.keys())[:6]:
                    model_folder_name = models[model]

                    logging.warning("Searching for the model in the current working directory")
                    # search:str="dataset", root:str="datasets"
                    flag_1, _ = self.control_download(to_search=model_folder_name, search="models", root="code_examples")   
                    flag_2, _ = self.control_download(to_search=model_folder_name, search="models", root="models")

                    if (flag_1 | flag_2) is False:
                        logging.warning("The model hasnt been found, starting to download")
                        self._dt.download_models(unbalanced = self._unbalanced, cropped = self._cropped,type_models=model)
                        
                        logging.info("Model Download in {}".format(os.path.join(self._dt._uri.split("/")[-1], "code_examples", )))
                    
                    else:logging.info(f"Model found in code examples\' directory or in models\' directory")

            
            else:

                logging.warning("Searching for the model in the current working directory")
                # search:str="dataset", root:str="datasets"
                flag_1, _ = self.control_download(to_search=self._model_folder_name, search="models", root="code_examples")   
                flag_2, _ = self.control_download(to_search=self._model_folder_name, search="models", root="models")

                if (flag_1 | flag_2) is False:
                    logging.warning("The model hasnt been found, starting to download")
                    self._dt.download_models(unbalanced = self._unbalanced,type_models=kind_models)
                    
                    logging.info("Model Download in {}".format(os.path.join(self._dt._uri.split("/")[-1], "code_examples", )))
                
                else:logging.info(f"Model found in code examples\' directory or in models\' directory")

        
        else:
            logging.info("No model is set, not searching for")

        ###### CSV DOWNLOAD PART
            
        if download_static == True:

            if not os.path.exists(self._dt.abs_path_code_ex_csv):
                os.makedirs(self._dt.abs_path_code_ex_csv)
            to_search = static_csv[tuple([type_split, unbalanced])]
            flag_csv = self.control_download(to_search=to_search, search="static", root="code_examples")

            if flag_csv is False:
                logging.warning("Static csv hasnt been downloaded, starting to download")
                time.sleep(1)
                self._dt.download_static_csv(partition_kind=type_split, unbalanced=unbalanced, cropped=cropped)

                logging.info("CSV Download in {}".format(os.path.join(os.getcwd(), "code_examples", "static")))
            else:
                logging.info("Static CSV had been downloaded, avoiding to download them")
        
        else:
            if kind == "no":
                flag, self._dataset_path = self.control_download(to_search=dataset, search="dataset", root="datasets")

                if flag == False:
                    logging.warning("No dataset to make the partitions, pls download it before to create your own partitions or download our partitions!!")
                    sys.exit()

            else:
                logging.info(f"Preparing partitions for the {type_split} partition behaviour")
                new_df =  self._prepare_csv(kind=kind,cropped=cropped) if unbalanced == False else self.get_unbalance_partition()

                ######### UNCOMMENT THIS LINE WHEN CODE IS FINISHED #########
                #self.set_static_path()
                ######### UNCOMMENT THIS LINE WHEN CODE IS FINISHED #########

                if len(new_df)  == 0:
                    logging.error("Some error occurred and the data couldnt been downloaded")
                    sys.exit()

                if type_split == "kfold":
                    self._kfold_partition(new_df, kind=kind)

                elif type_split == "few_shot":
                    if few_shot_split[0] != "random":
                        raise NotImplementedError
                    
                    self._shot_partition(new_df, proportion=few_shot_split[1:],metaclasses=metaclasses, kind=kind)

                else:
                    self._train_val_test_split(new_df, kind=kind)
                




                

    def change_path(self,path:str='/home/users/SIDTD/code_examples/split_normal/test_split_SIDTD.csv'):
        current_path = os.getcwd()
        df = pd.read_csv(path)
        df['image_path'] = current_path + df['image_path']
        df.to_csv(path, index=False)

    def set_static_path(self):
        current_path = os.getcwd()
        for d_set in ['train','val','test']:
            path_save_csv = current_path + '/code_examples/static/split_normal/' + d_set + '_split_SIDTD.csv'
            self.change_path(path_save_csv)
            for j in range(10):
                path_save_csv = current_path + '/code_examples/static/split_kfold/' + d_set + '_split_SIDTD_it_' + str(j) + '.csv'
                self.change_path(path_save_csv)
                

    def _kfold_partition(self, new_df, kind:str= "templates") -> Tuple[List[Image], List[Image], List[Image]]:

        # Window to split the dataset in training/validation/testing set for the 10-fold
        split_kfold_dir = os.path.join(self._save_dir,'split_kfold', kind ,self._dataset) if self._unbalanced == False else os.path.join(self._save_dir,'split_kfold_unbalanced', kind,self._dataset)
        if not os.path.exists(split_kfold_dir):
            os.makedirs(split_kfold_dir)

        print('Splitting dataset into {}-folds partition with train/validation/test sets...'.format(self._kfold_split))
        k = len(new_df) / self._kfold_split
        shuffled_df = shuffle(new_df)
        for iteration in range(self._kfold_split):
            b_low = int(iteration * k)
            b_high = int((1 + iteration) * k)
            df_test = shuffled_df[b_low:b_high]
            df_test.to_csv(split_kfold_dir+'/test_split_' + self._dataset + '_it_' + str(iteration) + '.csv', index=False)
            df_val_train = shuffled_df.drop(shuffled_df.index[b_low:b_high])
            if iteration == 9:
                b_high = int(k)
                df_val = df_val_train[:b_high]
                df_val.to_csv(split_kfold_dir+'/val_split_' + self._dataset + '_it_' + str(iteration) + '.csv', index=False)
                df_train = df_val_train.drop(df_val_train.index[:b_high])
                df_train.to_csv(split_kfold_dir+'/train_split_' + self._dataset + '_it_' + str(iteration) + '.csv', index=False)
            else:
                df_val = df_val_train[b_low:b_high]
                df_val.to_csv(split_kfold_dir+'/val_split_' + self._dataset + '_it_' + str(iteration) + '.csv', index=False)
                df_train = df_val_train.drop(df_val_train.index[b_low:b_high])
                df_train.to_csv(split_kfold_dir+'/train_split_' + self._dataset + '_it_' + str(iteration) + '.csv', index=False)



        print('Directory split_kfold created.')


    def _shot_partition(self,new_df, proportion:list = [0.6,0.4], metaclasses:list = ["id", "passport"], kind:str= "templates"):


        split_dir = os.path.join(self._save_dir,'split_shot', kind, self._dataset) if self._unbalanced == False else os.path.join(self._save_dir,'split_shot_unbalanced', kind, self._dataset)
        metatrain_prop, metatest_prop = float(proportion[0]), float(proportion[1])
         
        ngroups = len(metaclasses)
        assert ngroups >0
        if ngroups > 1:
            new_column = [" " for i in range(new_df.shape[0])]
            list_of_images = new_df.loc[:, "image_path"] #list with the paths of the images and his names to get the metaclasses
            for group in metaclasses:
                for idx,path_image in enumerate(list_of_images):
                    name = path_image.split("/")[-1]
                    if group in name:
                        new_column[idx] = group

            new_df["metaclass"] = new_column
            grouped_pandas = new_df.groupby(["class", "metaclass"])

        else:
            grouped_pandas = new_df.groupby(["class"])

        train = pd.DataFrame()
        test = pd.DataFrame()
        
        groups = list(grouped_pandas.groups.keys())
      
        train_groups = random.sample(groups, round(metatrain_prop * len(groups)))
        test_groups = list(set(groups)- set(train_groups))

        for name, group in grouped_pandas:

            if bool(set([name])&set(train_groups)) == True:             
                train = pd.concat([train,group])
                
            elif bool(set([name])&set(test_groups)) == True:
                test = pd.concat([test,group])
            
        df_train = train.reset_index().drop("index", axis=1)
        df_test = test.reset_index().drop("index", axis=1)
        
        if not os.path.exists(split_dir):
            os.makedirs(split_dir)

        
        print('Splitting dataset into {}-shot partition with meta_train/meta_test sets...'.format(self._few_shot_split))
        df_train.to_csv(split_dir+'/train_split_' + self._dataset + '.csv', index=False)
        df_test.to_csv(split_dir+'/test_split_' + self._dataset + '.csv', index=False)


        
        print('Directory split_shot created.')


    def _ranking_shot_partition(self, new_df, proportion, metaclasses:list = ["dni", "passport"]):
        raise NotImplementedError

    def _train_val_test_split(self, new_df, kind:str="templates") -> Tuple[List[Image], List[Image], List[Image]]:

        # Window to split the dataset in training/validation/testing set for the 10-fold
        split_dir = os.path.join(self._save_dir,'cross_val', kind, self._dataset) if self._unbalanced == False else os.path.join(self._save_dir,'cross_val_unbalanced', kind, self._dataset)

        if not os.path.exists(split_dir):
            os.makedirs(split_dir)

        print('Splitting dataset into {}-standard cross val partition with train/validation/test sets...'.format(self._normal_split))
        shuffled_df = shuffle(new_df)
        train_sec = int(len(shuffled_df) * self._normal_split[0])
        val_sec =  int(len(shuffled_df) * self._normal_split[1])
        test_sec = len(shuffled_df) - (train_sec+val_sec)
        ### TEST ####
        df_test = shuffled_df[:test_sec]
        df_test.to_csv(split_dir+'/test_split_' + self._dataset + '.csv', index=False)
        ### VALIDATION ###
        df_val_train = shuffled_df.drop(shuffled_df.index[:test_sec])
        df_val = df_val_train[:val_sec]
        df_val.to_csv(split_dir+'/val_split_' + self._dataset + '.csv', index=False)
        ### TRAIN ###
        df_train = df_val_train.drop(df_val_train.index[:val_sec])
        df_train.to_csv(split_dir+'/train_split_' + self._dataset + '.csv', index=False)

        print(f'Directory cross_val {kind} created.')

    ## probably debug it
    def get_unbalance_partition(self, kin_data:str="images", proportion:list=[0.8, 0.2], path_to_conversion: Optional[Path] = None) -> None:

        data_path =  os.path.join(os.getcwd(), "datasets",self._dataset, kin_data, "Images") 
        all_info_fakes = set(glob.glob(data_path+"/fakes/*"))
        all_info_reals = set(glob.glob(data_path+"/reals/*"))

        defined_number_of_images = 191
        new_data = {}

        static_info = None

        assert (sum(proportion) == 1) and (proportion[-1] <= 0.2) and len(proportion)==2, "The proportion is out of bounds of the proper behaviour"

        if path_to_conversion is not None:
            c_file = pd.read_csv(os.getcwd()+"/Fake_MIDV2020_videos_2_templates.csv")
            static_info = set(c_file["Fake_Document_ID"].apply(lambda x: os.path.join(data_path,"fakes", x)).values)
            classes = c_file["Nationality"].values

        _number_fakes = round((proportion[-1] * defined_number_of_images)/0.2)
        _number_reals = len(all_info_reals)

        ## Updating real part
        new_data["image_path"] = random.choices(list(all_info_reals), k=_number_reals)
        new_data["label"] = np.full(_number_reals, 0).tolist()
        new_data["class"] = self._dt.map_metaclass(new_data["image_path"])

        #updating fake part
        if isinstance(static_info, str):
            new_data["image_path"].extend(list(all_info_fakes & static_info))
            new_data["label"].extend((np.full_like(static_info, 1).tolist()))
            new_data["class"].extend(classes)

        else:
            tmp_data = random.choices(list(all_info_fakes), k=_number_fakes)
            new_data["image_path"].extend(tmp_data)
            new_data["label"].extend(np.full(_number_fakes, 1))
            new_data["class"].extend(self._dt.map_metaclass(tmp_data))


        new_data = pd.DataFrame(new_data)
        new_data["label_name"] = new_data["label"].map({0: "reals",1: 'fakes'},na_action=None)



        return new_data.sample(frac=1).reset_index(drop=True)
        
        


    def _prepare_csv(self, kind, cropped):
        l_label = []
        l_img = []
        l_conditioned = []
        if cropped:
            kind = 'cropped'
        for file in glob.glob('{}/*/*'.format(os.path.join(os.getcwd(), "datasets",self._dataset, kind, "Images"))):
            path = file.replace('\\', '/')
            path_decompose = path.split('/')
            if path_decompose[-1].startswith("index"):continue
            label = list(filter(lambda x: x in ["reals", "fakes"], path_decompose))[0]
            l_label.append(label)
            l_img.append(file)
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

    
    def control_download(self, to_search:str, search:str="dataset", root:str="datasets"):
        # Wlaking top-down from the Working directory
        for rt, dir, files in os.walk(os.path.join(os.getcwd(), root)):            
            if (to_search in dir):
                return True, os.path.join(rt, "/".join(dir))
        
        return False, []


    def load_dataset(self, path: str) -> list:
        
        structure = []
        df = pd.read_csv(path, delimiter=",")
        information = zip(df['label'].to_numpy(), df['image_path'].to_numpy(), df['class'].to_numpy())
        for label, path, clas in information:
            structure.append(self._Data(label, self.read_img(path), clas))
        
        return structure
    

    def make_batches(self, structure: List[Type[_Data]]) -> List[List[range]]:
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

    @staticmethod
    def read_img(path: str) -> Image:

        img = np.array(imageio.imread(path))
        if img.shape[-1] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            return img

if __name__ == "__main__":

    ## TODO add the flag to get templates videos or clips
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset",default="SIDTD",nargs="?", type=str, choices=["SIDTD", "Dogs", "Fungus", "Findit", "Banknotes"],help="Define what kind of the different datasets do you want to download")
    parser.add_argument("--kind",default="templates",nargs="?", type=str, choices=["templates", "clips", "videos","no"],help="Define what kind of the info from the benchmark do you want to download to use. If no is selected, then the dataset will not be download.")
    parser.add_argument("--download_static", action="store_true", help="set to 1 if you want to download the static csv for reproducibility of the models" )
    parser.add_argument("--kind_models",default="transfg_img_net",nargs="?", type=str, choices=["all_trained_models", "effnet", "resnet", "vit", "transfg", "arc", "transfg_img_net","no"],help="Define what kind of the trained model from the benchmark you want to download in order to reproduce results. Choose transfg_img_net, if you want to train the trans fg model (default mode). Choose no if you do not want to download any model.")
    parser.add_argument("--batch_size", default=1, type=int, nargs="?", help="Define the batch of the training set")
    parser.add_argument("-ts","--type_split",default="cross",nargs="?", choices=["cross", "kfold", "few_shot"], help="Diferent kind of split to train the models.")
    parser.add_argument("--conditioned", default=1 ,nargs="?",type=int, help="Flag to define if you want to train with the metaclasses inside the dataset thath downloaded ")
    parser.add_argument("--unbalanced", action="store_true", help="flag to prepare the unbalance partition")
    parser.add_argument("-c","--cropped", action="store_true", help="flag to use the cropped version of clips.")
    opts, rem_args = parser.parse_known_args()

    conditioned = False if opts.conditioned == 0 else True

    if opts.type_split != "kfold" and opts.type_split != "few_shot":
        parser.add_argument("--cross_split", default=[0.8,0.1,0.1], nargs="+",help="define the behaviour of the split" )
        op = parser.parse_args()

        t = DataLoader(dataset=op.dataset,kind=op.kind, download_static=op.download_static, kind_models=op.kind_models, conditioned=conditioned,batch_size=op.batch_size,type_split=op.type_split, cross_split=op.cross_split, unbalanced=op.unbalanced, cropped=op.cropped)

    elif opts.type_split != "kfold" and opts.type_split != "cross":
        parser.add_argument("--few_shot_split", nargs="+",default=["random", 0.75, 0.25], help="define the behaviour of the split, the first value must be between [random, ranked] and the other values must be the proportion example(0.75,0.25)")
        parser.add_argument("--metaclasses", nargs="+",default=["id", "passport"], help="define the secopnd level to group the metatrain and the metatest")

        op = parser.parse_args()
        print(op.few_shot_split)
        t = DataLoader(dataset=op.dataset,kind=op.kind, download_static=op.download_static, kind_models=op.kind_models, conditioned=conditioned,batch_size=op.batch_size,type_split=op.type_split, few_shot_split=op.few_shot_split, metaclasses=op.metaclasses,unbalanced=op.unbalanced, cropped=op.cropped)

    else:
        parser.add_argument("--kfold_split", default=10, type=int, nargs="?",help="define the number of folds")
        op = parser.parse_args()
        t = DataLoader(dataset=op.dataset,kind=op.kind, download_static=op.download_static, kind_models=op.kind_models, conditioned=conditioned,batch_size=op.batch_size,type_split=op.type_split, kfold_split=op.kfold_split, unbalanced=op.unbalanced, cropped=op.cropped)