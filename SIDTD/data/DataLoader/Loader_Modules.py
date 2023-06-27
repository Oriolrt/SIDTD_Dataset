from SIDTD.data.DataLoader.Datasets  import *
from SIDTD.utils.util import read_img

from typing import *
from PIL.Image import Image
from sklearn.utils import shuffle

import numpy as np
import pandas as pd
import time
import os
import os.path
import sys
import argparse
import glob
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




    def __init__(self, dataset:str="SIDTD",kind:str="templates", download_static:bool= False,type_split:str = "hold_out", kfold_split:int=10, hold_out_split:list=[0.8,0.1,0.1]
, few_shot_split:Optional[str]=None, metaclasses:Optional[list] = None, unbalanced:bool= False, cropped:bool= False):

        """
        Initialize the class.

        Parameters:
        - dataset (str): Define the type of dataset you want to download [SIDTD, Dogs, Fungus, Findit, Banknotes].
                         These datasets have been modified to fit our working approach.
        - kind (str): Type of split for training the models. Available options are: [kfold, hold_out, or few_shot].
        - download_static (bool): Indicates whether to download the static files.
        - type_split (str): Type of split for training the models. Available options: "kfold", "hold_out", or "few_shot".
        - kfold_split (int): Number of folds in case of using k-fold.
        - hold_out_split (list): List that defines the split ratio in case of using hold-out.
                                 The list should sum up to 1 and contain 3 elements [train_ratio, val_ratio, test_ratio].
        - few_shot_split (Optional[str]): Few-shot split option. Currently, only "random" is supported.
        - metaclasses (Optional[list]): List of metaclasses for the few-shot partition.
        - unbalanced (bool): Indicates whether the dataset is unbalanced.
        - cropped (bool): Indicates whether the clips are cropped.
        """

        if kind == "clips_cropped": kind = "cropped"

        ### ASSERTS AND ERROR CONTROL ###
        hold_out_split = list(float(x) for x in hold_out_split)
        if type_split == "hold_out": assert (type(hold_out_split) == list and int(np.sum(hold_out_split)) == 1)
        elif type_split == "kfold": assert (kfold_split > 0 and type(kfold_split) == int)
        elif type_split == "shot": assert (isinstance(hold_out_split, list) and len(hold_out_split) == 3 and isinstance(hold_out_split[0], str)
                                           and isinstance(hold_out_split[1], float) and isinstance(hold_out_split[2], float)), "Some parameter have been badly declared"

        assert dataset in ["SIDTD", "Dogs", "Fungus", "Findit", "Banknotes"]

        ### PLACEHOLDERS  ###
        self._dataset = dataset
        self._dataset_type = kind
        self._unbalanced = unbalanced
        self._cropped = cropped
        self._type_split = type_split
        self._kfold_split = kfold_split
        self._few_shot_split = few_shot_split
        self._hold_out_split = hold_out_split

        
        current_path = os.getcwd()
        self._save_dir = os.path.join(current_path,"models","explore")
        
        self._datasets = [SIDTD, Dogs, Fungus, Findit, Banknotes]

        self._dt = list(filter(lambda dts : dts.__name__ == self._dataset, self._datasets))[0]()
        self._map_classes = self._dt.map_classes(type_data=kind)

        ### DOWNLOADING THE DATASET TO MAKE THE EXPERIMENTS ###
        
        logging.info("Searching for the dataset in the current working directory")

        flag, self._dataset_path = self.control_download(to_search=dataset, root="datasets")

        if flag is False:
            logging.warning("The dataset hasnt been found, starting to download")
            time.sleep(1)
            if self._dataset == "SIDTD":
                self._dt.download_dataset(type_download=kind)
            else:
                self._dt.download_dataset()

            logging.info("Dataset Download in {}".format(os.path.join(self._dt._uri.split("/")[-2], "explore")))
        else:
            kinds = os.listdir(self._dataset_path)
            if kind not in kinds:
                self._dt.download_dataset(type_download=kind)

            else:
                empty = (len(glob.glob(os.path.join(self._dataset_path, kind, "Images","fakes", "*"))) == 0) and (len(glob.glob(os.path.join(self._dataset_path, kind, "Images", "reals", "*"))) == 0)
                if empty:
                    os.rmdir(os.path.join(self._dataset_path, kind))
                    self._dt.download_dataset(type_download=kind)
                else:
                    logging.info(f"Dataset found in {self._dataset_path}, check if it is empty")


        ###### CSV DOWNLOAD PART
            
        if download_static == True:
            time.sleep(1)
            self._dt.download_static_csv(partition_kind=type_split, type_download=kind)
            logging.info("CSV Download in {}".format(os.path.join(os.getcwd(), "data", "static")))

        else:
            logging.info(f"Preparing partitions for the {type_split} partition behaviour")
            new_df =  self._prepare_csv(kind=kind)

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
                self._hold_out_partition(new_df, kind=kind)




    def change_path(self,path:str='/home/users/SIDTD/explore/split_normal/test_split_SIDTD.csv'):
        current_path = os.getcwd()
        df = pd.read_csv(path)
        df['image_path'] = current_path + df['image_path']
        df.to_csv(path, index=False)


    def set_static_path(self):
        current_path = os.getcwd()
        for d_set in ['train','val','test']:
            path_save_csv = current_path + '/data/explore/static/split_normal/' + d_set + '_split_SIDTD.csv'
            self.change_path(path_save_csv)
            for j in range(10):
                path_save_csv = current_path + '/data/explore/static/split_kfold/' + d_set + '_split_SIDTD_it_' + str(j) + '.csv'
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

    def _hold_out_partition(self, new_df, kind:str="templates") -> Tuple[List[Image], List[Image], List[Image]]:

        # Window to split the dataset in training/validation/testing set for the 10-fold
        split_dir = os.path.join(self._save_dir,'hold_out', kind, self._dataset) if self._unbalanced == False else os.path.join(self._save_dir,'hold_out_unbalanced', kind, self._dataset)

        if not os.path.exists(split_dir):
            os.makedirs(split_dir)

        print('Splitting dataset into {}-standard hold out val partition with train/validation/test sets...'.format(self._hold_out_split))
        shuffled_df = shuffle(new_df)
        train_sec = int(len(shuffled_df) * self._hold_out_split[0])
        val_sec =  int(len(shuffled_df) * self._hold_out_split[1])
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

        print(f'Directory hold out {kind} created.')



    def _prepare_csv(self, kind):
        l_label = []
        l_img = []
        l_conditioned = []
        if kind=='clips_cropped':
            kind = 'cropped'
        for file in glob.glob('{}/*/*'.format(os.path.join(os.getcwd(), "datasets",self._dataset, kind, "Images"))):
            path = file.replace('\\', '/')
            path_decompose = path.split('/')
            if path_decompose[-1].startswith("index"):continue
            label = list(filter(lambda x: x in ["reals", "fakes"], path_decompose))[0]
            l_label.append(label)
            l_img.append(file)
            clas_to_ap = self._map_classes[label]
            l_conditioned.append(clas_to_ap.get(file, -1))


        columns = ["label_name", "label", "image_path", "class"]
        data = np.array([l_label, l_label, l_img, l_conditioned]).T
        new_df = pd.DataFrame(data=data, columns=columns)

        new_df['label'] = new_df['label'].map({'reals': 0,
                                               'fakes': 1},
                                              na_action=None)

        return new_df

    
    def control_download(self, to_search:str, root:str="datasets"):
        # Wlaking top-down from the Working directory
        for rt, dir, files in os.walk(os.path.join(os.getcwd(), root)):            
            if (to_search in dir):
                print(os.path.join(rt, "/".join(dir)))
                return True, os.path.join(rt, "/".join(dir))
        
        return False, []


    def load_dataset(self, path: str) -> list:
        
        structure = []
        df = pd.read_csv(path, delimiter=",")
        information = zip(df['label'].to_numpy(), df['image_path'].to_numpy(), df['class'].to_numpy())
        for label, path, clas in information:
            structure.append(self._Data(label, read_img(path), clas))
        
        return structure
    

    @staticmethod
    def make_batches(structure: List[Type[_Data]], batch_size:int) -> List[List[range]]:
        """
        :param structure: list with the images or nested list with the images for every fold
        :return: a list with tuples in case basic split or a nested list with the fold len with tuples inside
         this tuples are the index of the batch size defined by the parameter batch_size
        """
        if len(structure) == 1:
            size = len(structure[0])
            nb_batch = int(np.ceil(size / float(batch_size)))
            res = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]

        else:
            res = []
            for fold in structure:
                size = len(fold)
                nb_batch = int(np.ceil(size / float(batch_size)))
                tmp = [(i * batch_size, min(size, (i + 1) * batch_size)) for i in range(0, nb_batch)]

                res.append(tmp)
        
        return res

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    # Define arguments
    parser.add_argument("--dataset", default="SIDTD", type=str,
                        choices=["SIDTD", "Dogs", "Fungus", "Findit", "Banknotes"],
                        help="Specify the dataset to download")
    parser.add_argument("--kind", default="templates", type=str, choices=["templates", "clips", "clips_cropped", "videos", "no"],
                        help="Specify the type of benchmark data to download. If 'no' is selected, the dataset will not be downloaded.")
    parser.add_argument("--download_static", action="store_true",
                        help="Set this flag to download the static CSV for reproducibility of the models")
    parser.add_argument("-ts","--type_split", default="hold_out", type=str, choices=["hold_out", "kfold", "few_shot"],
                        help="Specify the type of data split to train the models")
    parser.add_argument("--unbalanced", action="store_true", help="Flag to prepare the unbalanced partition")
    parser.add_argument("-c", "--cropped", action="store_true", help="Flag to use the cropped version of clips")

    # Parse arguments
    args, _ = parser.parse_known_args()

    # Handle different split types
    if args.type_split == "hold_out":
        parser.add_argument("--hold_out_split", default=[0.8, 0.1, 0.1], nargs="+",
                            help="Define the split ratios for hold-out validation")
        options = parser.parse_args()

        t = DataLoader(
            dataset=options.dataset,
            kind=options.kind,
            download_static=options.download_static,
            type_split=options.type_split,
            hold_out_split=options.hold_out_split,
            unbalanced=args.unbalanced,
            cropped=args.cropped
        )

    elif args.type_split == "few_shot":
        parser.add_argument("--few_shot_split", nargs="+", default=["random", 0.75, 0.25],
                            help="Define the split behavior for few-shot learning. The first value must be either 'random' or 'ranked', and the other values must specify the proportions.")
        parser.add_argument("--metaclasses", nargs="+", default=["id", "passport"],
                            help="Define the second level to group the metatrain and the metatest")

        options = parser.parse_args()
        print(options.few_shot_split)
        t = DataLoader(
            dataset=options.dataset,
            kind=options.kind,
            download_static=options.download_static,
            type_split=options.type_split,
            few_shot_split=options.few_shot_split,
            metaclasses=options.metaclasses,
            unbalanced=args.unbalanced,
            cropped=args.cropped
        )

    elif args.type_split == "kfold":
        parser.add_argument("--kfold_split", default=10, type=int, nargs="?",
                            help="Specify the number of folds for k-fold validation")
        options = parser.parse_args()

        t = DataLoader(
            dataset=options.dataset,
            kind=options.kind,
            download_static=options.download_static,
            type_split=options.type_split,
            kfold_split=options.kfold_split,
            unbalanced=args.unbalanced,
            cropped=args.cropped
        )
