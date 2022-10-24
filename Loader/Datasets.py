from matplotlib.pyplot import get
import numpy as np 
import math
from abc import ABC,abstractmethod
import time as t
import wget
import os
import subprocess
import glob
from collections import Counter
import imageio.v2 as imageio
import json
import cv2
import sys



class Dataset(ABC):

    def __init__(self,download_original:bool=True,conditioned:bool=False,uri:str = "localhost/Benchmarking/DataLoader/dataset") -> None:

        self._uri = uri
        self._conditioned = conditioned
        self._download_original = download_original



    @abstractmethod
    def num_fake_classes(self):
        raise NotImplementedError


    @abstractmethod
    def num_real_classes(self):
        raise NotImplementedError

    @abstractmethod
    def map_classes(self):
        raise NotImplementedError

    @abstractmethod
    def number_of_real_sampling(self):
        raise NotImplementedError
    @abstractmethod
    def number_of_fake_sampling(self):
        raise NotImplementedError

    @abstractmethod
    def download_dataset(self, output_directory:str = None):
        raise NotImplementedError


        
        

class Dogs(Dataset):
    pass

class Fungus(Dataset):
    pass

class Findit(Dataset):
    pass

class Banknotes(Dataset):
    pass


class Midv(Dataset):

    def __init__(self,conditioned:bool=True,download_original:bool =True ,uri: str = "http://0.0.0.0:8000/Benchmarking/DataLoader/dataset/midv") -> None:
        super().__init__(conditioned=conditioned, download_original=download_original,uri=uri)
        self._map_classes = self.map_classes() if conditioned is True else None
        self._original_abs_path = "MIDV2020/templates"
        self._original_imgs_path = os.path.join(self._original_abs_path, "images")
        self._original_ann_path  = os.path.join(self._original_abs_path, "annotations")
        
    def download_dataset(self):
        #files = os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self._abs_path,self._uri))

        if self._download_original:
            self._abs_path = "../datasets/"+self._uri.split("/")[-1]
            self._img_abs_path = os.path.join(self._abs_path, "Images", "Reals")
            self._ann_abs_path = os.path.join(self._abs_path, "Annotations", "Reals")
            self.create_structure()
           
    def create_and_map_classes_imgs(self):
        map_class = {
            
        }
        for image in os.listdir(self._img_abs_path):
            if image.endswith("html"):continue
            class_image = image.split("_")[0]
            original_class_path = os.path.join("..","datasets",self._original_imgs_path, class_image)
            if os.path.exists(original_class_path):
                if map_class.get(class_image) is not None:
                    map_class[class_image].append(os.path.join(self._img_abs_path, image))
                else:
                    map_class[class_image] = [os.path.join(self._img_abs_path, image)]

            else:
                map_class[class_image] = [os.path.join(self._img_abs_path, image)]
                os.makedirs(original_class_path)
  
        return map_class                
            
    
    def create_and_map_classes_annotations(self):
        map_annotation = {
            
        }
        for annotation in os.listdir(self._ann_abs_path):
            if annotation.endswith("html"):continue
            class_ann = annotation.split("_")[0]
            original_class_path = os.path.join("..","datasets",self._original_ann_path, class_ann)

            if os.path.exists(original_class_path):
                map_annotation[class_ann] = os.path.join(self._ann_abs_path,annotation)
                continue
            else:
                map_annotation[class_ann] = os.path.join(self._ann_abs_path,annotation)
                os.makedirs(original_class_path)

                                
        return map_annotation    
                       
           
            
    def create_structure(self):
        map_imgs = self.create_and_map_classes_imgs()
        map_annotations = self.create_and_map_classes_annotations()
        classes = list(map_imgs.keys())
        

        for clas, img_set in map_imgs.items():
            template = self.read_json(map_annotations[clas]) #dict
            path_ann_save = os.path.join("..", "datasets", self._original_ann_path,clas)
            self.write_json(template, path_ann_save, clas)
            
            path_img_save = os.path.join("..", "datasets", self._original_imgs_path, clas)
            for img in img_set:
                name_img = img.split("/")[-1].split("_")[-1] #get the image numbe (82.jpg example)
                im = self.read_img(img)  
                imageio.imwrite(os.path.join(path_img_save,name_img), im)
                

    def map_classes(self):
        classes = {"reals":{}, "fakes":{}}
        fakes = [(file, "fakes") for file in glob.glob("dataset/SIDTD/Images/fakes/*.jpg")]
        reals = [(file, "reals") for file in glob.glob("dataset/SIDTD/Images/reals/*.jpg")]
        for file in (fakes+reals):
            section = classes[file[1]]
            clas = file[0].split("_")[0].split("/")[-1]
            if clas.startswith("index"):continue

            section[file[0]] =  clas if self._conditioned else -1

        return classes

    def number_of_real_sampling(self):
        return dict(Counter(self._map_classes["reals"].values()))

    def number_of_fake_sampling(self):
        return dict(Counter(self._map_classes["fakes"].values()))

    def num_fake_classes(self):
        return len(self.number_of_fake_sampling().keys())
    def num_real_classes(self):
        return len(self.number_of_real_sampling().keys())
    
    @staticmethod
    def read_json(path: str):
        with open(path) as f:
            return json.load(f)    
    @staticmethod      
    def read_img(path: str):
        
        #os.path.dirname(os.path.dirname(__file__))+"/"+
        img = np.array(imageio.imread(path))

        if img.shape[-1] == 4:
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        else:
            return img
    
    @staticmethod   
    def write_json(data:dict, path:str, name:str=None):
        if name is None:
            with open(path, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
        else:
            path_to_save = os.path.join(path,name+".json")
            with open(path_to_save, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    data = Midv(uri="http://0.0.0.0:8000/SIDTD")
    data.download_dataset()