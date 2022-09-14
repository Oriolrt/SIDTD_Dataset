#from unicodedata import name
#from xml.etree.ElementInclude import default_loader
import numpy as np 
import math
from abc import ABC,abstractmethod
import time as t
import wget
import os
import subprocess
import glob
from collections import Counter




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
    def dowload_dataset(self, output_directory:str = None):
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

    def dowload_dataset(self, output_directory: str = None):
   
        if self._download_original:
            files = os.system("bash -c 'wget -erobots=off --level=100 --cut-dirs=4 -nH -P {}  -r {}'".format(os.path.join(self._uri.split("/")[-2],self._uri.split("/")[-1]),self._uri))
        else:
            files = os.system("bash -c 'wget -nH -P Fake_Midv --cut-dirs=5 -r {}'".format(self._uri+"/fake_dataset"))

    def map_classes(self):
        classes = {"reals":{}, "fakes":{}}
        fakes = [(file, "fakes") for file in glob.glob("dataset/Midv/fakes/*.jpg")]
        reals = [(file, "reals") for file in glob.glob("dataset/Midv/reals/*.jpg")]
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

if __name__ == "__main__":
    t = Midv()
    print(t.num_fake_classes())