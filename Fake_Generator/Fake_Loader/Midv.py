#Description This file has the super classes with the image class, the video class and the façade of the Midv500 loader

from dataclasses import *
from dataclasses import dataclass
import numpy as np
import cv2
import json
import copy as cp
from abc import ABC, abstractmethod
import os



class Midv(ABC):

    __slots__ = ["_absolute_path","_img_loader", "_fake_img_loader","_transformations"]

    def __init__(self, path: str):
        self._absolute_path = path

        # PlaceHolders for the fake imgs
        self._img_loader = []
        self._fake_img_loader = []
        #self._fake_metadata = fake_template
        self._transformations = [self.Crop_and_Replace, self.Inpaint_and_Rewrite]




    @property
    def absoulute_path(self):
        return self._absolute_path
    
    
    def get_template_path(self):
        return os.path.join(self.absoulute_path, "templates")

    def get_video_path(self):
        return os.path.join(self.absoulute_path, "clips")


    @abstractmethod
    def Crop_and_Replace(self):
        pass

    @abstractmethod
    def Inpaint_and_Rewrite(self):
        pass




    ####################################################################################################
    class Img(object):

        __slots__ = ["_img", "_meta","_name" ,"_fake_name","_fake_img", "_fake_meta", "_complement_img","_relative_path"]

        def __init__(self, img:np.ndarray, metadades: dict, name:str, relative_path:str=None):
            self._img = img
            self._meta = metadades
            self._name = name
            self._relative_path = relative_path

            #still not declared
            self._fake_name = None
            self._fake_img =  None
            self._fake_meta = None


            #La imatge amb la que ha generat la falsificació en cas de que sigui un crop and replace
            self._complement_img = []


        #GETTERS AND SETTERS
        @property
        def fake_name(self):
            return self._fake_name

        @fake_name.setter
        def fake_name(self, name):
            self._fake_name = name

        @property
        def fake_img(self):
            return self._fake_img

        @fake_img.setter
        def fake_img(self, img):
            self._fake_img = img



        @property
        def fake_meta(self):
            return self._fake_meta

        @fake_meta.setter
        def fake_meta(self, meta):
            self._fake_meta = meta


        @property
        def complement_img(self):
            return self._complement_img
        @complement_img.setter
        def complement_img(self, img):
            self._complement_img = img


    ####################################################################################################
    @dataclass
    class MetaData:
        name:str =None
        src:str=None
        second_src:str=None
        loader:str=None
        field:str=None 
        second_field:str = None 
        shift:list=None 
        type_transformation:str=None


    ####################################################################################################

    class Video(object):
        

        __slots__ = ["_video", "_fake_meta", "_true_class_template", "_projections"]



        def __init__(self, class_template:dict):
           
            self._video = []
            self._true_class_template = class_template

            self._fake_meta =None
            self._projections =[]



        def add_frame(self, img: object):
            self._video.append(img)

        def add_projection(self,points):
            self._projections.append(points)

        def compute_projection(self):
            pass

        def plot_square(self):
            pass


