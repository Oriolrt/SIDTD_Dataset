#Description This file has the super classes with the image class, the video class and the façade of the Midv500 loader
from SIDTD.utils.transforms import *
from SIDTD.utils.util   import *

from ast import Str
from dataclasses import dataclass
from PIL import Image
from typing import *


import inspect
import sys


class Midv():

    __slots__ = ["_absolute_path","_img_loader", "_fake_img_loader","_transformations"]

    def __init__(self, path: str):
        self._absolute_path = path

        # PlaceHolders for the fake imgs
        self._img_loader = []
        self._fake_img_loader = []
        self._transformations = [self.Crop_and_Replace, self.Inpaint_and_Rewrite]
        self._flag = 1 if os.path.dirname(inspect.getframeinfo(sys._getframe(1)).filename).split("/")[-1] == "Midv2020" else 0


    @property
    def absoulute_path(self):
        return self._absolute_path
    
    
    def get_template_path(self):
        assert os.path.exists(os.path.join(self.absoulute_path, "Images")), "There's not any image folder in this directory"
        return os.path.join(self.absoulute_path, "Images")

    def get_video_path(self):
        assert os.path.exists(os.path.join(self.absoulute_path, "Images")), "There's not any image folder in this directory"

        return os.path.join(self.absoulute_path, "clips")

    def get_img_annotations_path(self):
        assert os.path.exists(os.path.join(self.absoulute_path, "Annotations")), "There's not any metadata (.json) folder in this directory"

        return os.path.join(self._absolute_path, "Annotations")
        

    def get_field_info(self, info:dict, img_id1:int=None, mark:str = None, force_flag:int=1) -> Tuple[Dict, Str]:
        assert type(info) is dict
        
        if (self._flag != 1 or force_flag != 1) and img_id1 is not None:
            fields = list(np.unique(list(info.keys())))
            for tr in ["photo", "signature"]:
                try:
                    fields.remove(tr)
                except:
                    continue
                
        else:
            assert type(img_id1) is int
            
            selected = list(info["_via_img_metadata"])[img_id1]
            fields = info["_via_img_metadata"][selected]["regions"]

        if not mark or mark == None:
            # (2-12) to avoid the face, the photo and the signature
            field_to_change = random.randint(2, len((fields))-2) if self._flag else random.choice(fields)
            swap_info = fields[field_to_change]
            field_to_return = swap_info["region_attributes"]["field_name"] if self._flag else field_to_change 
        else:
            swap_info = fields[mark]
            field_to_return = mark
            
        return swap_info, field_to_return

    def Crop_and_Replace(self,img1:np.ndarray, img2:np.ndarray, info:dict, additional_info:dict, img_id1:int=None, img_id2:int=None, delta1:list=[2,2], delta2:list = [2,2], mark:str=None, force_flag:int=1) -> Tuple[Image.Image, Image.Image, Str, Str]:
        
        sInfo = True if (np.random.randint(100) > 5) or (mark is not None) else False
        
        if bool(self._flag) is True or bool(force_flag) is True:
            
            if additional_info is None:
                if mark is  None:
                    swap_info_1, field_to_return1 = self.get_field_info(info=info, img_id1=img_id1)
                    swap_info_2, field_to_return2 = self.get_field_info(info=info, img_id1=img_id2)
                    if not sInfo and (swap_info_1 == swap_info_2):
                        while(swap_info_1 == swap_info_2):
                            swap_info_2, field_to_return2 = self.get_field_info(info=additional_info, img_id1=img_id2)
                
                else:
                    swap_info_1, field_to_return1 = self.get_field_info(info=info, img_id1=img_id1, mark=mark)
                    swap_info_2, field_to_return2 = self.get_field_info(info=info, img_id1=img_id2, mark=mark)

            else:
                swap_info_1, field_to_return1 = self.get_field_info(info=info, img_id1=img_id1, mark=mark)
                if mark is not None:
                    swap_info_2, field_to_return2 = self.get_field_info(info=additional_info, img_id1=img_id2, mark=mark)  
                else:
                    swap_info_2, field_to_return2 = self.get_field_info(info=additional_info, img_id1=img_id2)  
                    if not sInfo and (swap_info_1 == swap_info_2):
                        while(swap_info_1 == swap_info_2):
                            swap_info_2, field_to_return1 = self.get_field_info(info=additional_info, img_id1=img_id2)
    
            fake_document1, fake_document2 = replace_info_documents(img1, img2, swap_info_1, swap_info_2, delta1=delta1, delta2=delta2)
        
            return fake_document1, fake_document2 , field_to_return1, field_to_return2           
        
        else:
            assert additional_info != None, "When use Midv500 additional template information must be supplied"
            
            mixed = True if (type(img_id1) is int) or (type(img_id2) is int) else False
            
            if mixed:
                idd = img_id1 if type(img_id1) is int else img_id2
                try:
                    swap_info_1, field_to_return1 = self.get_field_info(info=info, img_id1=idd)
                    swap_info_2, field_to_return2 = self.get_field_info(info=additional_info, force_flag=0)
                except:
                    swap_info_1, field_to_return1 = self.get_field_info(info=additional_info, img_id1=idd)
                    swap_info_2, field_to_return2 = self.get_field_info(info=info, force_flag=0)
        
        
                fake_document1, fake_document2 = replace_info_documents(img1, img2, swap_info_1,swap_info_2, delta1, delta2, flag=self._flag, mixed=mixed)
                
                return fake_document1, fake_document2, field_to_return1, field_to_return2                          

            else:           
                swap_info_1, field_to_return1 = self.get_field_info(info=info, force_flag=0)
                swap_info_2, field_to_return2 = self.get_field_info(info=additional_info, force_flag=0)

                fake_document1, fake_document2 = replace_info_documents(img1, img2, swap_info_1,swap_info_2, delta1, delta2, flag=self._flag, mixed=mixed)
                
                return fake_document1, fake_document2, field_to_return1, field_to_return2                          
                                                                

    
    def Inpaint_and_Rewrite(self,img: np.ndarray, info: dict,img_id: int=None, mark:str=None, force_flag:int=1) -> Tuple[Image.Image, Str]:
        if bool(self._flag) is True or bool(force_flag) is True:
            swap_info, field_to_change = self.get_field_info(info=info,img_id1=img_id, mark=mark)

        else:
            swap_info, field_to_change = self.get_field_info(info=info, mark=mark)

        x0, y0, w, h = bbox_info(swap_info)
        shape = bbox_to_coord(x0, y0, w, h)

        
        try:
            text_str = swap_info["region_attributes"]["value"]

        except:
            text_str =  swap_info["value"]

        mask, _ = mask_from_info(img, shape)
        coord = [x0, y0, w, h]

        fake_text_image =  inpaint_image(img=img, coord=coord, mask=mask, text_str=text_str)
        return fake_text_image, field_to_change

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


