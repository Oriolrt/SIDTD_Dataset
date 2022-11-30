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
import zipfile
import shutil
try:
    import imageio.v2 as imageio
except:
    import imageio
import json
import cv2
import sys
from path import Path



class Dataset(ABC):
    def __init__(self) -> None:
        self._cluster_link = "http://datasets.cvc.uab.es"
        
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


class SIDTD(Dataset):

    def __init__(self,conditioned:bool=True,download_original:bool =False) -> None:
        
        super().__init__()
        
        ######### Conditioned and original structure variables
        self._conditioned = conditioned
        self._download_original = download_original
        
        #dict to map classes
        self._map_classes = self.map_classes() if conditioned is True else None
        
        ###### Static links
        self._uri = self._cluster_link + "/SIDTD"
        self._images_path = "http://datasets.cvc.uab.es/SIDTD/templates.zip "
        self._clips_path = "http://datasets.cvc.uab.es/SIDTD/clips.zip"
        self._uri_videos = "http://datasets.cvc.uab.es/SIDTD/videos.zip"
        self._uri_transfg_pretrained = "http://datasets.cvc.uab.es/SIDTD/imagenet21k+imagenet2012_ViT-L_16.zip"

        ## static Csv
        self._uri_static_kfold_balanced = 'http://datasets.cvc.uab.es/SIDTD/split_kfold.zip'
        self._uri_static_kfold_unbalanced = "http://datasets.cvc.uab.es/SIDTD/split_kfold_unbalanced.zip"
        self._uri_static_normal_balanced = 'http://datasets.cvc.uab.es/SIDTD/split_normal.zip'
        self._uri_static_normal_unbalanced = 'http://datasets.cvc.uab.es/SIDTD/cross_val_unbalanced.zip'
        self._uri_static_shot_balanced = 'http://datasets.cvc.uab.es/SIDTD/split_shot.zip'
        self._uri_static_shot_unbalanced = 'http://datasets.cvc.uab.es/SIDTD/split_shot_unbalanced.zip'
        


        if download_original == True:
            self._define_paths()
        
        
        #path to download
        self._path_to_download = os.path.join(os.getcwd(), "datasets")

        self._abs_path = os.path.join(self._path_to_download,os.path.basename(self._uri)) # cwd/datasets/SIDTD/...
        self.abs_path_code_ex_csv = os.path.join(os.getcwd(), "code_examples", "static")
        self.abs_path_code_ex = os.path.join(os.getcwd(), "code_examples", "pretrained_models")
        self.abs_path_trans_fg = os.path.join(os.getcwd(), "models", "transfg", "transfg_pretrained")

        if not os.path.exists(self.abs_path_trans_fg):
            os.makedirs(self.abs_path_trans_fg)
            
  
    def _define_paths(self) ->None:        
        ## Path to reconstruct the original structure
        ##images
        self._original_abs_imgs_path = "MIDV2020/templates"
        self._original_imgs_path = os.path.join(self._original_abs_imgs_path, "images")
        self._original_ann_path  = os.path.join(self._original_abs_imgs_path, "annotations")
        
        ## videos
        self._original_videos = "MIDV2020/video"

        ##clips
        self._original_clips_path = "MIDV2020/clips"
        self._original_clips_imgs_path = os.path.join(self._original_clips_path, "images")
        self._original_clips_ann_path  = os.path.join(self._original_clips_path, "annotations")
        
    def download_static_csv(self, partition_kind:str="cross" ,unbalanced:bool=True):

        if partition_kind == "kfold":
            if not unbalanced:                    
                os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex_csv,self._uri_static_kfold_balanced))
                with zipfile.ZipFile(self.abs_path_code_ex_csv+"/split_kfold.zip", 'r') as zip_ref:
                    zip_ref.extractall(self.abs_path_code_ex_csv)
            else:
                os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex_csv,self._uri_static_kfold_unbalanced))
                with zipfile.ZipFile(self.abs_path_code_ex_csv+"/split_kfold_unbalanced.zip", 'r') as zip_ref:
                    zip_ref.extractall(self.abs_path_code_ex_csv)

        elif partition_kind == "cross":
            if not unbalanced:
                os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex_csv,self._uri_static_normal_balanced))
                with zipfile.ZipFile(self.abs_path_code_ex_csv+"/split_normal.zip", 'r') as zip_ref:
                    zip_ref.extractall(self.abs_path_code_ex_csv)
            else:
                os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex_csv,self._uri_static_normal_unbalanced))
                with zipfile.ZipFile(self.abs_path_code_ex_csv+"/cross_val_unbalanced.zip", 'r') as zip_ref:
                    zip_ref.extractall(self.abs_path_code_ex_csv)

        else:
            if not unbalanced:
                os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex_csv,self._uri_static_shot_balanced))
                with zipfile.ZipFile(self.abs_path_code_ex_csv+"/split_shot.zip", 'r') as zip_ref:
                    zip_ref.extractall(self.abs_path_code_ex_csv)
            else:
                os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex_csv,self._uri_static_shot_unbalanced))
                with zipfile.ZipFile(self.abs_path_code_ex_csv+"/split_shot_unbalanced.zip", 'r') as zip_ref:
                    zip_ref.extractall(self.abs_path_code_ex_csv)


    def download_dataset(self, type_download:str = "images"):
        
        if type_download == "all_dataset":    
            os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self._abs_path,self._uri))
            if self._download_original:raise NotImplementedError

        elif type_download == "clips":
            os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self._abs_path,self._clips_path))
            with zipfile.ZipFile(self._abs_path+"/clips.zip", 'r') as zip_ref:
                zip_ref.extractall(self._abs_path)
            if self._download_original:raise NotImplementedError
        
        elif type_download == "videos":
            os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self._abs_path,self._videos_path))
            with zipfile.ZipFile(self._abs_path+"/videos.zip", 'r') as zip_ref:
                zip_ref.extractall(self._abs_path)
            if self._download_original:raise NotImplementedError

        elif type_download == "images":
            os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self._abs_path,self._images_path))
            with zipfile.ZipFile(self._abs_path+"/templates.zip", 'r') as zip_ref:
                zip_ref.extractall(self._abs_path)
            if self._download_original:self.create_structure_images()
            
        else:
            print("OPTION: do not download dataset files") 

    def download_models(self, unbalanced:str="yes", type_models:str="transfg_img_net"):
        
        if not unbalanced:
            balanced_folder = "balanced_templates_SIDTD"
            arc_name = "/coatten_fcn_model_trained_models.zip"
        else:
            balanced_folder = "unbalanced_clip_background_SIDTD"
            arc_name = "/coattention_trained_models.zip"
        
        abs_path_model = os.path.join(self.abs_path_code_ex, balanced_folder)
        server_path_model = os.path.join(self._uri, balanced_folder)

        if type_models == "all_trained_models":   
            os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex,server_path_model +"/efficientnet-b3_trained_models.zip" ))
            with zipfile.ZipFile(abs_path_model+"/efficientnet-b3_trained_models.zip", 'r') as zip_ref:
                zip_ref.extractall(abs_path_model)

            os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex,server_path_model+"/resnet50_trained_models.zip"))
            with zipfile.ZipFile(abs_path_model+"/resnet50_trained_models.zip", 'r') as zip_ref:
                zip_ref.extractall(abs_path_model)

            os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex, server_path_model+"/vit_large_patch16_224_trained_models.zip"))
            with zipfile.ZipFile(abs_path_model+"/vit_large_patch16_224_trained_models.zip", 'r') as zip_ref:
                zip_ref.extractall(abs_path_model)

            os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex, server_path_model+"/trans_fg_trained_models.zip"))
            with zipfile.ZipFile(abs_path_model+"/trans_fg_trained_models.zip", 'r') as zip_ref:
                zip_ref.extractall(abs_path_model)

            os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex, server_path_model+ arc_name))
            with zipfile.ZipFile(abs_path_model+ arc_name, 'r') as zip_ref:
                zip_ref.extractall(abs_path_model)

            os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_trans_fg,self._uri_transfg_pretrained))
            with zipfile.ZipFile(self.abs_path_trans_fg+"/imagenet21k+imagenet2012_ViT-L_16.zip", 'r') as zip_ref:
                zip_ref.extractall(self.abs_path_trans_fg)

        if type_models == "effnet":
            os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex,server_path_model+"/efficientnet-b3_trained_models.zip"))
            with zipfile.ZipFile(abs_path_model+"/efficientnet-b3_trained_models.zip", 'r') as zip_ref:
                zip_ref.extractall(abs_path_model)

        elif type_models == "resnet":
            os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex,server_path_model+"/resnet50_trained_models.zip"))
            with zipfile.ZipFile(abs_path_model+"/resnet50_trained_models.zip", 'r') as zip_ref:
                zip_ref.extractall(abs_path_model)

        elif type_models == "vit":
            os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex,server_path_model+"/vit_large_patch16_224_trained_models.zip"))
            with zipfile.ZipFile(abs_path_model+"/vit_large_patch16_224_trained_models.zip", 'r') as zip_ref:
                zip_ref.extractall(abs_path_model)

        elif type_models == "transfg":
            os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex, server_path_model+ "/trans_fg_trained_models.zip"))
            with zipfile.ZipFile(abs_path_model+"/trans_fg_trained_models.zip", 'r') as zip_ref:
                zip_ref.extractall(abs_path_model)

            os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_trans_fg,self._uri_transfg_pretrained))
            with zipfile.ZipFile(self.abs_path_trans_fg+"/imagenet21k+imagenet2012_ViT-L_16.zip", 'r') as zip_ref:
                zip_ref.extractall(self.abs_path_trans_fg)

        elif type_models == "arc":
            os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex, server_path_model +  arc_name))
            with zipfile.ZipFile(abs_path_model+ arc_name, 'r') as zip_ref:
                zip_ref.extractall(abs_path_model)

        elif type_models == "transfg_img_net":
            os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_trans_fg,self._uri_transfg_pretrained))
            with zipfile.ZipFile(self.abs_path_trans_fg+"/imagenet21k+imagenet2012_ViT-L_16.zip", 'r') as zip_ref:
                zip_ref.extractall(self.abs_path_trans_fg)

        else:
            print("OPTION: do not download trained model files") 
           
                          
    # TODO test it
    def create_structure_videos(self):
        videos_abs_path = os.path.join(self._abs_path, "videos", "reals")
        
        map_videos = self.create_and_map_classes_objects(path=videos_abs_path)
        
        for clas, video_set in map_videos.items():
            path_video_save = os.path.join(self._path_to_download, self._original_videos_path, clas)            
            for video in video_set:
                name_video = video.split("/")[-1].split("_")[-1]
                #make the copy 
                shutil.copyfile(video, path_video_save+f"/{name_video}")           
        pass 

    def create_structure_clips(self):
        clips_abs_path = os.path.join(self._abs_path, "clips","Images", "reals")
        clips_ann_abs_path = os.path.join(self._abs_path, "clips","Annotations", "reals")
        
        map_imgs = self.create_and_map_classes_objects(path=clips_abs_path)
        map_annotations = self.create_and_map_classes_annotations(path=clips_ann_abs_path)
        
 
        for clas, img_set in map_imgs.items():
            template = self.read_json(map_annotations[clas]) #dict
            path_ann_save = os.path.join(self._path_to_download, self._original_ann_path,clas)
            self.write_json(template, path_ann_save, clas)
            
            path_img_save = os.path.join(self._path_to_download, self._original_imgs_path, clas)
            for img in img_set:
                name_img = img.split("/")[-1].split("_")[-1] #get the image numbe (82.jpg example)
                shutil.copyfile(img, path_img_save+f"/{name_img}")           
       
        pass  
                  
    def create_structure_images(self):
        
        
        img_abs_path = os.path.join(self._abs_path,"templates","Images", "reals")
        ann_abs_path = os.path.join(self._abs_path, "templates", "Annotations", "reals")       
        
        map_imgs = self.create_and_map_classes_objects(path=img_abs_path)
        map_annotations = self.create_and_map_classes_annotations(path=ann_abs_path)

        for clas, img_set in map_imgs.items():
            template = self.read_json(map_annotations[clas]) #dict
            path_ann_save = os.path.join(self._path_to_download, self._original_ann_path,clas)
            self.write_json(template, path_ann_save, clas)
            
            path_img_save = os.path.join(self._path_to_download, self._original_imgs_path, clas)
            for img in img_set:
                name_img = img.split("/")[-1].split("_")[-1] #get the image numbe (82.jpg example)
                shutil.copyfile(img, path_img_save+f"/{name_img}")           

                #im = self.read_img(img)  
                #imageio.imwrite(os.path.join(path_img_save,name_img), im)
                
                
    def create_and_map_classes_objects(self, path:Path):
        map_class = {}
        
        for obj in os.listdir(path):
            
            if obj.endswith("html"):continue
            spl = obj.split("_")
            class_image = "_".join(spl[:2]) if not spl[1].isnumeric() else spl[0]
            original_class_path = os.path.join(self._path_to_download,self._original_imgs_path, class_image)
            if os.path.exists(original_class_path):
                if map_class.get(class_image) is not None:
                    map_class[class_image].append(os.path.join(path, obj))
                else:
                    map_class[class_image] = [os.path.join(path, obj)]

            else:
                map_class[class_image] = [os.path.join(path, obj)]
                os.makedirs(original_class_path)
  
        return map_class                
            
    
    def create_and_map_classes_annotations(self, path:Path):
        map_annotation = {
            
        }
        for annotation in os.listdir(path):
            if annotation.endswith("html"):continue
            spl = annotation.split("_")
            key = os.path.splitext("_".join(spl[:2]) if not spl[1].isnumeric() else spl[0])[0]
            class_ann = key
            original_class_path = os.path.join(self._path_to_download,path, class_ann)

            if os.path.exists(original_class_path):
                map_annotation[class_ann] = os.path.join(path,annotation)
                continue
            else:
                map_annotation[class_ann] = os.path.join(path,annotation)
                os.makedirs(original_class_path)

                                
        return map_annotation    


    def map_classes(self, type_data:str="templates"):
        classes = {"reals":{}, "fakes":{}}
        if type_data == "videos":
            fakes = [(file, "fakes") for file in glob.glob(os.path.join(os.getcwd(), "datasets",self.__name__(),type_data, 'fakes',"*"))]
            reals = [(file, "reals") for file in glob.glob(os.path.join(os.getcwd(), "datasets",self.__name__(), type_data, 'reals',"*"))]
        else:     
            fakes = [(file, "fakes") for file in glob.glob(os.path.join(os.getcwd(), "datasets",self.__name__(),type_data, "Images", 'fakes',"*"))]
            reals = [(file, "reals") for file in glob.glob(os.path.join(os.getcwd(), "datasets",self.__name__(), type_data, "Images", 'reals',"*"))]
        
        for file in (fakes+reals):
            section = classes[file[1]]
            clas = file[0].split("/")[-1].split("_")[0]
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
    
    def map_metaclass(self, l:list):
        # from path get the first word of the name of the file that is the metaclass
        return [i.split("/")[-1].split("_")[0] for i in l ]
    
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
                
    def __name__(self):
        return "SIDTD"


if __name__ == "__main__":
    data = SIDTD()
    data.download_models(type_models="all_trained_models") 