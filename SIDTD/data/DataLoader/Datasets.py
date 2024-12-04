import logging

from SIDTD.utils.util import *


from abc import ABC,abstractmethod
from collections import Counter
from pathlib import Path


import numpy as np
import os
import glob
import zipfile
import shutil
import tqdm

try:
    import imageio.v2 as imageio
except:
    import imageio



class Dataset(ABC):
    def __init__(self) -> None:
        self._cluster_link = 'http://datasets.cvc.uab.es'
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

    def __init__(self,download_original:bool =False, custom_path_to_download:Optional[str]=None) -> None:
        
        super().__init__()
        
        ######### Conditioned and original structure variables
        self._download_original = download_original
        
        
        ###### Static links
        self._uri = self._cluster_link + "/SIDTD/"
        self._images_path = os.path.join(self._uri,"templates.zip")
        self._clips_path = os.path.join(self._uri,"clips.zip")
        self._clips_cropped_path = os.path.join(self._uri,"clips_cropped.zip")
        self._videos_path = os.path.join(self._uri,"videos.zip")

        ## static Csv

        self._uri_static_kfold_templates = os.path.join(self._uri, 'kfold_split_templates.zip')
        self._uri_static_kfold_clips = os.path.join(self._uri,"kfold_split_clips.zip")
        self._uri_static_kfold_cropped_clips = os.path.join(self._uri,"kfold_split_clips_cropped.zip")
        self._uri_static_normal_templates = os.path.join(self._uri,'hold_out_split.zip')
        self._uri_static_normal_clips = os.path.join(self._uri,'unbalanced_hold_out_split.zip')
        self._uri_static_shot_templates = os.path.join(self._uri, 'few_shot_split_templates.zip')
        self._uri_static_shot_clips = os.path.join(self._uri, 'few_shot_split_clips_cropped.zip')
        


        if download_original == True:
            self._define_paths()
        
        
        #path to download
        if custom_path_to_download is None:
            self._path_to_download = os.getcwd()
        else:
            self._path_to_download = custom_path_to_download



        self._abs_path = os.path.join(self._path_to_download,os.path.basename(self._uri)) # cwd/datasets/SIDTD/...
        
        

        self.abs_path_code_ex_csv = os.path.join(os.getcwd(), "models", "explore", "static")

  
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
        
    def download_static_csv(self, partition_kind:str="hold_out", type_download:str = "templates"):
        os.system(f"""mkdir  {self._abs_path}""")

        if partition_kind == "kfold":
            if type_download=="templates":                    
                os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex_csv,self._uri_static_kfold_templates))
                with zipfile.ZipFile(self.abs_path_code_ex_csv+"/split_kfold.zip", 'r') as zip_ref:
                    zip_ref.extractall(self.abs_path_code_ex_csv)
            elif type_download=="clips":
                os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex_csv,self._uri_static_kfold_clips))
                with zipfile.ZipFile(self.abs_path_code_ex_csv+"/split_kfold_unbalanced.zip", 'r') as zip_ref:
                    zip_ref.extractall(self.abs_path_code_ex_csv)
            elif type_download=="clips_cropped":    
                os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex_csv,self._uri_static_kfold_cropped_clips))
                with zipfile.ZipFile(self.abs_path_code_ex_csv+"/split_kfold_cropped_unbalanced.zip", 'r') as zip_ref:
                    zip_ref.extractall(self.abs_path_code_ex_csv)

        elif partition_kind == "hold_out":
            if type_download=="templates": 
                os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex_csv,self._uri_static_normal_templates))
                with zipfile.ZipFile(self.abs_path_code_ex_csv+"/split_normal.zip", 'r') as zip_ref:
                    zip_ref.extractall(self.abs_path_code_ex_csv)
            elif type_download=="clip":
                os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex_csv,self._uri_static_normal_clips))
                with zipfile.ZipFile(self.abs_path_code_ex_csv+"/cross_val_unbalanced.zip", 'r') as zip_ref:
                    zip_ref.extractall(self.abs_path_code_ex_csv)

        else:
            if type_download=="templates":
                os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex_csv,self._uri_static_shot_templates))
                with zipfile.ZipFile(self.abs_path_code_ex_csv+"/split_shot.zip", 'r') as zip_ref:
                    zip_ref.extractall(self.abs_path_code_ex_csv)
            elif type_download=="clip":
                os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex_csv,self._uri_static_shot_clips))
                with zipfile.ZipFile(self.abs_path_code_ex_csv+"/split_shot_unbalanced.zip", 'r') as zip_ref:
                    zip_ref.extractall(self.abs_path_code_ex_csv)

    def download_dataset(self, type_download: str = "templates"):
        os.system(f"""mkdir  {self._abs_path}""")

        if type_download == "all_dataset":
            os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self._abs_path, self._uri))
            if self._download_original: raise NotImplementedError

        elif type_download == "clips_cropped":
            os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self._abs_path,
                                                                                           self._clips_cropped_path))


            with zipfile.ZipFile(self._abs_path + "/clips_cropped.zip", 'r') as zip_ref:
                print(
                    "Starting to decompress the data you actually downloaded, It may spend a lot of time")

                for member in tqdm.tqdm(zip_ref.infolist(), desc='Extracting '):
                    try:
                        zip_ref.extract(member, self._abs_path)
                    except zipfile.error as e:
                        pass

            os.remove(self._abs_path + "/clips_cropped.zip")

            if self._download_original: raise NotImplementedError

        elif type_download == "clips":
            os.system(
                "bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self._abs_path, self._clips_path))
            with zipfile.ZipFile(self._abs_path + "/clips.zip", 'r') as zip_ref:
                print(
                    "Starting to decompress the data you actually downloaded, It may spend a lot of time")

                for member in tqdm.tqdm(zip_ref.infolist(), desc='Extracting '):
                    try:
                        zip_ref.extract(member, self._abs_path)
                    except zipfile.error as e:
                        pass

            os.remove(self._abs_path + "/clips.zip")

            if self._download_original: raise NotImplementedError

        elif type_download == "videos":
            os.system(
                "bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self._abs_path, self._videos_path))
            with zipfile.ZipFile(self._abs_path + "/videos.zip", 'r') as zip_ref:
                print(
                    "Starting to decompress the data you actually downloaded, It may spend a lot of time")

                for member in tqdm.tqdm(zip_ref.infolist(), desc='Extracting '):
                    try:
                        zip_ref.extract(member, self._abs_path)
                    except zipfile.error as e:
                        pass

            os.remove(self._abs_path + "/videos.zip")

            if self._download_original: raise NotImplementedError

        elif type_download == "templates":
            os.system(
                "bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self._abs_path, self._images_path))
            with zipfile.ZipFile(self._abs_path + "/templates.zip", 'r') as zip_ref:
                logging.warning("Starting to decompress the data you actually downloaded, It may spend a lot of time")

                for member in tqdm.tqdm(zip_ref.infolist(), desc='Extracting '):
                    try:
                        zip_ref.extract(member, self._abs_path)
                    except zipfile.error as e:
                        pass

            os.remove(self._abs_path + "/templates.zip")

            if self._download_original: self.create_structure_images()

        else:
            print("OPTION: do not download dataset files")

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

            section[file[0]] =  clas

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

    def __name__(self):
        return "SIDTD"


if __name__ == "__main__":
    data = SIDTD()
