from abc import ABC,abstractmethod
from collections import Counter
from path import Path


import numpy as np
import os
import glob
import zipfile
import shutil
try:
    import imageio.v2 as imageio
except:
    import imageio
import json
import cv2


""""
        self._uri_transfg_pretrained = "http://datasets.cvc.uab.es/SIDTD/imagenet21k+imagenet2012_ViT-L_16.zip"
        self.abs_path_trans_fg = os.path.join(os.getcwd(), "models", "transfg", "transfg_pretrained")
        self.abs_path_code_ex = os.path.join(os.getcwd(), "explore", "pretrained_models")
"all_trained_models", "effnet", "resnet", "vit", "transfg", "arc", "transfg_img_net","no"

"""

class load_models(ABC):
    def __init__(self):
        self._cluster_link = "http://datasets.cvc.uab.es"
        self._url = os.path.join(self._cluster_link, "/SIDTD")
        self._holder_path = os.path.join(os.getcwd(), "models", "explore", "pretrained_models")
        if not os.path.exists(self._holder_path):
            os.makedirs(self._holder_path)

    @property
    def holder_path(self):
        return self._holder_path
    @holder_path.setter
    def holder_path(self, value):
        self._holder_path = value




class efficientnet_b3(load_models):
    def __init__(self, path:Path=None, cropped:bool=False):
        super().__init__()
        if path is not None:
            self._holder_path = path

        base_folder = "balanced_templates_SIDTD"

        # path where to put the pretrained model
        abs_path_model = os.path.join(self._holder_path, base_folder)

        # the path where the model is in the server
        server_path_model = os.path.join(self._url, base_folder)
        os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex,
                                                                                       server_path_model + "/efficientnet-b3_trained_models.zip"))
        with zipfile.ZipFile(abs_path_model + "/efficientnet-b3_trained_models.zip", 'r') as zip_ref:
            zip_ref.extractall(abs_path_model)




class resnet50(load_models):
    def __init__(self):
        super().__init__()

class vit_large_patch16_224(load_models):
    def __init__(self):
        super().__init__()

class trans_fg(load_models):
    def __init__(self):
        super().__init__()

class coatten_fcn_model(load_models):
    def __init__(self):
        super().__init__()



def download_models(unbalanced :bool =False, cropped :bool =False, type_models :str ="transfg_img_net"):
    if not unbalanced:
        base_folder = "balanced_templates_SIDTD"
        arc_name = "/coatten_fcn_model_trained_models.zip"
    else:
        arc_name = "/coattention_trained_models.zip"
        if cropped:
            base_folder = "unbalanced_clip_cropped_SIDTD"
        else:
            base_folder = "unbalanced_clip_background_SIDTD"


    abs_path_model = os.path.join(self.abs_path_code_ex, base_folder)
    server_path_model = os.path.join(self._uri, base_folder)

    if type_models == "all_trained_models":
        os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex
                                                                                       ,server_path_model +"/efficientnet-b3_trained_models.zip"))
        with zipfile.ZipFile(abs_path_model + "/efficientnet-b3_trained_models.zip", 'r') as zip_ref:
            zip_ref.extractall(abs_path_model)

        os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex,
                                                                                       server_path_model + "/resnet50_trained_models.zip"))
        with zipfile.ZipFile(abs_path_model + "/resnet50_trained_models.zip", 'r') as zip_ref:
            zip_ref.extractall(abs_path_model)

        os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex,
                                                                                       server_path_model + "/vit_large_patch16_224_trained_models.zip"))
        with zipfile.ZipFile(abs_path_model + "/vit_large_patch16_224_trained_models.zip", 'r') as zip_ref:
            zip_ref.extractall(abs_path_model)

        os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex,
                                                                                       server_path_model + "/trans_fg_trained_models.zip"))
        with zipfile.ZipFile(abs_path_model + "/trans_fg_trained_models.zip", 'r') as zip_ref:
            zip_ref.extractall(abs_path_model)

        os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex,
                                                                                       server_path_model + arc_name))
        with zipfile.ZipFile(abs_path_model + arc_name, 'r') as zip_ref:
            zip_ref.extractall(abs_path_model)

        os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_trans_fg,
                                                                                       self._uri_transfg_pretrained))
        with zipfile.ZipFile(self.abs_path_trans_fg + "/imagenet21k+imagenet2012_ViT-L_16.zip", 'r') as zip_ref:
            zip_ref.extractall(self.abs_path_trans_fg)

    if type_models == "effnet":
        os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex,
                                                                                       server_path_model + "/efficientnet-b3_trained_models.zip"))
        with zipfile.ZipFile(abs_path_model + "/efficientnet-b3_trained_models.zip", 'r') as zip_ref:
            zip_ref.extractall(abs_path_model)

    elif type_models == "resnet":
        os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex,
                                                                                       server_path_model + "/resnet50_trained_models.zip"))
        with zipfile.ZipFile(abs_path_model + "/resnet50_trained_models.zip", 'r') as zip_ref:
            zip_ref.extractall(abs_path_model)

    elif type_models == "vit":
        os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex,
                                                                                       server_path_model + "/vit_large_patch16_224_trained_models.zip"))
        with zipfile.ZipFile(abs_path_model + "/vit_large_patch16_224_trained_models.zip", 'r') as zip_ref:
            zip_ref.extractall(abs_path_model)

    elif type_models == "transfg":
        os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex,
                                                                                       server_path_model + "/trans_fg_trained_models.zip"))
        with zipfile.ZipFile(abs_path_model + "/trans_fg_trained_models.zip", 'r') as zip_ref:
            zip_ref.extractall(abs_path_model)

        os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_trans_fg,
                                                                                       self._uri_transfg_pretrained))
        with zipfile.ZipFile(self.abs_path_trans_fg + "/imagenet21k+imagenet2012_ViT-L_16.zip", 'r') as zip_ref:
            zip_ref.extractall(self.abs_path_trans_fg)

    elif type_models == "arc":
        os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_code_ex,
                                                                                       server_path_model + arc_name))
        with zipfile.ZipFile(abs_path_model + arc_name, 'r') as zip_ref:
            zip_ref.extractall(abs_path_model)

    elif type_models == "transfg_img_net":
        os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(self.abs_path_trans_fg,
                                                                                       self._uri_transfg_pretrained))
        with zipfile.ZipFile(self.abs_path_trans_fg + "/imagenet21k+imagenet2012_ViT-L_16.zip", 'r') as zip_ref:
            zip_ref.extractall(self.abs_path_trans_fg)

    else:
        print("OPTION: do not download trained model files")

