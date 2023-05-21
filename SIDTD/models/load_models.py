from abc import ABC
from path import Path


import os
import zipfile
try:
    import imageio.v2 as imageio
except:
    import imageio


class load_models(ABC):
    def __init__(self):
        self._cluster_link = "http://datasets.cvc.uab.es"
        self._url = os.path.join(self._cluster_link, "SIDTD")
        self._holder_path = os.path.join(os.getcwd(), "models", "explore", "pretrained_models")
        if not os.path.exists(self._holder_path):
            os.makedirs(self._holder_path)

    @property
    def holder_path(self):
        return self._holder_path
    @holder_path.setter
    def holder_path(self, value):
        self._holder_path = value


    def download(self, server_path:Path, local_path:Path):
        os.system("bash -c 'wget -erobots=off -m -k --cut-dirs=1 -nH -P {} {}'".format(local_path,server_path))
        zip_filename = server_path.split("/")[-1]
        with zipfile.ZipFile(os.path.join(local_path, zip_filename), 'r') as zip_ref:
            zip_ref.extractall(local_path)



class efficientnet_b3(load_models):
    def __init__(self, path:Path=None, weights:str="clips"):
        super().__init__()
        if path is not None:
            self._holder_path = path

        if weights == "templates":
            base_folder = "balanced_templates_SIDTD"
        elif weights == "clips":
            base_folder = "unbalanced_clip_background_SIDTD"
        elif weights == "clips_cropped":
            base_folder = "unbalanced_clip_cropped_SIDTD"

        else:
            raise "You need to choose among [templates, clips, clips_cropped]"

        # path where to put the pretrained model
        self._abs_path_model = os.path.join(self._holder_path, base_folder)
        # the path where the model is in the server
        self._server_path_model = os.path.join(self._url, base_folder) + "/efficientnet-b3_trained_models.zip"

    def download(self):
        super().download(server_path=self._server_path_model, local_path=self._abs_path_model)


class resnet50(load_models):
    def __init__(self, path:Path=None, weights:str="clips"):
        super().__init__()
        if path is not None:
            self._holder_path = path

        if weights == "templates":
            base_folder = "balanced_templates_SIDTD"
        elif weights == "clips":
            base_folder = "unbalanced_clip_background_SIDTD"
        elif weights == "clips_cropped":
            base_folder = "unbalanced_clip_cropped_SIDTD"
        else:
            raise "You need to choose among [templates, clips, clips_cropped]"

        # path where to put the pretrained model
        self._abs_path_model = os.path.join(self._holder_path, base_folder)
        # the path where the model is in the server
        self._server_path_model = os.path.join(self._url, base_folder) + "/resnet50_trained_models.zip"

    def download(self):
        super().download(server_path=self._server_path_model, local_path=self._abs_path_model)


class vit_large_patch16_224(load_models):
    def __init__(self, path:Path=None, weights:str="clips"):
        super().__init__()
        if path is not None:
            self._holder_path = path

        if weights == "templates":
            base_folder = "balanced_templates_SIDTD"
        elif weights == "clips":
            base_folder = "unbalanced_clip_background_SIDTD"
        elif weights == "clips_cropped":
            base_folder = "unbalanced_clip_cropped_SIDTD"
        else:
            raise "You need to choose among [templates, clips, clips_cropped]"

        # path where to put the pretrained model
        self._abs_path_model = os.path.join(self._holder_path, base_folder)
        # the path where the model is in the server
        self._server_path_model = os.path.join(self._url, base_folder) + "/vit_large_patch16_224_trained_models.zip"

    def download(self):
        super().download(server_path=self._server_path_model, local_path=self._abs_path_model)


class trans_fg(load_models):
    def __init__(self, path:Path=None, weights:str="clips"):
        super().__init__()
        if path is not None:
            self._holder_path = path

        if weights == "templates":
            base_folder = "balanced_templates_SIDTD"
        elif weights == "clips":
            base_folder = "unbalanced_clip_background_SIDTD"
        elif weights == "clips_cropped":
            base_folder = "unbalanced_clip_cropped_SIDTD"
        else:
            raise "You need to choose among [templates, clips, clips_cropped]"

        # path where to put the pretrained model
        self._abs_path_model = os.path.join(self._holder_path, base_folder)
        # the path where the model is in the server
        self._server_path_model = os.path.join(self._url, base_folder) + "/trans_fg_trained_models.zip"

    def download(self):
        super().download(server_path=self._server_path_model, local_path=self._abs_path_model)


class coatten_fcn_model(load_models):
    def __init__(self, path:Path=None, weights:str="clips"):
        super().__init__()
        if path is not None:
            self._holder_path = path

        if weights == "templates":
            base_folder = "balanced_templates_SIDTD"
            arc_name = "/coatten_fcn_model_trained_models.zip"

        elif weights == "clips":
            base_folder = "unbalanced_clip_background_SIDTD"
            arc_name = "/coattention_trained_models.zip"
        elif weights == "clips_cropped":
            base_folder = "unbalanced_clip_cropped_SIDTD"
            arc_name = "/coattention_trained_models.zip"
        else:
            raise "You need to choose among [templates, clips, clips_cropped]"

        # path where to put the pretrained model
        self._abs_path_model = os.path.join(self._holder_path, base_folder)
        # the path where the model is in the server
        self._server_path_model = os.path.join(self._url, base_folder) + arc_name

    def download(self):
        super().download(server_path=self._server_path_model, local_path=self._abs_path_model)


if __name__ == "__main__":
    trans_fg(weights="clips_cropped").download()