from SIDTD.data.DataGenerator.Midv  import Midv
from SIDTD.utils.util  import *

import os




class Video_Generator(Midv):

    __slots__ = ["sample", "absolute_path", "_annotations_path", "_imgs_path","_classes", "__video_loader"]

    def __init__(self, path: str, fake_template: dict):
        
        super().__init__(path, fake_template)
        path_true_templates = super().get_template_path()
        path_clips = super().get_video_path()

        self._annotations_path = os.path.join(path_clips,"annotations")
        self._imgs_path = os.path.join(path_clips, "images")

        self._classes = list(map(lambda class_folder: os.path.join(self._imgs_path, class_folder),
                                 os.listdir(self._imgs_path)))
        
        self._video_loader = {i: {int(j): None} for i in os.listdir(self._imgs_path)}

        #Creating Placeholder for the true videos
        for i in os.listdir(self._imgs_path):
            self._video_loader[i] = {}
            for j in os.listdir(i):
                self._video_loader[i][int(j)] = None

        #creating folder of annotations [for each class we have the path to the video template]
        for i in os.listdir(self._annotations_path):
            self._annotation_loader[i] = []
            for j in os.listdir(i):
                self._annotations_path.append(os.path.join(self._annotations_path,i,j))

                #Afegir un for mes per a passar les imatges (read dels jsons....)




        