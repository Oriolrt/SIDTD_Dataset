from ast import Str
import copy
from typing import *
from unicodedata import name
from Midv  import Midv
from utils import *
import os
import numpy as np
import random
import cv2
import tqdm
from PIL import ImageFont, ImageDraw, Image


class Template_Generator(Midv):

    __slots__ = ["_img_loader", "_classes", "_fake_template", "_transformations","_fake_img_loader","_annotations_path","_imgs_path","_delta_boundary","_static_path", "_flag"]

    def __init__(self, absolute_path:str ,fake_template:dict = None, delta_boundary:int=10):


        """
            The initialitation of the Generator just create the structure with Images in memory that will serve as a template to create the 
            new information. 
        """

        if fake_template is None:
            self._fake_template = super().MetaData

        #assert isinstance(fake_template, dict), "The metadata template to save the information must be a dict"


        super().__init__(absolute_path)

        path_template = super().get_template_path()
        self._delta_boundary = delta_boundary
        self._annotations_path = os.path.join(path_template,"annotations")
        self._imgs_path = os.path.join(path_template, "images")

        self._classes = list(map(lambda class_folder: os.path.join(self._imgs_path, class_folder),
                                 os.listdir(self._imgs_path)))
        self._img_loader = {i: [] for i in os.listdir(self._imgs_path)}
        
        #static path
        self._static_path = "/dataset/MIDV2020/dataset/templates/images"

        self.create_loader()



    def create_loader(self) -> List[object]:

        for path_classes in self._classes:
            class_template = read_json(os.path.join(self._annotations_path , path_classes.split("/")[-1] )+ ".json")


            for im in os.listdir(path_classes):
                ninf = path_classes.split("/")
                name_img = ninf[-1] + "_" + im
                src_img = os.path.join(self._imgs_path,ninf[-1],im)
                img = read_img(src_img)

                self._img_loader[ninf[-1]].append(super(Template_Generator, self).Img(img,class_template,name_img,src_img))

    def fit(self,sample) -> List[Image.Image]:
        
        for counter, (key,img_bucket) in enumerate(self._img_loader.items()):
            for idx in range(len(img_bucket)):

                img = random.choice(img_bucket)
                img_id = int(img._relative_path.split("/")[-1].split(".")[0])
                fake_img, field =  super().Inpaint_and_Rewrite(img=img._img,img_id=img_id,info=img._meta)
                name_fake_generated =  img._name.split(".")[0] + "_fake_" + str(counter) + "_" + str(idx)

                #Creating the dict with the metadata
                fake_meta = vars(self._fake_template(src=img._relative_path, type_transformation="Inpaint_and_Rewrite",field=field,loader="Midv2020",name=name_fake_generated))

                generated_img = super().Img(img._img, img._meta, img._name)

                generated_img.fake_meta = fake_meta
                generated_img.fake_name = fake_meta["name"]
                generated_img.fake_img = fake_img

                self._fake_img_loader.append(generated_img)
            
            print(f"General Inpainted for the class {key} Done")
            # equal generation
            for smpl in tqdm.tqdm(range(sample//len(self._classes))):

                img = random.choice(img_bucket)
                img_id = int(img._relative_path.split("/")[-1].split(".")[0])

                name_fake_generated =  img._name.split(".")[0] + "_fake_" + str(counter) + "_" + str(idx)

                transformation = random.choice(self._transformations)
                name_transform = (transformation.__name__).split(".")[-1]
                if name_transform == "Inpaint_and_Rewrite":
                    fake_img, field = super().Inpaint_and_Rewrite(img=img._img, img_id=img_id,info=img._meta)


                    fake_meta = vars(self._fake_template(src=img._relative_path, type_transformation=name_transform,field=field,loader="Midv2020",name=name_fake_generated))


                    # craeting fake img
                    generated_img = super().Img(img._img, img._meta, img._name)

                    generated_img.fake_meta = fake_meta
                    generated_img.fake_name = fake_meta["name"]
                    generated_img.fake_img = fake_img

                    self._fake_img_loader.append(generated_img)

                else:

                    delta1 = random.sample(range(self._delta_boundary),2)
                    delta2 = random.sample(range(self._delta_boundary),2)

                    img2 = random.choice(img_bucket)
                    img_id2 = int(img2._relative_path.split("/")[-1].split(".")[0])
                    fake_img1, fake_img2 , field, field2 = super().Crop_and_Replace(img1=img._img, img2=img2._img, info=img._meta, additional_info=None, img_id1=img_id ,img_id2=img_id2,delta1=delta1,delta2=delta2)
                    
                    #img1 info
                    name_fake_generated =  img._name.split(".")[0] + "_fake_" + str(counter) + "_" + str(smpl + 1 +idx)

                    fake_meta = vars(self._fake_template(src=img._relative_path, second_src=img2._relative_path, shift=(delta1,delta2),type_transformation=name_transform,field=field, second_field=field2,loader="Midv2020",name=name_fake_generated))
            
                    # craeting fake img1
                    generated_img = super().Img(img._img, img._meta, img._name)

                    generated_img.fake_meta = fake_meta
                    generated_img.fake_name = fake_meta["name"]
                    generated_img.fake_img = fake_img1
                    generated_img.complement_img = fake_img2

                    self._fake_img_loader.append(generated_img)


                    #img2 info
                    name_fake_generated =  img._name.split(".")[0] + "_fake_" + str(counter) + "_" + str(smpl + 2 +idx)
                    fake_meta2 = vars(self._fake_template(second_src=img._relative_path, src=img2._relative_path, shift=(delta1,delta2),type_transformation=name_transform,second_field=field, field=field2,loader="Midv2020",name=name_fake_generated))
                    

                    # craeting fake img2
                    generated_img2 = super().Img(img2._img, img2._meta, img2._name)

                    generated_img2.fake_meta = fake_meta2
                    generated_img2.fake_name = fake_meta2["name"]
                    generated_img2.fake_img = fake_img2
                    generated_img2.complement_img = fake_img1

                    self._fake_img_loader.append(generated_img2)


    def store_generated_dataset(self):
        store(self._fake_img_loader, path_store=self.absoulute_path+"/Fake_Benchmark_Generated")


if __name__ == "__main__":
    gen = Template_Generator("/home/cboned/MIDV2020/dataset")
    
    gen.fit(1000)
    
    #gen.store_generated_dataset()