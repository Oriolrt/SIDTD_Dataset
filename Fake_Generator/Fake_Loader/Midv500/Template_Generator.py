from matplotlib.pyplot import flag
from Fake_Loader.Midv  import Midv
from ..utils import *
import os
import numpy as np
import random
import cv2



class Template_Generator(Midv):
    __slots__ = ["_img_loader", "_classes", "_fake_metadata", "_transformations","_fake_img_loader"]

    def __init__(self, absolute_path:str, fake_template:dict=None):
        
        if fake_template is None:
            self._fake_template = super().MetaData

        super().__init__(absolute_path)

        self._classes = list(map(lambda class_folder: os.path.join(absolute_path, class_folder),
                                 os.listdir(absolute_path)))


        self.create_loader()



    def create_loader(self):
        for path_classes in self._classes:  # -> [path_n dels fiderents folders per ses diferents classes]
            name_img = path_classes.split("/")[-1]
            src_img = os.path.join(path_classes, "images" ,name_img + ".tif")
            img = read_img(os.path.join(path_classes, "images" ,name_img + ".tif"))
            template = read_json(os.path.join(path_classes, "ground_truth" ,name_img + ".json"))

            self._img_loader.append(super().Img(img, template, name_img,src_img))


    def fit(self, sample):
        #Generate an inpainting for each img of the true loader
        for counter , img in enumerate(self._img_loader):

            fake_img, field = super().Inpaint_and_Rewrite(img=img._img, info=img._meta)
            name_fake_generated =  img._name.split(".")[0] + "_fake_" + str(counter)

            fake_meta = vars(self._fake_template(src=img._relative_path, type_transformation="Inpaint_and_Rewrite",field=field,loader="Midv500",name=name_fake_generated))

            # craeting fake img
            generated_img = super().Img(img._img, img._meta, img._name)

            generated_img.fake_meta = fake_meta
            generated_img.fake_name = fake_meta["name"]
            generated_img.fake_img = fake_img

            self._fake_img_loader.append(generated_img)

        for smpl in tqdm.tqdm(range(sample)):

            img = random.choice(self._img_loader)
            transformation = random.choice(self._transformations)
            name_transform = (transformation.__name__).split(".")[-1]

            if name_transform == "Inpaint_and_Rewrite":
                fake_img, field = super().Inpaint_and_Rewrite(img._img, img._meta)
                name_fake_generated =  img._name.split(".")[0] + "_fake_" + str(counter) + "_" + str(smpl)

                fake_meta = vars(self._fake_template(src=img._relative_path, type_transformation=name_transform,field=field,loader="Midv500",name=name_fake_generated))

                #craeting fake img
                generated_img = super().Img(img._img, img._meta, img._name)

                generated_img.fake_meta = fake_meta
                generated_img.fake_name = fake_meta["name"]
                generated_img.fake_img = fake_img

                self._fake_img_loader.append(generated_img)

            else:
                delta1 = random.sample(range(self._delta_boundary),2)
                delta2 = random.sample(range(self._delta_boundary),2)
                img2 = random.choice(self._img_loader)

                fake_img1, fake_img2 , field1, field2 = super().Crop_and_Replace(img._img, img2._img, img._meta, img2._meta,delta1=delta1, delta2=delta2)

                #img1 info
                name_fake_generated =  img._name.split(".")[0] + "_fake_" + str(counter) + "_" + str(smpl)
                    
                fake_meta = vars(self._fake_template(src=img._relative_path, second_src=img2._relative_path, shift=(delta1,delta2),type_transformation=name_transform,field=field, second_field=field2,loader="Midv500",name=name_fake_generated))

                # craeting fake img1
                generated_img = super().Img(img._img, img._meta, img._name)

                generated_img.fake_meta = fake_meta
                generated_img.fake_name = fake_meta["name"]
                generated_img.fake_img = fake_img1
                generated_img.complement_img = fake_img2

                self._fake_img_loader.append(generated_img)


                #img2 info
                name_fake_generated =  img._name.split(".")[0] + "_fake_" + str(counter) + "_" + str(smpl + 1)                   
                fake_meta2 = vars(self._fake_template(src=img._relative_path, second_src=img2._relative_path, shift=(delta1,delta2),type_transformation=name_transform,field=field, second_field=field2,loader="Midv500",name=name_fake_generated))

                # craeting fake img1
                generated_img2 = super().Img(img2._img, img2._meta, img2._name)

                generated_img2.fake_meta = fake_meta2
                generated_img2.fake_name = fake_meta2["name"]
                generated_img2.fake_img = fake_img2
                generated_img2.complement_img = fake_img1

                self._fake_img_loader.append(generated_img2)

def store_generated_dataset(self):
    store(self._fake_img_loader, path_store=self.absoulute_path+"/Fake_Benchmark_Generated")
