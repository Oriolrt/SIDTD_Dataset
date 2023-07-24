import matplotlib.pyplot as plt

from SIDTD.data.DataGenerator.Midv  import Midv
from SIDTD.utils.util  import *

from typing import *
from unicodedata import name
from PIL import ImageFont, ImageDraw, Image

import os
import random
import tqdm


class Template_Generator(Midv):

    #__slots__ = ["_img_loader", "_classes", "_fake_template", "_transformations","_fake_img_loader","_annotations_path","_imgs_path","_delta_boundary","_static_path", "_flag"]

    def __init__(self, absolute_path:str, fake_template:dict = None, delta_boundary:int=10):


        """
            The initialitation of the Generator just create the structure with Images in memory that will serve as a template to create the 
            new information. 
        """
        self._static_path_images = "MIDV2020/dataset/templates/images"
        self._static_path_annotations = "MIDV2020/dataset/templates/annotations"
        
        if fake_template is None:
            self._fake_template = super().MetaData

        #assert isinstance(fake_template, dict), "The metadata template to save the information must be a dict"


        super().__init__(absolute_path)

        path_template = super().get_template_path()
        path_annotatons = super().get_img_annotations_path()
        self._delta_boundary = delta_boundary
        self._annotations_path = os.path.join(path_annotatons,"reals")
        self._imgs_path = os.path.join(path_template, "reals")

        self._img_loader = self.create_and_map_classes_imgs() 

        self.create_loader()


    def create_and_map_classes_imgs(self):
        map_class = {
            
        }
        for image in os.listdir(self._imgs_path):
            class_image = image.split("_")[0]
            map_class[class_image] = []
        return map_class
    
    def create_and_map_classes_annotations(self):
        map_annotation = {
            
        }
        for annotation in os.listdir(self._annotations_path):
            class_ann = annotation.split("_")[0]    
            map_annotation[class_ann] = os.path.join(self._annotations_path,annotation)                
        return map_annotation    
            
    def create_loader(self) -> List[object]:
        map_annotations = self.create_and_map_classes_annotations()

        for im in os.listdir(self._imgs_path):
            clas, original, id = im.split("_")
            name_img = im
            src_img = os.path.join(self._static_path_images, clas,original, id)
            img = read_img(os.path.join(self._imgs_path, im))

            class_template = read_json(map_annotations[clas])
            self._img_loader[clas].append(super(Template_Generator, self).Img(img,class_template,name_img,src_img))

    def create(self,sample) -> List[Image.Image]:
        
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
            for smpl in tqdm.tqdm(range(sample//len(self._img_loader))):

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


    def store_generated_dataset(self, path_store: Optional[str] = None):

        if path_store is None:
            path = self.absoulute_path+"/SIDTD_Generated"
        else:
            path = path_store

        print(f"Data beeing stored in the path: {path}")
        store(self._fake_img_loader, path_store=path)




if __name__ == "__main__":
    gen = Template_Generator("/home/cboned/Midv2020/dataset/SIDTD")
    
    gen.create(5)
    
    gen.store_generated_dataset()