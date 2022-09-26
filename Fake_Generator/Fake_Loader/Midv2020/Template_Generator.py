from ast import Str
import copy
from typing import *
from unicodedata import name
from Fake_Generator.Fake_Loader.Midv  import Midv
from Fake_Generator.Fake_Loader.utils import *
import os
import numpy as np
import random
import cv2
import tqdm
from PIL import ImageFont, ImageDraw, Image


class Template_Generator(Midv):

    __slots__ = ["_img_loader", "_classes", "_fake_metadata", "_transformations","_fake_img_loader","_annotations_path","_imgs_path","_delta_boundary","_static_path"]

    def __init__(self, absolute_path:str ,fake_template = None, delta_boundary:int=10):


        """
            The initialitation of the Generator just create the structure with Images in memory that will serve as a template to create the 
            new information. 
        """

        if fake_template is None:
            fake_template = {
                            "name": "None",
                            "ctype": "None",
                            "loader": "None",
                            "shift": "None",
                            "src": "None",
                            "second_src": "None",
                            "field": "None",
                            "second_field": "None"
                            }

        assert isinstance(fake_template, dict), "The metadata template to save the information must be a dict"


        super().__init__(absolute_path,fake_template)

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
                src_img = os.path.join(self._static_path,ninf[-1],im)
                img = read_img(src_img)

                self._img_loader[ninf[-1]].append(super(Template_Generator, self).Img(img,class_template,name_img,src_img))


    def Crop_and_Replace(self, img1: np.ndarray, img1_id: int, img2: np.ndarray, img2_id: int, info: dict,
                         delta1: list = [2, 2], delta2: list = [2, 2])-> Tuple[Image.Image, Image.Image, Str, Str]: 

        selected1 = list(info["_via_img_metadata"])[img1_id]
        selected2 = list(info["_via_img_metadata"])[img2_id]
        fields1 = info["_via_img_metadata"][selected1]["regions"]
        fields2 = info["_via_img_metadata"][selected2]["regions"]

        sInfo = True if np.random.randint(100) > 5 else False

        # (2-12) to avoid the face, the photo and the signature
        if sInfo and (len(fields1) == len(fields2)):
            field_to_change1 = field_to_change2 = random.randint(2, len((fields1))-2)

        else:
            field_to_change1 = random.randint(2, len((fields1))-2)
            field_to_change2 = random.randint(2, len((fields2))-2)
            while field_to_change2 == field_to_change1:
                field_to_change2 = random.randint(2, 12)

        fake_document1, fake_document2 = replace_info_documents_2020(img1, img2, fields1[field_to_change1],
                                                                     fields2[field_to_change2], delta1, delta2)

        return fake_document1, fake_document2, fields1[field_to_change1]["region_attributes"]["field_name"], fields2[field_to_change2]["region_attributes"]["field_name"]

    def Inpaint_and_Rewrite(self, img: np.ndarray, img_id: int, info: dict, mark=False) -> Tuple[Image.Image, Str]:

        selected = list(info["_via_img_metadata"])[img_id]
        fields = info["_via_img_metadata"][selected]["regions"]

        if not mark:

            # (2-12) to avoid the face, the photo and the signature
            field_to_change = random.randint(2, len((fields))-2)
            swap_info = fields[field_to_change]
        #mark if we want to inpaint certain field
        else:
            swap_info = fields[mark]

        mask, img_masked = mask_from_info(img, swap_info)

        inpaint = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

        fake_text_image = copy.deepcopy(inpaint)


        x0, y0, w, h = bbox_info(swap_info)

        text_str = swap_info["region_attributes"]["value"]

        color = (0, 0, 0)

        font = get_optimal_font_scale(text_str, w)

        img_pil = Image.fromarray(fake_text_image)
        draw = ImageDraw.Draw(img_pil)
        draw.text(((x0, y0)), text_str, font=font, fill=color)
        #+h
        fake_text_image = np.array(img_pil)

        return fake_text_image, swap_info["region_attributes"]["field_name"]


#todo mirar lo del counter per afegir al nom
    def fit(self,sample) -> List[Image.Image]:
        #genereate and inpainting for each img of the true loader

        for counter, (key,img_bucket) in enumerate(self._img_loader.items()):
            for idx in range(len(img_bucket)):

                img = random.choice(img_bucket)
                img_id = int(img._relative_path.split("/")[-1].split(".")[0])
                fake_img, field =  self.Inpaint_and_Rewrite(img._img,img_id,img._meta)
                name_fake_generated =  img._name.split(".")[0] + "_fake_" + str(counter) + "_" + str(idx)

                #Creating the dict with the metadata
                fake_meta = vars(super().MetaData(src=img._relative_path, type_transformation="Inpaint_and_Rewrite",field=field,loader="Midv2020",name=name_fake_generated))

                generated_img = super().Img(img._img, img._meta, img._name)

                generated_img.fake_meta = fake_meta
                generated_img.fake_name = fake_meta["name"]
                generated_img.fake_img = fake_img

                self._fake_img_loader.append(generated_img)
            
            print(f"General Inpainted for the class {key} Done")
            # equal generation
            for smpl in tqdm.tqdm(range(sample//len(self._classes))):

                fake_meta = copy.copy(self._fake_metadata)
                img = random.choice(img_bucket)
                img_id = int(img._relative_path.split("/")[-1].split(".")[0])

                name_fake_generated =  img._name.split(".")[0] + "_fake_" + str(counter) + "_" + str(idx)

                transformation = random.choice(self._transformations)
                name_transform = (transformation.__name__).split(".")[-1]
                if name_transform == "Inpaint_and_Rewrite":
                    fake_img, field = transformation(img._img, img_id,img._meta)


                    fake_meta = vars(super().MetaData(src=img._relative_path, type_transformation=name_transform,field=field,loader="Midv2020",name=name_fake_generated))


                    # craeting fake img
                    generated_img = super().Img(img._img, img._meta, img._name)

                    generated_img.fake_meta = fake_meta
                    generated_img.fake_name = fake_meta["name"]
                    generated_img.fake_img = fake_img

                    self._fake_img_loader.append(generated_img)

                else:

                    delta1 = random.sample(range(self._delta_boundary),2)
                    delta2 = random.sample(range(self._delta_boundary),2)
                    fake_meta2 =  copy.copy(fake_meta)

                    img2 = random.choice(img_bucket)
                    img_id2 = int(img2._relative_path.split("/")[-1].split(".")[0])
                    fake_img1, fake_img2 , field1, field2 = transformation(img._img, img_id ,img2._img, img_id2 ,img._meta,delta1,delta2)
                    
                    #img1 info
                    name_fake_generated =  img._name.split(".")[0] + "_fake_" + str(counter) + "_" + str(sample + 1 +idx)

                    fake_meta = vars(super().MetaData(src=img._relative_path, second_src=img2._relative_path, shift=(delta1,delta2),type_transformation=name_transform,field=field, second_field=field2,loader="Midv2020",name=name_fake_generated))
            
                    # craeting fake img1
                    generated_img = super().Img(img._img, img._meta, img._name)

                    generated_img.fake_meta = fake_meta
                    generated_img.fake_name = fake_meta["name"]
                    generated_img.fake_img = fake_img1
                    generated_img.complement_img = fake_img2

                    self._fake_img_loader.append(generated_img)


                    #img2 info
                    name_fake_generated =  img._name.split(".")[0] + "_fake_" + str(counter) + "_" + str(sample + 2 +idx)
                    fake_meta = vars(super().MetaData(second_src=img._relative_path, src=img2._relative_path, shift=(delta1,delta2),type_transformation=name_transform,field2=field, field=field2,loader="Midv2020",name=name_fake_generated))
                    

                    # craeting fake img2
                    generated_img2 = super().Img(img2._img, img2._meta, img2._name)

                    generated_img2.fake_meta = fake_meta2
                    generated_img2.fake_name = fake_meta2["name"]
                    generated_img2.fake_img = fake_img2
                    generated_img2.complement_img = fake_img1

                    self._fake_img_loader.append(generated_img2)


    def store_generated_dataset(self):
        store(self._fake_img_loader)
