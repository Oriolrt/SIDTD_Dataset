from Fake_Loader.Midv  import Midv
from ..utils import *
import os
import numpy as np
import random
import cv2



class Template_Generator(Midv):
    __slots__ = ["_img_loader", "_classes", "_fake_metadata", "_transformations","_fake_img_loader"]

    def __init__(self, absolute_path:str ,sample:int, fake_template:dict):

        super().__init__(absolute_path,fake_template)

        self._classes = list(map(lambda class_folder: os.path.join(absolute_path, class_folder),
                                 os.listdir(absolute_path)))


        self.create_loader()

        self.fit(sample)


    # TODO: Carlos, this code is the same that in the generate_crop_and_replace function from the generate_fake_test.py file. It must be a single function in the transforms.py file
    def Crop_and_Replace(self, img1, img2, info1, info2, delta1, delta2):

        fields1 = list(np.unique(list(info1.keys())))
        fields2 = list(np.unique(list(info2.keys())))

        non_compare = {i:True for i in set(fields1) -  set(fields2)}

        dx1, dy1 = delta1
        dx2, dy2 = delta2

        # Hi ha dpcuments que no tenen signatura o fotos
        try:
            fields1.remove("photo")
            fields1.remove("signature")
        except:
            pass
        try:
            fields2.remove("photo")
            fields2.remove("signature")
        except:
            pass

        same_info = True if np.random.randint(100) > 5 else False
        if same_info:
            field_to_change1 = random.choice(fields1)
            field_to_change2 = field_to_change1 if non_compare.get(field_to_change1, None) is None else random.choice(fields2)
        else:
            field_to_change1 = random.choice(fields1)
            field_to_change2 = random.choice(fields2)

        fake_document1 , fake_document2 = replace_info_documents(img1,img2,info1[field_to_change1], info2[field_to_change2],dx1,dy1,dx2,dy2)


        return fake_document1, fake_document2, field_to_change1, field_to_change2

    # TODO: Carlos, this code is the same that in the generate_inpaint function from the generate_fake_test.py file. It must be a single function in the transforms.py file 
    def Inpaint_and_Rewrite(self,img: np.ndarray, info: dict):
        fields = list(np.unique(list(info.keys())))
        # The photo  and the signature will be treated different
        try:
            fields.remove("photo")
        except:
            pass
        try:
            fields.remove("signature")
        except:
            pass

        field_to_change = random.choice(fields)
        swap1_info = info[field_to_change]
        mask1, img_masked = mask_from_info(img, swap1_info["quad"])

        # inpainting to have a clear canvas
        inpaint = cv2.inpaint(img, mask1, 3, cv2.INPAINT_TELEA)

        x0, y0, w, h = midv500_bbox_info(swap1_info)
        text_str = swap1_info['value']  # use the same text as in the original
        color = (0, 0, 0)
        fontScale = get_optimal_font_scale(text_str, w)
        thickness = 1
        font = random.choice([cv2.FONT_HERSHEY_SIMPLEX,
                              cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                              cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                              cv2.FONT_HERSHEY_COMPLEX_SMALL,
                              cv2.FONT_HERSHEY_TRIPLEX,
                              cv2.FONT_HERSHEY_COMPLEX,
                              cv2.FONT_HERSHEY_DUPLEX,
                              cv2.FONT_HERSHEY_PLAIN])
        fake_text_image = copy.deepcopy(inpaint)
        fake_text_image = cv2.putText(fake_text_image, text_str, (x0, y0 + h), font, fontScale,
                                      color, thickness, cv2.LINE_AA)

        return fake_text_image, field_to_change

    # function to generate the loader of the images
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
            fake_meta = copy.copy(self._fake_metadata)
            fake_img, field = self.Inpaint_and_Rewrite(img._img, img._meta)

            fake_meta["ctype"] = "Inpaint_and_Rewrite"
            fake_meta["loader"] = "Midv500"
            fake_meta["src"] = img._relative_path
            fake_meta["field"] = field
            fake_meta["name"] = img._name + "_fake_" + str(counter)

            # craeting fake img
            generated_img = super().Img(img._img, img._meta, img._name)

            generated_img.fake_meta = fake_meta
            generated_img.fake_name = fake_meta["name"]
            generated_img.fake_img = fake_img

            self._fake_img_loader.append(generated_img)

        for smpl in range(sample):

            fake_meta = copy.copy(self._fake_metadata)
            img = random.choice(self._img_loader)
            transformation = random.choice(self._transformations)
            name_transform = (transformation.__name__).split(".")[-1]

            if name_transform == "Inpaint_and_Rewrite":
                fake_img, field = transformation(img._img, img._meta)

                fake_meta["ctype"] = name_transform
                fake_meta["loader"] = "Midv500"
                fake_meta["src"] = img._relative_path
                fake_meta["field"] = field
                fake_meta["name"] = img._name + "_fake_" + str(smpl + sample)

                #craeting fake img
                generated_img = super().Img(img._img, img._meta, img._name)

                generated_img.fake_meta = fake_meta
                generated_img.fake_name = fake_meta["name"]
                generated_img.fake_img = fake_img

                self._fake_img_loader.append(generated_img)

            else:
                img2 = random.choice(self._img_loader)
                fake_meta2 =  copy.copy(fake_meta)

                fake_img1, fake_img2 , field1, field2 = transformation(img._img, img2._img, img._meta, img2._meta,[0,0],[0,0])

                #img1 info
                fake_meta["ctype"] = name_transform
                fake_meta["loader"] = "Midv500"
                fake_meta["src"] = img._relative_path
                fake_meta["second_src"] = img2._relative_path
                fake_meta["field"] = field1
                fake_meta["second_field"] =  field2
                fake_meta["name"] = img._name + "_fake_" + str(smpl + sample)

                # craeting fake img1
                generated_img = super().Img(img._img, img._meta, img._name)

                generated_img.fake_meta = fake_meta
                generated_img.fake_name = fake_meta["name"]
                generated_img.fake_img = fake_img1
                generated_img.complement_img = fake_img2

                self._fake_img_loader.append(generated_img)


                #img2 info
                fake_meta2["ctype"] = name_transform
                fake_meta2["loader"] = "Midv500"
                fake_meta2["src"] = img2._relative_path
                fake_meta2["second_src"] = img._relative_path
                fake_meta2["field"] = field2
                fake_meta2["second_field"] =  field1
                fake_meta2["name"] = img2._name + "_fake_" + str(smpl + sample + 1)

                # craeting fake img1
                generated_img2 = super().Img(img2._img, img2._meta, img2._name)

                generated_img2.fake_meta = fake_meta2
                generated_img2.fake_name = fake_meta2["name"]
                generated_img2.fake_img = fake_img2
                generated_img2.complement_img = fake_img1

                self._fake_img_loader.append(generated_img2)
