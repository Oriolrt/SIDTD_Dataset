# To generate the crop and replace and the inpainting example we wiill call the MIDV() constructor
from Fake_Generator.Fake_Loader.Midv import *
from Fake_Generator.Fake_Loader import utils
from typing import *
from PIL import Image

def make_inpaint(img:np.ndarray, annotations:dict, img_id_annotation:int=None,mark:str=None, force_flag:int=1):
    """_summary_

    Args:
        img (np.ndarray): The img which will be inpainted
        annotations (dict): the dictionary of the image with the metadata of the Bbox as descripted in the Midv500/Midv2020
        img_id_annotation (int, optional): Default None, just change the value when working with the midv2020 annotations
        mark (str, optional): If you want to change concrete field.
        force_flag (int, optional): This flag will tell the code if the type of the annotations came from midv2020 (1) or midv500(0)
    """
    constructor= Midv(path=None)
    
    fake_image, filed_changed = constructor.Inpaint_and_Rewrite(img=img, info=annotations, img_id=img_id_annotation, mark=mark)
    
    return fake_image, filed_changed


def make_crop_and_reaplace(img1:np.ndarray, img2:np.ndarray, info:dict, additional_info:dict, img_id1:int=None, img_id2:int=None, delta1:list=[2,2], delta2:list = [2,2], mark:str=None) -> Tuple[Image.Image, Image.Image, Str, Str]:
    """_summary_

    Args:
        img1 (np.ndarray): The src image which will change the information
        img2 (np.ndarray): The target img that will change the information
        info (dict): the dictionary of the image with the metadata of the Bbox as descripted in the Midv500/Midv2020
        additional_info (dict): In case we work with different src of image (Mixing midv500 and midv2020)
        img_id1 (int, optional): Index of the src img (case Midv2020)
        img_id2 (int, optional): Index of the target img (case Midv2020)
        delta1 (list, optional): shift of the pixels result src img
        delta2 (list, optional): shift of the pixels result target img
        mark (str, optional): If you want to replace concrete field.
        force_flag (int, optional): This flag will tell the code if the type of the annotations came from midv2020 (1) or midv500(0)
                                    if force_flag == 0 additional info must be supplied.


    Returns:
        Tuple[Image.Image, Image.Image, Str, Str]: return the two images and the fields that have been replaced
    """
    constructor= Midv(path=None)
    
    fake_img1, fake_img2, filed_changed1, field_changed2 = constructor.Crop_and_Replace(img1=img1, img2=img2, info=info, additional_info=additional_info, img_id1=img_id1, img_id2=img_id2, delta1=delta1, delta2=delta2, mark=mark)

    return fake_img1, fake_img2, filed_changed1, field_changed2


def custom_crop_and_replace(img:np.ndarray, img2:np.ndarray, info:dict, info2:dict, delta1:list, delta2:list):
    """In case you want to create your own crop and replace with your own images you can do it, just make sure that the 
    info with the coordenades must be as follows:
         
        {
            "quad": [ [0, 0], [111, 0],
                    [111, 222], [0, 222] ]
        }
        
    Args:
        img (_type_): src img
        img2 (_type_): target img
        info (_type_): src dict metadata with the bbox as a value and "quad" as a key.
        info2 (_type_): trg dict metadata with the bbox as a value and "quad" as a key.
        delta1 (list, optional): shift of the pixels result src img
        delta2 (list, optional): shift of the pixels result target img
    """

    img_generated1, img_generated2 = utils.replace_info_documents(img, img2, info, info2, delta1=delta1, delta2=delta2, flag=0, mixed=False)
    
    return img_generated1, img_generated2

def custom_inpaint_and_rewrite(img: np.ndarray, info: dict, text_str:str,shaped:bool=False):
    
    """In case you want to create your own inpaint with your own images you can do it, just make sure that the 
    info with the coordenades must be as follows:

            {
            "quad": [ [0, 0], [111, 0],
                    [111, 222], [0, 222] ]
        }
        
    or in case you put shaped = True the list must be the bbox with the cordenates as follows
    
    [ [x_0, y_0], [x_1, y_1], [x_2, y_2], [x_3, y_3] ]
    
    
    Args:
        img (np.ndarray): The img which will be inpainted
        info (dict): the dictionary of the image with the metadata of the Bbox described
        shaped: in case shaped==True THe code will assume that the shape is List[List, List, List, List] with the coordenades of the Bbox as described
        text_str: The original text that you want to inpaint
        
    Returns:
        the image inpainted
    """
    
    mask, img_masked = mask_from_info(img, info, shaped=shaped)
    inpaint = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    fake_text_image = copy.deepcopy(inpaint)
    x0, y0, w, h = bbox_info(info, flag=0, shaped=shaped)
    

    color = (0,0,0)
    font = get_optimal_font_scale(text_str, w)

    img_pil = Image.fromarray(fake_text_image)
    draw = ImageDraw.Draw(img_pil)
    draw.text(((x0, y0)), text_str, font=font, fill=color)
    fake_text_image = np.array(img_pil)

    return fake_text_image