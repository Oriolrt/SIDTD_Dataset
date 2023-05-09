from .util import *
from typing import *
from PIL import ImageFont, ImageDraw, Image


import random
import copy
import cv2


import numpy as np


def inpaint_image(img: np.ndarray, coord:np.ndarray[int, ...], mask: np.ndarray, text_str: str):
    """
    Inpaints the masked region in the input image using the TELEA algorithm and adds text to it.

    Args:
        img (np.ndarray): Input image.
        coord (np.ndarray[int, ...]): An array of integers representing the (x,y) coordinates of the top-left corner,
            as well as the width and height of the region where the text will be added.
        mask (np.ndarray): A binary mask with the same shape as `img`, where the masked pixels have value 0.
        text_str (str): The text to be added to the inpainted region.

    Returns:
        np.ndarray: A numpy array representing the inpainted image with the text added to it.
    """

    inpaint = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    fake_text_image = copy.deepcopy(inpaint)
    x0, y0, w, h = coord

    color = (0, 0, 0)
    font = get_optimal_font_scale(text_str, w)

    img_pil = Image.fromarray(fake_text_image)
    draw = ImageDraw.Draw(img_pil)
    draw.text(((x0, y0)), text_str, font=font, fill=color)
    fake_text_image = np.array(img_pil)

    return fake_text_image




def crop_replace(im_a:np.ndarray, im_b:np.ndarray, coord_a:np.ndarray[int,...], H:np.matrix, dx1:int, dy1:int, dx2:int, dy2:int):

    """
    Crop and replace a region from one image onto another using a homography matrix.

    Args:
        im_a (numpy.ndarray): The source image to crop the region from.
        im_b (numpy.ndarray): The target image to replace the region in.
        coord_a (numpy.ndarray): The coordinates of the region to crop from im_a, in the form of a 3xN array.
        H (numpy.matrix): The homography matrix used to warp the coordinates of the region from im_a to im_b.
        dx1 (int): The horizontal shift applied to the region after it is pasted onto im_b.
        dy1 (int): The vertical shift applied to the region after it is pasted onto im_b.
        dx2 (int): The horizontal shift applied to the region before it is pasted onto im_b.
        dy2 (int): The vertical shift applied to the region before it is pasted onto im_b.

    Returns:
        A tuple containing the resulting image with the region replaced, and a boolean value indicating if there was an issue with the dimensions.
    """
    dim_issue = False
    mask_a = np.zeros_like(im_a)
    cv2.drawContours(mask_a, [coord_a.astype(int)], -1, color=(255, 255, 255), thickness=cv2.FILLED)
    y_a, x_a = np.where((mask_a[..., 0] / 255).astype(int) == 1)
    coordh_a = np.ones((3, len(x_a)), dtype=np.float32())
    coordh_a[0, :] = x_a
    coordh_a[1, :] = y_a

    coordh_b = H @ coordh_a
    coordh_b = coordh_b / coordh_b[-1, ...]

    x_b = coordh_b.T[:, 0].astype(int)
    y_b = coordh_b.T[:, 1].astype(int)

    im_rep = copy.deepcopy(im_b)

    try :
        im_rep[y_b +  dy1, x_b + dx1, ...] = im_a[y_a + dy2, x_a + dx2, ...]
    except:
        print('ERROR SHAPE CROP REPLACE')
        dim_issue = True

    return im_rep, dim_issue





# TODO FUnció Refactoritzada falta explicar que fa la funció
def copy_paste(image:np.ndarray, coord_a:List[int, ...], coord_b:List[int, ...], shift_copy:int) -> Tuple[np.ndarray, bool]:
    """
    This function performs a deep copy of an input image and pastes a region of interest (ROI) at a specific position in the output image.

    Args:
        image (np.ndarray): The input image as a NumPy array.
        coord_a (List[int, ...]): Coordinates of the rectangle containing the ROI. Expected to be a list with four integer values representing x, y, width, and height coordinates of the ROI respectively.
        coord_b (List[int, ...]): Coordinates of the rectangle in the output image where the ROI will be pasted. Expected to be a list with four integer values representing x, y, width, and height coordinates of the rectangle respectively.
        shift_copy (int): Integer value that will be used as a shift for copying the ROI.

    Returns:
        Tuple[np.ndarray, bool]: Returns a tuple containing the resulting image after pasting the ROI and a boolean indicator specifying if there was a dimension issue or not.
    """
    im_rep = copy.deepcopy(image)
    r_noise = random.randint(10, shift_copy)

    x1, y1, w1, h1 = coord_a
    source = image[y1:y1 + h1, x1:x1 + w1]
    x2, y2, w2, h2 = coord_b

    try:
        im_rep[y2 + r_noise:y2 + r_noise + h1, x2 + r_noise:x2 + r_noise + w1] = source
        dim_issue = False
    except:
        dim_issue = True
        print('COPY PASTE ERROR')

    return im_rep, dim_issue


#### AIXÒ ho he de discutir amb en maxime
#TODO Aquesta funció no s'utilitza en ningun lloc
def copy_paste_on_two_documents(image_a:np.ndarray, image_b:np.ndarray, coord_a:List[int, ...], coord_b:List[int, ...], shift_crop:int) -> Tuple[np.ndarray, bool]:
    im_rep = copy.deepcopy(image_a)
    r_noise = random.randint(10, shift_crop)

    x1, y1, w1, h1 = bbox_info(coord_b)
    source = image_b[y1:y1 + h1, x1:x1 + w1]

    x2, y2, w2, h2 = bbox_info(coord_a)
    source = cv2.resize(source, (w2, h2))

    try:
        im_rep[y2 + r_noise:y2 + r_noise + h2, x2 + r_noise:x2 + r_noise + w2] = source
        dim_issue = False
    except:
        dim_issue = True

    return im_rep, dim_issue


def copy_paste_on_document(image:np.ndarray, annotations:dict, shift_copy):
    """Copy a text randomly chosen among the field available and Paste in a random text field area.

    Args:
        prob (float): probability to perform CopyPaste. Must be between 0 and 1.
    """

    list_text_field = list(annotations.keys())
    if 'image' in list_text_field:
        list_text_field.remove('image')
    if 'photo' in list_text_field:
        list_text_field.remove('photo')
    if 'signature' in list_text_field:
        list_text_field.remove('signature')
    if 'face' in list_text_field:
        list_text_field.remove('face')

    dim_issue = True
    while dim_issue:
        source_field_to_change_txt = random.choice(list_text_field)
        target_field_to_change_txt = random.choice(list_text_field)
        source_info_txt = annotations[source_field_to_change_txt]
        target_info_txt = annotations[target_field_to_change_txt]
        img_tr, dim_issue = copy_paste(image, source_info_txt, target_info_txt, shift_copy)

    return img_tr


# TODO Refactor this Function és una "guarrada"
"""
def Inpainting(image, annotations, id_country):

    Copy a text randomly chosen among the field available and Paste in a random text field area.

    Args:
        prob (float): probability to perform CopyPaste. Must be between 0 and 1.
        
    list_text_field = list(annotations.keys())
    if 'image' in list_text_field:
        list_text_field.remove('image')
    if 'photo' in list_text_field:
        list_text_field.remove('photo')
    if 'signature' in list_text_field:
        list_text_field.remove('signature')
    if 'face' in list_text_field:
        list_text_field.remove('face')
    field_to_change = random.choice(list_text_field)

    if field_to_change == 'name':
        text_str = names.get_first_name()
    elif field_to_change == 'surname':
        text_str = names.get_last_name()
    elif field_to_change == 'sex':
        if id_country in ['esp', 'alb', 'fin', 'grc', 'svk']:
            text_str = random.choice(['F', 'M'])
        else:
            text_str = random.choice(['K/M', 'N/F'])
    elif field_to_change == 'nationality':
        if id_country == 'esp':
            text_str = 'ESP'
        elif id_country == 'alb':
            text_str = 'Shqiptare/Albanian'
        elif id_country == 'aze':
            text_str = 'AZORBAYCA/AZERBAIJAN'
        elif id_country == 'est':
            text_str = 'EST'
        elif id_country == 'fin':
            text_str = 'FIN'
        elif id_country == 'grc':
            text_str = 'EAAHNIKH/HELLENIC'
        elif id_country == 'lva':
            text_str = 'LVA'
        elif id_country == 'rus':
            text_str = 'AOMNHNKA'
        elif id_country == 'srb':
            text_str = 'SRPSKO'
        elif id_country == 'svk':
            text_str = 'SVK'
    elif field_to_change == 'birthdate':
        fake = Faker() # TODO Això no se que pues és
        t = fake.date_time_between(start_date='-60y', end_date='-18y')
        text_str = t.strftime('%d %m %Y')

    swap_info = annotations[field_to_change]
    img_tr = inpaint_image(img=image, swap_info=swap_info, text_str=text_str)
    return img_tr
"""