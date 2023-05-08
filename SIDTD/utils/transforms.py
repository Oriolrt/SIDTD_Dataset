from util import *
from typing import *
from PIL import ImageFont, ImageDraw, Image


import random
import copy
import cv2


import numpy as np


def inpaint_image(img: np.ndarray, swap_info: dict, text_str: str):

    if text_str is None:
        text_str = swap_info["value"]

    mask, img_masked = mask_from_info(img, swap_info)
    inpaint = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    fake_text_image = copy.deepcopy(inpaint)
    x0, y0, w, h = bbox_info(swap_info)

    color = (0, 0, 0)
    font = get_optimal_font_scale(text_str, w)

    img_pil = Image.fromarray(fake_text_image)
    draw = ImageDraw.Draw(img_pil)
    draw.text(((x0, y0)), text_str, font=font, fill=color)
    fake_text_image = np.array(img_pil)

    return fake_text_image


# TODO Refactor this function in order not to pass as input the json info (only coordenates)
def compute_homography(info1, info2):
    # take the coordinates of the region to replace
    x, y, w, h = bbox_info(info1)
    x2, y2, w2, h2 = bbox_info(info2)

    coord0 = np.array(bbox_to_coord(x, y, w, h), dtype=np.float32())
    coord1 = np.array(bbox_to_coord(x2, y2, w2, h2), dtype=np.float32())

    # Homography: p1 = H@p0 | p0 = inv(H)@p1
    # (where @ denotes matricial product, inv the inverse of the homography
    # and p0 and p1 are homogeneous coordinates for the pixel coordinates)

    H, mask = cv2.findHomography(coord1, coord0, cv2.RANSAC, 1.0)

    return H, coord0, coord1, mask

def replace_one_document(im_a, im_b, coord_a, H,dx1,dy1,dx2,dy2):

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


# TODO check if this function is necessary and it is not redundant
def replace_info_documents(im0, im1, data0, data1, delta1, delta2):
    H, coord0, coord1, _ = compute_homography(data0, data1)

    dx1,dy1 = delta1
    dx2,dy2 = delta2

    im_rep, dim_issue = replace_one_document(im1, im0, coord1, H, dx1,dy1,dx2,dy2)

    return im_rep, dim_issue


# TODO refactor this function per que és una guarrada
# TODO also need to be refactored in order to not need to get into the metadata json
def copy_paste_on_document(im_a, coord_a, coord_b, shift_copy):
    im_rep = copy.deepcopy(im_a)
    r_noise = random.randint(10, shift_copy)

    x1, y1, w1, h1 = bbox_info(coord_a)
    source = im_a[y1:y1 + h1, x1:x1 + w1]

    x2, y2, w2, h2 = bbox_info(coord_b)

    try:
        im_rep[y2 + r_noise:y2 + r_noise + h1, x2 + r_noise:x2 + r_noise + w1] = source
        dim_issue = False
    except:
        dim_issue = True
        print('COPY PASTE ERROR')

    return im_rep, dim_issue

# TODO canviar aquesta funció també per que és una guarrada

def copy_paste_on_two_documents(im_a, coord_a, im_b, coord_b, shift_crop):
    im_rep = copy.deepcopy(im_a)
    r_noise = random.randint(10, shift_crop)

    x1, y1, w1, h1 = bbox_info(coord_b)
    source = im_b[y1:y1 + h1, x1:x1 + w1]

    x2, y2, w2, h2 = bbox_info(coord_a)
    source = cv2.resize(source, (w2, h2))

    try:
        im_rep[y2 + r_noise:y2 + r_noise + h2, x2 + r_noise:x2 + r_noise + w2] = source
        dim_issue = False
    except:
        dim_issue = True

    return im_rep, dim_issue


def CopyPaste(images, annotations, shift_copy):
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
        img_tr, dim_issue = copy_paste_on_document(images, source_info_txt, target_info_txt, shift_copy)

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