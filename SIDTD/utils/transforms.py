from .util import *
from typing import *
from PIL import ImageFont, ImageDraw, Image


import random
import copy
import cv2
import names
from faker import Faker


import numpy as np


def inpaint_image(img: np.ndarray, coord:np.ndarray, mask: np.ndarray, text_str: str):
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


def crop_replace(im_a:np.ndarray, im_b:np.ndarray, coord_a:np.ndarray, H:np.matrix, dx1:int, dy1:int, dx2:int, dy2:int):

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





# Functions for forgery on-the-fly
def copy_paste(image:np.ndarray, coord_a:List[int], coord_b:List[int], shift_copy:int) -> Tuple[np.ndarray, bool]:
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
    r_noise = random.randint(5, shift_copy)

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

def copy_paste_on_document(im_a, coord_a, coord_b, shift_copy):

    im_rep = copy.deepcopy(im_a)
    r_noise = random.randint(5,shift_copy)
    
    x1, y1, w1, h1 = coord_a['x'], coord_a['y'], coord_a['width'], coord_a['height']
    source = im_a[y1:y1+h1, x1:x1+w1]
    
    x2, y2, w2, h2 = coord_b['x'], coord_b['y'], coord_b['width'], coord_b['height']
    
    try:
        im_rep[y2 + r_noise:y2 + r_noise + h1, x2 + r_noise:x2 + r_noise + w1] = source
        dim_issue = False
    except:
        dim_issue = True
        print('COPY PASTE ERROR')

    
    return im_rep, dim_issue



def copy_paste_on_two_documents(im_a, coord_a, im_b, coord_b, shift_crop):

    im_rep = copy.deepcopy(im_a)
    r_noise = random.randint(5,shift_crop)
    
    x1, y1, w1, h1 = coord_b['x'], coord_b['y'], coord_b['width'], coord_b['height']
    #x1, y1, w1, h1 = bbox_info(coord_b)
    source = im_b[y1:y1+h1, x1:x1+w1]
    
    x2, y2, w2, h2 = coord_a['x'], coord_a['y'], coord_a['width'], coord_a['height']
    #x2, y2, w2, h2 = bbox_info(coord_a)
    source = cv2.resize(source, (w2,h2))
    
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


def CropReplace(image, annotations, image_target, annotations_target, list_image_field, shift_crop):
    """Copy a text randomly chosen among the field available and Paste in a random text field area.

    Args:
        prob (float): probability to perform CopyPaste. Must be between 0 and 1.
    """


    field_to_change = random.choice(list_image_field)
    info_source = annotations[field_to_change]
    info_target = annotations_target[field_to_change]
    img_tr, dim_issue = copy_paste_on_two_documents(image, info_source, image_target, info_target, shift_crop)
    return img_tr, dim_issue


def Inpainting(image, annotations):
    """Copy a text randomly chosen among the field available and Paste in a random text field area.

    Args:
        prob (float): probability to perform CopyPaste. Must be between 0 and 1.
    """

    
    list_iso_eur = ['GER', 'AUT', 'BEL', 'BGR', 'CYP', 'HRV', 'DAN', 'ESP', 'EST', 'FIN', 'FRA', 'GRC', 'HUN', 'IRL', 'ITA', 'LVA', 'LTU', 'LUX', 'MLT', 'NLD', 'POL', 'PRT', 'CZE', 'ROU', 'SVK', 'SVN', 'SWE']
    field_to_change = random.choice(['firstNames', 'lastNames', 'birthDate', 'gender', 'nationality'])
    if field_to_change == 'firstNames':
        text_str = names.get_first_name()
    elif field_to_change == 'lastNames':
        text_str = names.get_last_name()
    elif field_to_change == 'gender':
        text_str = random.choice(['F','M'])
    elif field_to_change == 'nationality':
        text_str = random.choice(list_iso_eur)
    elif field_to_change == 'birthDate':
        fake = Faker()
        t = fake.date_time_between(start_date='-70y', end_date='-18y')
        text_str = t.strftime('%d %m %Y')
    
    swap_info = annotations[field_to_change]
    coord = swap_info['x'], swap_info['y'], swap_info['width'], swap_info['height']
    shape_quad = bbox_to_coord(coord[0], coord[1], coord[2], coord[3])
    mask, _ = mask_from_info(image, shape_quad)
    img_tr = inpaint_image(img = image, coord = coord, mask = mask, text_str = text_str)
    return img_tr

def forgery_augmentation(dataset_name, image, list_path_img, path_img: str, shift_copy: int):

    """Perform randomly a forgery among the available list of falsification: copy-paste, crop & replace and inpainting .

    Args:
        dataset_name: dataset name. Use to find annotations folder.
        list_path_img: list of image paths. One image from that list will be draw in order to perform crop & replace.
        path_img: image path of the chosen document to forge.
        shift_copy: shift perform on pasting in order to not create a perfect forgery. 
    """
        
    l_fake_type = ['crop', 'inpainting', 'copy']
    image = Image.open(path_img).convert("RGB")
    id_img = path_img.split('/')[-1][:-4]
    path_json = os.getcwd() +  '/split_kfold/{}/annotations/annotations_{}.json'.format(dataset_name, id_img)    
    annotations = read_json(path_json)   # read json with document annotations of fields area
    list_fields = list(annotations.keys())
    list_image_field = ['photo', 'signature']
    if "photo" not in list_fields:
        list_image_field.remove('photo')
    if "signature" not in list_fields:
        list_image_field.remove('signature')
    if len(list_image_field) == 0:
        l_fake_type.remove('crop')

    fake_type = random.choice(l_fake_type)   # randomly draw one forgery techniques among: copy paste, crope & replace and inpainting
        
    # perform copy pasting
    if fake_type == 'copy':
        image = CopyPaste(np.asarray(image), annotations, shift_copy)

    # perform inpainting
    if fake_type == 'inpainting':
        image = Inpainting(np.asarray(image), annotations)

    # perform crop & replace
    if fake_type == 'crop':
                        
        # Loop until crop & replace does not create dimension issue
        dim_issue = True
        while dim_issue:
            img_path_clips_target = random.choice(list_path_img)    # choose a document to crop the signature or a photo
            id_img_target = img_path_clips_target.split('/')[-1][:-4]
            image_target = Image.open(img_path_clips_target).convert("RGB")   # read document where a signature or a photo will be cropped to be paste on current document
            path = os.getcwd() +  '/split_kfold/{}/annotations/annotations_{}.json'.format(dataset_name, id_img_target)  
            annotations_target = read_json(path)
            list_fields_target = list(annotations_target.keys())
            if ('signature' in list_image_field) and ('signature' not in list_fields_target):
                list_image_field.remove('signature')
            if ('photo' in list_image_field) and ('photo' not in list_fields_target):
                list_image_field.remove('photo')
            
            if len(list_image_field) == 0:
                dim_issue = True
            else:
                image, dim_issue = CropReplace(np.asarray(image), annotations, np.asarray(image_target), annotations_target, list_image_field, shift_copy)
        
    return image