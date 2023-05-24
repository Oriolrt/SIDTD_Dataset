from .transforms import *

from importlib.resources import path
from typing import *
from PIL import ImageFont, ImageDraw, Image

import json
import random
import cv2
import os
import imageio

import numpy as np

# TODO This function need to be refactored (generalize the fonts)
def get_optimal_font_scale(text, width):
    fontsize = 1  # starting font size
    sel_font =  get_font_scale()  # "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    stop  = False  # portion of image width you want text width to be
    img_fraction = 1
    try:
        font = ImageFont.truetype(font=sel_font, size=fontsize ,encoding="unic")
    except:
        sel_font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = ImageFont.truetype(font=sel_font, size=fontsize ,encoding="unic")

    while (font.getsize(text)[0] < img_fraction*width) and (stop == False):
        # iterate until the text size is just larger than the criteria
        if font.getsize(text)[0] == 0:
            sel_font =  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

            if font.getsize(text)[1] == 0:
                stop = True
                break

        fontsize += 1
        font = ImageFont.truetype(sel_font, fontsize ,encoding="unic")

    # optionally de-increment to be sure it is less than criteria
    fontsize -= 1
    font = ImageFont.truetype(sel_font, fontsize ,encoding="unic")

    return font


def get_font_scale(inner_path: str = os.getcwd() + "/TTF"):
    deja = [i for i in os.listdir(inner_path) if "DejaVu" in i]

    selected = random.choice(deja)

    return os.path.join(inner_path, selected)




def mask_from_info(img:np.ndarray, shape:np.ndarray):

    """"
        This f(x) extract the  ROI that will be inpainted

    """
    def midpoint(x1, y1, x2, y2):
        x_mid = int((x1 + x2) / 2)
        y_mid = int((y1 + y2) / 2)
        return (x_mid, y_mid)

    x0, x1, x2, x3 = shape[0][0], shape[1][0], shape[2][0], shape[3][0]
    y0, y1, y2, y3 = shape[0][1], shape[1][1], shape[2][1], shape[3][1]


    xmid0, ymid0 = midpoint(x1, y1, x2, y2)
    xmid1, ymid1 = midpoint(x0, y0, x3, y3)

    thickness = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.line(mask, (xmid0, ymid0), (xmid1, ymid1), 255, thickness)

    masked = cv2.bitwise_and(img, img, mask=mask)

    return mask, masked


def read_img(path: str):
    img = np.array(imageio.imread(path))

    if img.shape[-1] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        return img

def read_json(path: str):
    with open(path) as f:
        return json.load(f)

def write_json(data:dict, path:str, name:str=None):
    if name is None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    else:
        path_to_save = os.path.join(path,name+".json")
        with open(path_to_save, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)


def store(img_loader: list, path_store: str = None):
    if path_store is None:
        path_store = os.path.join(os.getcwd(), "NeoSIDTD")

    advisor = len(img_loader) // 10
    for idx, image in enumerate(img_loader):
        class_name = image._name.split("_")[0]
        joined_path = os.path.join(path_store, class_name)

        if os.path.exists(joined_path):
            path_to_save = os.path.join(joined_path, image.fake_name + ".jpg")
            imageio.imwrite(path_to_save, image.fake_img)
            write_json(image.fake_meta, joined_path, image.fake_name)

        else:
            os.makedirs(joined_path)
            path_to_save = os.path.join(joined_path, image.fake_name + ".jpg")

            imageio.imwrite(path_to_save, image.fake_img)
            write_json(image.fake_meta, joined_path, image.fake_name)

        if (idx % advisor) == 0:
            print(f"{(idx // advisor) * 10} % of the dataset stored")

    print("Data Successfuly stored")

def bbox_to_coord(x, y, w, h):
    """This function convert the kin of the shape from bbox rectangle x0,y0 + heigh and weight to the polygon coordenades.

    Returns:
        _type_: _description_
    """
    x_f = x + w
    y_f = y + h

    c1, c2, c3, c4 = [x, y], [x_f, y], [x_f, y_f], [x, y_f]

    return [c1, c2, c3, c4]


def bbox_info(info) -> Tuple[int,...]:
    """This function return the rectangle of the template where are in located,

    Returns:
        Tuple[Int, Int, Int, Int]
    """

    try:
        shape = info["quad"]  # here if the info is like [[x0,y0], [x1,y1]...] #Here if the info is x = [x0,x1,x2,x3] and y = [y0,y1,y2,y3]
        x0, x1, x2, x3 = shape[0][0], shape[1][0], shape[2][0], shape[3][0]
        y0, y1, y2, y3 = shape[0][1], shape[1][1], shape[2][1], shape[3][1]

        w = np.max([x0, x1, x2, x3]) - np.min([x0, x1, x2, x3])
        h = np.max([y0, y1, y2, y3]) - np.min([y0, y1, y2, y3])

    except:
        shape = info["shape_attributes"]
        x0 = shape["x"]
        y0 = shape["y"]
        w = shape["width"]
        h = shape["height"]



    return x0, y0, w, h


def replace_info_documents(im0:np.ndarray, im1:np.ndarray, data0:dict, data1:dict, delta1:np.ndarray, delta2:np.ndarray):

    """
    This function replaces a rectangular region of interest (ROI) in one image with a homography-transformed ROI from another image.

    Args:
        im0 (np.ndarray): The first input image as a NumPy array.
        im1 (np.ndarray): The second input image as a NumPy array.
        data0 (dict): A dictionary containing information about the first image's ROI.
        data1 (dict): A dictionary containing information about the second image's ROI.
        delta1 (np.ndarray[float, ...]): An array containing the x and y translation values for the first ROI.
        delta2 (np.ndarray[float, ...]): An array containing the x and y translation values for the second ROI.

    Returns:
        Tuple[np.ndarray, bool]: Returns a tuple containing the resulting image after replacing the ROI and a boolean indicator specifying if there was a dimension issue or not.
    """
    # take the coordinates of the region to replace
    x, y, w, h = bbox_info(data0)
    x2, y2, w2, h2 = bbox_info(data1)

    coord0 = np.array(bbox_to_coord(x, y, w, h), dtype=np.float32())
    coord1 = np.array(bbox_to_coord(x2, y2, w2, h2), dtype=np.float32())

    H, _ = compute_homography(coord0, coord1)

    dx1,dy1 = delta1
    dx2,dy2 = delta2

    im_rep, dim_issue = crop_replace(im1, im0, coord1, H, dx1,dy1,dx2,dy2)

    return im_rep, dim_issue

# TODO Refactor this function in order not to pass as input the json info (only coordenates)
def compute_homography(coord0:np.ndarray, coord1:np.ndarray):

    # Homography: p1 = H@p0 | p0 = inv(H)@p1
    # (where @ denotes matricial product, inv the inverse of the homography
    # and p0 and p1 are homogeneous coordinates for the pixel coordinates)
    H, mask = cv2.findHomography(coord1, coord0, cv2.RANSAC, 1.0)

    return H, mask
