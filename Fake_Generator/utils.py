from importlib.resources import path
import json
from ntpath import join
import random
import copy
import sys
from typing import *
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import cv2
from tqdm import tqdm
import os
import imageio

import matplotlib.pyplot as plt
from traitlets import Int


def get_optimal_font_scale(text, width):
    fontsize = 1  # starting font size
    sel_font =  get_font_scale()#"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    stop  = False  # portion of image width you want text width to be
    img_fraction = 1
    try:
        font = ImageFont.truetype(font=sel_font, size=fontsize,encoding="unic")
    except:
        sel_font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = ImageFont.truetype(font=sel_font, size=fontsize,encoding="unic")

    while (font.getsize(text)[0] < img_fraction*width) and (stop == False):
        # iterate until the text size is just larger than the criteria
        if font.getsize(text)[0] == 0:
            sel_font =  "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
 
            if font.getsize(text)[1] == 0:
                stop = True
                break

        fontsize += 1
        font = ImageFont.truetype(sel_font, fontsize,encoding="unic")

    # optionally de-increment to be sure it is less than criteria
    fontsize -= 1
    font = ImageFont.truetype(sel_font, fontsize,encoding="unic")

    return font


def get_font_scale(inner_path: str = "/usr/share/fonts/TTF"):

    deja = [i for i in os.listdir(inner_path) if "DejaVu" in i]
    
    selected = random.choice(deja)

    return os.path.join(inner_path,selected)

def mask_from_info(img, shape:np.ndarray,flag:int=1,shaped:bool = False,  shaped_kin:str="rect"):
    def midpoint(x1, y1, x2, y2):
        x_mid = int((x1 + x2) / 2)
        y_mid = int((y1 + y2) / 2)
        return (x_mid, y_mid)

    if not shaped:
        x0, y0, w, h = bbox_info(shape, flag)
        shape = bbox_to_coord(x0, y0, w, h)

    x0, x1, x2, x3 = shape[0][0], shape[1][0], shape[2][0], shape[3][0] if shaped_kin =="rect" else shape[0], shape[1], shape[2], shape[3]
    y0, y1, y2, y3 = shape[0][1], shape[1][1], shape[2][1], shape[3][1] if shaped_kin =="rect" else shape[0], shape[1], shape[2], shape[3]

    xmid0, ymid0 = midpoint(x1, y1, x2, y2)
    xmid1, ymid1 = midpoint(x0, y0, x3, y3)

    thickness = int(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

    mask = np.zeros(img.shape[:2], dtype="uint8")
    cv2.line(mask, (xmid0, ymid0), (xmid1, ymid1), 255, thickness)

    masked = cv2.bitwise_and(img, img, mask=mask)

    return mask, masked

def inpaint_image(img: np.ndarray, swap_info: dict, text_str: str, flag:int=1):
    
    mask, img_masked = mask_from_info(img, swap_info)
    inpaint = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    fake_text_image = copy.deepcopy(inpaint)
    x0, y0, w, h = bbox_info(swap_info, flag=flag)

    color = (0,0,0)
    font = get_optimal_font_scale(text_str, w)

    img_pil = Image.fromarray(fake_text_image)
    draw = ImageDraw.Draw(img_pil)
    draw.text(((x0, y0)), text_str, font=font, fill=color)
    fake_text_image = np.array(img_pil)
    
    return fake_text_image


def read_img(path: str):
    
    img = np.array(imageio.imread(path))

    if img.shape[-1] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    else:
        return img



def read_json(path: str):
    with open(path) as f:
        return json.load(f)


def get_class(document_folder: str, gt: list):
    info = document_folder.split("_")[1:]

    for pc in info:
        if pc in gt:
            return pc
        else:
            continue
    return -1

def write_json(data:dict, path:str, name:str=None):
    if name is None:
        with open(path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
    else:
        path_to_save = os.path.join(path,name+".json")
        with open(path_to_save, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False, indent=4)

def store(img_loader: list,path_store:str=None):
    
    if path_store is None:
        path_store = os.path.join(os.getcwd() , "SIDTD_Generated")
        
    advisor = len(img_loader)//10
    for idx, image in enumerate(img_loader):
        class_name = image._name.split("_")[0]
        joined_path = os.path.join(path_store, class_name)

        if os.path.exists(joined_path):
            path_to_save = os.path.join(joined_path,image.fake_name+".jpg")
            imageio.imwrite(path_to_save,image.fake_img)
            write_json(image.fake_meta, joined_path,image.fake_name)
        
        else:
            os.makedirs(joined_path)
            path_to_save = os.path.join(joined_path,image.fake_name+".jpg")

            imageio.imwrite(path_to_save,image.fake_img)
            write_json(image.fake_meta, joined_path,image.fake_name)

        if (idx%advisor) == 0:
            print(f"{(idx//advisor)*10} % of the dataset stored")

        
        
    print("Data Successfuly stored")

def bbox_to_coord(x, y, w, h) ->Tuple[List[Int, Int],List[Int, Int],List[Int, Int],List[Int, Int]]:
    
    """This function convert the kin of the shape from bbox rectangle x0,y0 + heigh and weight to the polygon coordenades.

    Returns:
        _type_: _description_
    """
    x_f = x + w
    y_f = y + h

    c1, c2, c3, c4 = [x, y], [x_f, y], [x_f, y_f], [x, y_f]

    return [c1, c2, c3, c4]

def bbox_info(info, flag, shaped:bool=False, shaped_kin:str="rect") -> Tuple[Int, Int, Int, Int]:
    
    """This function return the rectangle of the template where are in located,
    
    Shaped: if shaped is True assume that the form of the info is an array that can represent the polygon or the rectangle
            
            If shapend kin is set to rect the info have this form [[x0,y0], [x1,y1]...] 
            If shaped kin is polygon the info have this form x = [x0,x1,x2,x3] and y = [y0,y1,y2,y3]

    Returns:
        Tuple[Int, Int, Int, Int]
    """
    
    if flag == 1:
        shape = info["shape_attributes"]
        x = shape["x"]
        y = shape["y"]
        w = shape["width"]
        h = shape["height"]
        
        return x,y,w,h
          
    shape = info["quad"] if not shaped else info   #here if the info is like [[x0,y0], [x1,y1]...] #Here if the info is x = [x0,x1,x2,x3] and y = [y0,y1,y2,y3]
    x0, x1, x2, x3 = shape[0][0], shape[1][0], shape[2][0], shape[3][0] if shaped_kin =="rect" else shape[0], shape[1], shape[2], shape[3]
    y0, y1, y2, y3 = shape[0][1], shape[1][1], shape[2][1], shape[3][1] if shaped_kin =="rect" else shape[0], shape[1], shape[2], shape[3]

    w = np.max([x0, x1, x2, x3]) - np.min([x0, x1, x2, x3])
    h = np.max([y0, y1, y2, y3]) - np.min([y0, y1, y2, y3])

    return x0, y0, w, h


def compute_homography(info1, info2, flag:int=1, mixed:bool=False):
    # take the coordinates of the region to replace
    if not mixed:   
        x, y, w, h = bbox_info(info1, flag=flag)
        x2, y2, w2, h2 = bbox_info(info2, flag=flag)
    else:
        try:
            x, y, w, h = bbox_info(info1, flag=flag)
            x2, y2, w2, h2 = bbox_info(info2, flag=abs(flag-1))
        except:
            x, y, w, h = bbox_info(info1, flag=abs(flag-1))
            x2, y2, w2, h2 = bbox_info(info2, flag=flag)
    
    
                       
    coord0 = np.array(bbox_to_coord(x, y, w, h), dtype=np.float32())
    coord1 = np.array(bbox_to_coord(x2, y2, w2, h2), dtype=np.float32())
    
    # Homography: p1 = H@p0 | p0 = inv(H)@p1
    # (where @ denotes matricial product, inv the inverse of the homography
    # and p0 and p1 are homogeneous coordinates for the pixel coordinates)

    H, mask = cv2.findHomography(coord0, coord1, cv2.RANSAC, 1.0)

    return H, coord0, coord1, mask


def replace_one_document(im_a, im_b, coord_a, H,dx1,dy1,dx2,dy2):

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
    im_rep[y_b +  dy1, x_b + dx1, ...] = im_a[y_a + dy2, x_a + dx2, ...]

    return im_rep

def replace_info_documents(im0, im1, data0, data1, delta1, delta2, flag:int=1, mixed:bool=False):
    H, coord0, coord1, _ = compute_homography(data0, data1, flag=flag, mixed=mixed)

    dx1,dy1 = delta1
    dx2,dy2 = delta2

    im1_rep = replace_one_document(im0, im1, coord0, H, dx1,dy1,dx2,dy2)

    im0_rep = replace_one_document(im1, im0, coord1, np.linalg.inv(H), dx1,dy1,dx2,dy2)

    return im0_rep, im1_rep
