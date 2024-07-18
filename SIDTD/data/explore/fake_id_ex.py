from SIDTD.utils.transforms import *

from PIL import Image

import matplotlib.pyplot as plt
import argparse
import random 

# TODO Mirar si aquest fitxer és necessari ( Nota no ho es )
def custom_crop_and_replace(img:np.ndarray, img2:np.ndarray, info:dict, info2:dict, delta1:list, delta2:list):
    """In case you want to create your own crop and replace with your own images you can do it, just make sure that the 
    info with the coordenades must be as follows:
         
        field [name, surname....] : {
                
            value : "Carlos",
            
            ----------- here is the important thing, make sure that this part is as follows.
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

    img_generated1, img_generated2 = replace_info_documents(img, img2, info, info2, delta1=delta1, delta2=delta2, flag=0, mixed=False)
    
    return img_generated1, img_generated2

def custom_inpaint_and_rewrite(img: np.ndarray, info: dict, text_str:str=None):
    
    """In case you want to create your own inpaint with your own images you can do it, just make sure that the 
    info with the coordenades must be as follows:

            field [name, surname....] : {
                
            value : "Carlos",
            "quad": [ [0, 0], [111, 0],
                    [111, 222], [0, 222] ]
        }
    
    Args:
        img (np.ndarray): The img which will be inpainted
        info (dict): the dictionary of the image with the metadata of the Bbox described
        text_str: The original text that you want to inpaint
        
    Returns:
        the image inpainted
    """
    if text_str is None:
        text_str = info["value"]
        
    fake_text_image = inpaint_image(img=img,swap_info=info, text_str=text_str, flag=0)

    return fake_text_image


def plt_inpaint(img:np.ndarray, fake_img:np.ndarray):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img)
    plt.legend("Original Image")
    plt.subplot(1,2,2)
    plt.imshow(fake_img)
    plt.legend("img Inpainted")
    
    plt.show()

def plt_crop_and_replace(img1:np.ndarray, img2:np.ndarray, fake_document1:np.ndarray, fake_document2:np.ndarray):
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img1)
    plt.subplot(1,2,2)
    plt.imshow(fake_document1)

    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(img2)
    plt.subplot(1,2,2)
    plt.imshow(fake_document2)

    plt.show()
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Main for the execution of the examples")
    parser.add_argument("--transformation", "-t",choices=["cr", "ir"], required=True, type=str, help="Mode")
    
    
    parser.add_argument("--src_path", "-srcpath", required=True, type=str, help="path of the src image")
    parser.add_argument("--src_annotations","-srcan", required=True, type=str, help="annotations of the src img ")
    parser.add_argument("--save_path1", required=True, type=str, help="path of the saved image 1")
    
    #crop and replace
    parser.add_argument("--trg_path", "-trgpath", type=str, help="path of the trg image")
    parser.add_argument("--trg_annotations", "-trgan", default=None,type=str, help="annotations of the trg img ")
    parser.add_argument("--shift_boundary", "-sb", type=int, default=10, help= "shifting constant for crop and replace")
    parser.add_argument("--save_path2", type=str, help="path of the saved image 2")

    #inpainting 
    parser.add_argument("--field_to_change", "-f",  default=None,type=str, help="The field that you want to inpaint")
    parser.add_argument("--custom_text", "-ct",  default=None, type=str, help="The field that you want to inpaint")
    
    # Get help
    parser.add_argument("--help_func", action='store_true', help="If set up to True the description of the function will be displayed")
    
    

    args = parser.parse_args()

    img = utils.read_img(args.src_path)
    annotations = utils.read_json(args.src_annotations)
    
    if args.transformation == "ir":

        if args.help_func:
            help(custom_inpaint_and_rewrite)

        if args.field_to_change is None:
            field_to_change = random.choice(list(annotations.keys()))
            info = annotations[field_to_change]
            print("The info that will be inpainted is ", field_to_change)
        else:
            info = annotations[args.field_to_change]
            
        fake_img = custom_inpaint_and_rewrite(img, info, text_str=args.custom_text)
        
        fake_data1 = Image.fromarray(fake_img)  
        fake_data1.save(args.save_path1)

        
    elif args.transformation == "cr":
        if args.help_func:
            help(custom_crop_and_replace)
            
        assert args.trg_path != None
        delta1 = random.sample(range(args.shift_boundary),2)
        delta2 = random.sample(range(args.shift_boundary),2)
        trg_image = utils.read_img(args.trg_path)
        
        if args.trg_annotations is not None:
            annotations2 = utils.read_json(args.trg_annotations)
            fake_img1, fake_imgs2 = custom_crop_and_replace(img, trg_image, annotations,annotations2,delta1, delta2)
        else:
            fake_img1, fake_imgs2 = custom_crop_and_replace(img, trg_image, annotations,delta1=delta1, delta2=delta2)

        fake_data1 = Image.fromarray(fake_img1)  
        fake_data2 = Image.fromarray(fake_imgs2)           
        fake_data1.save(args.save_path1)
        fake_data2.save(args.save_path2)


        

        
        

            