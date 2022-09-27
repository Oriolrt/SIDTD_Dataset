import sys 
from Fake_Loader.utils import *
import random


def generate_crop_and_replace(path_img1:str, path_img2:str, annotations_path:str,shift_pixels:list=[5,5] ,additional_annotation_path:str=None):
    delta1 = delta2 = shift_pixels
    
    img1 = read_img(path_img1)
    img_id_1 = int(path_img1.split("/")[-1].split(".")[0])

    img2 = read_img(path_img2)
    img_id_2 = int(path_img2.split("/")[-1].split(".")[0])
    
    annotations = read_json(annotations_path)

    
    selected1 = list(annotations["_via_img_metadata"])[img_id_1]

    fields1 = annotations["_via_img_metadata"][selected1]["regions"]
    
    if additional_annotation_path is not None:
        img2_annotations = read_json(additional_annotation_path)
        selected2 = list(img2_annotations["_via_img_metadata"])[img_id_2]
        fields2 = img2_annotations["_via_img_metadata"][selected2]["regions"]
        
    else:
        selected2 = list(annotations["_via_img_metadata"])[img_id_2]
        fields2 = annotations["_via_img_metadata"][selected2]["regions"]

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





def generate_inpaint(path_img:str, annotations_path:str, mark=False):
    
    img = read_img(path_img)
    img_id = int(path_img.split("/")[-1].split(".")[0])


    
    annotations = read_json(annotations_path)

    selected = list(annotations["_via_img_metadata"])[img_id]
    print(selected)
    fields = annotations["_via_img_metadata"][selected]["regions"]

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



if __name__ == "__main__":

    img1, img2, field1, field2 = generate_crop_and_replace("Test_Example/Samples_Test/MIDV2020/alb_id/images/00.jpg",
                                                           "Test_Example/Samples_Test/MIDV2020/aze_passport/images/01.jpg","Test_Example/Samples_Test/MIDV2020/alb_id/annotations/alb_id.json",
                                                           additional_annotation_path="Test_Example/Samples_Test/MIDV2020/aze_passport/annotations/aze_passport.json")
    
    
    img, field = generate_inpaint("Test_Example/Samples_Test/MIDV2020/alb_id/images/00.jpg","Test_Example/Samples_Test/MIDV2020/alb_id/annotations/alb_id.json")
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    #plt.subplot(1, 2, 2)
    plt.imshow(img2)
    plt.show()

    
    print(field1, field2)