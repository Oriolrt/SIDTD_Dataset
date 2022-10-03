# Load the data into array to make the Batch loader faster during the training

import os
import argparse
import pandas as pd
import imageio.v2 as imageio
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to ARC')
parser.add_argument("--csv_dataset_path", default = os.getcwd() + '/split_kfold/', type=str)
parser.add_argument("--dataset", default = 'dataset_raw', type=str, help='dataset to use')
parser.add_argument("--array_path", default = os.getcwd() + '/omniglot/', type=str)
opt = parser.parse_args()

def read_image(image_path, image_size):
    image = imageio.imread(image_path)
    if image.shape[-1]>=4:
        image = image[...,:-1]
    image = cv2.resize(image, (image_size,image_size))
    
    return np.moveaxis(image, -1, 0) 

def load_data():
    # Load all the images path for the chosen dataset for each fold train/validation/test set
    print('+++++++++++++++++++++++++++++++++++++++++++++++')
    print('+++++++++++     Loading Dataset     +++++++++++')
    
    # Create omniglot directory and a subdirectory with dataset name to store the data array
    np_array_path = opt.array_path +  opt.dataset + '/'
    if not os.path.exists(np_array_path):
        os.makedirs(np_array_path)
    
    for  iteration in range(10):
        print('+++++++++++       Partition {}       +++++++++++'.format(iteration))
        
        classes = ["reals","fakes"]
        
        for d_set in ['train','val','test']:
            
            path_set = opt.csv_dataset_path +  opt.dataset + '/' + d_set + '_split_' +  opt.dataset + '_it_' + str(iteration) + '.csv'
            df = pd.read_csv(path_set)
            
            for lbl in classes:
                
                print('****** Loading {} set for {} images ******'.format(d_set,lbl))
                array_data = []
                imgs_path = df[df['label_name']==lbl].image_path.values
                
                for path in imgs_path:
                    img = read_image(path, image_size=opt.imageSize)  # read and resize image
                    array_data.append(img)
                np_array_save = np_array_path + d_set + '_split_' + lbl + '_it_' + str(iteration) + '.npy'
                np.save(np_array_save, np.array(array_data))  # save all images to one single array of the label (reals or fakes)

def main() -> None:
    load_data()

if __name__ == "__main__":
    main()