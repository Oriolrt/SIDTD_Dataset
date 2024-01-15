# This file creates csv with image path, label number and label name for each k-fold partition of train, validation and test set.
# The train set represent 80% of the dataset. The validation and test set represent each 10% of the dataset
# The images must be classified in 2 directories with the following name: 'reals' for genuine images and 'fakes' for forged images.
# The directories structure must be the following: 
# dataset_name
#   |── reals
#   |── fakes
#the directory root is the name of the dataset that contains two directory, 'reals' and 'fakes'.

import os
import argparse
import numpy as np
import pandas as pd
import glob
from sklearn.utils import shuffle
from distutils.dir_util import copy_tree
import shutil

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", default='dataset_raw', type=str, help='Name of the dataset to perform the kfold partition')
parser.add_argument("--kfold", default=10, type=int, help='Number of k-fold partition')
args = parser.parse_args()

l_label = []
l_img = []
for file in glob.glob('{}/*/*'.format(args.dataset_name)):
    path = file.replace('\\','/')
    path_decompose = path.split('/',2)
    path_img = os.getcwd() + '/' + path
    label = path_decompose[1]
    l_label.append(label)
    l_img.append(path_img)

columns = ["label_name", "label", "image_path"]
data = np.array([l_label, l_label, l_img]).T
new_df = pd.DataFrame(data=data, columns=columns)

new_df['label'] = new_df['label'].map({'reals':0,
                                      'fakes':1},
                                      na_action=None)
                                      
# Window to split the dataset in training/validation/testing set for the 10-fold
print('Splitting dataset into {}-folds partition with train/validation/test sets...'.format(args.kfold))    
dataset = args.dataset_name
k = len(new_df)/args.kfold
shuffled_df = shuffle(new_df)
if not os.path.exists('split_kfold/'+ dataset):
        os.makedirs('split_kfold/'+ dataset)
for iteration in range(args.kfold):
    b_low = int(iteration*k)
    b_high = int((1 + iteration)*k)
    df_test = shuffled_df[b_low:b_high]
    df_test.to_csv('split_kfold/'+ dataset +'/test_split_'+ dataset + '_it_' + str(iteration) + '.csv',index=False)
    df_val_train = shuffled_df.drop(shuffled_df.index[b_low:b_high])
    if iteration == args.kfold -1 :
        b_high = int(k)
        df_val = df_val_train[:b_high]
        df_val.to_csv('split_kfold/'+ dataset +'/val_split_'+ dataset + '_it_' + str(iteration) + '.csv', index=False)
        df_train = df_val_train.drop(df_val_train.index[:b_high])
        df_train.to_csv('split_kfold/'+ dataset +'/train_split_'+ dataset + '_it_' + str(iteration) + '.csv', index=False)
    else:
        df_val = df_val_train[b_low:b_high]
        df_val.to_csv('split_kfold/'+ dataset +'/val_split_'+ dataset + '_it_' + str(iteration) + '.csv', index=False)
        df_train = df_val_train.drop(df_val_train.index[b_low:b_high])
        df_train.to_csv('split_kfold/'+ dataset +'/train_split_'+ dataset + '_it_' + str(iteration) + '.csv', index=False)

print('Directory split_kfold created.')