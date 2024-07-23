## Usage examples
This folder contains scripts used to train and test the benchmark models.

# Train Models

The script __train.py__ train one of the five models implemented in the Benchmark study: EfficientNet, ResNet, Vision Transformer (ViT), Trans FG and Co-Attention Attentive Recurrent Comparator (Co-Attention ARC).

We describe below the process, step by step, to train a custom dataset with our benchmark models.

### 1. Dataset Structure

The images must be classified in 2 directories with the following name: 'reals' for genuine images and 'fakes' for forged images. In a nutshell, the directories must respect the following structure:

```
{DATASET_NAME}
    |-- reals
    |-- fakes
```
Here, instead of {DATASET_NAME} you can choose the dataset name you want. The directory contains two other directories, 'reals' and 'fakes'. Remember to always refer to it as it is written when you call the dataset in training phase.

### 2. Generate CSV to store image paths

Depending on the type of training you wish to perform, two solutions are available to generate the CSV that will store the image paths: k-fold cross-validation and train, validation and test split. Choose the solution that fits better to your needs.

#### a. k-fold cross-validation

Run generate_kfold_csv file to generate 10-fold cross validation partition of your custom dataset:

```
python generate_kfold_csv.py --dataset_name {DATASET_NAME} --kfold 10
```
Specify the dataset name set in the previous step. Change kfold number to change the number of k-fold cross validation partition.

This operation should take only a few seconds. The 'split_kfold' directory is created.

#### b. Train, validation and test split

You must create three different files splitted as you wish with the following names: 'train_split_{DATASET_NAME}.csv', 'val_split_{DATASET_NAME}.csv' and 'test_split_{DATASET_NAME}.csv'. For each three CSVs, the image paths are stored in a column named 'image_path'. The labels are stored in the column 'label_name', 'reals' stands for the real documents and 'fakes' stands for the falsified documents. Another column must be created called 'label', where 0 corresponds to 'reals' and 1 corresponds to 'fakes'.
The files should be saved in the following path: 'SIDTD/models/explore/split_normal/{DATASET_NAME}/'.

### 3. Model training

During training, the model is saved in the *trained_models* folder. The model with the highest accuracy that is saved. You can change the location of the folder to another path with the flag --save_model_path.

To train one of the five models with CUDA on your custom dataset, you should run the line below that correspond to the model you want to train with the corresponding name of the dataset used:
```
# Train EfficientNet model
CUDA_VISIBLE_DEVICES=0 python train.py -- name --dataset {DATASET_NAME} --model='efficientnet-b3'
```
train.py --name='EfficientNet_faker_data_augmentation' --dataset='clip_cropped_MIDV2020' --model='efficientnet-b3' --type_split kfold --epochs 2 --nsplits 1 --faker_data_augmentation

### Basic flag options

You can choose the model with the flag --model, but be careful to write without typo the model name that correspond to your chosen model:  
+ EfficientNet -> 'efficientnet-b3'
+ ResNet -> 'resnet50'
+ ViT -> 'vit_large_patch16_224'
+ Trans FG -> 'trans_fg'
+ CoAARC 'coatten_fcn_model' 

-- name : important to choose a different name to each experiment your are working on. It will allow to save weight models and results in different files.
-- dataset : You can choose to train on the dataset of your choice with this flag. The name of the dataset must be the exact same name as the one of the folder that store the dataset you want to use. 
--faker_data_augmentation : write this flag to perform forgery augmentation during training.
--nsplits : You can choose the number of partition you want to train. 10 is set as default.
--device cpu : run training with CPU. Default option is --device cuda to use a GPU.
--static : If yes, use static CSV file to reproduce training. Else,  if no, set as default, use *your own* CSV file with image paths.
-td or --type_data : If static flag is set to 'yes', choose type of SIDTD data to train on. Choose among: "templates", "clips", "clips_cropped". "templates" is set as default.
-ts, --type_split : Choose the type of partitionning to perform training: "cross" and "kfold". "kfold", stands for k-fold cross validation. "cross" stands for train, validation and test split. "kfold" is set as default.

Please, remember to train according to the static CSV you have downloaded. For instance, if you have downloaded the k-fold cross validation version of cropped clips you should mentionned the flags such that: --type_data='clips_cropped' --type_split='kfold'.

### Specific training flag options

More flags can modify the default parameters. If you need to fine-tune the parameters you can modify the models' parameter thanks to the flags that corresponds to each model. 

See specific flag options for EfficientNet, ResNet and ViT in options/baseline_options.py
See specific flag options for Trans FG in options/trans_fg_options.py
See specific flag options for CoAARC in options/coaarn_options.py

### Results

Results by fold are generated in csv file in the 'results_files' directory. The training history (loss and accuracy) can be found in the 'plots' directory. Trained models will be located in the trained_models directory.

### Reproduce results

If you want to reproduce results or train models on SIDTD, read info in SIDTD/data/DataLoader/README.md about how to load data. The CSV file will be automatically downloaded.

# Test Models

The models can be tested on a custom dataset. The procedure is as follows:


### 1. Download models

git clone le repo https://github.com/Oriolrt/SIDTD_Dataset
cd SIDTD_Dataset/SIDTD/models
python load_models.py --kind_models='all_trained_models'

### 2. Create image paths CSV file

Create 10 identical csv files with the paths for each image in the dataset. The paths are stored in a column named 'image_path'. The labels are stored in the column 'label_name', 'reals' stands for the real documents and 'fakes' stands for the falsified documents. Another column must be created called 'label', where 0 corresponds to 'reals' and 1 corresponds to 'fakes'.

### 3. Folder structure

Create a 'split_kfold' folder and then in 'split_kfold' create another folder with the name of the dataset used, for example 'custom_dataset'. Store the CSVs created in step 2 in this folder. The csv names must be in the following format: test_split_{DATASET_NAME}_it_0.csv, test_split_{DATASET_NAME}_it_1.csv, ..., test_split_{DATASET_NAME}_it_9.csv. To illustrate my point, let's see below the folder's structure:

```
SIDTD_Dataset
    |-- SIDTD
          |-- models
                |-- explore
                      |-- split_kfold
                            |-- {DATASET_NAME}
                                      |-- test_split_{DATASET_NAME}_it_0.csv
                                      |-- test_split_{DATASET_NAME}_it_1.csv
                                      |-- test_split_{DATASET_NAME}_it_2.csv
                                      |-- test_split_{DATASET_NAME}_it_3.csv
                                      |-- test_split_{DATASET_NAME}_it_4.csv
                                      |-- test_split_{DATASET_NAME}_it_5.csv
                                      |-- test_split_{DATASET_NAME}_it_6.csv
                                      |-- test_split_{DATASET_NAME}_it_7.csv
                                      |-- test_split_{DATASET_NAME}_it_8.csv
                                      |-- test_split_{DATASET_NAME}_it_9.csv
```

### 4. Run inference

To test our models on a custom dataset with CUDA. Run in code_examples: 

```
python test.py --dataset='dataset_raw' --pretrained='yes' --type_split kfold --nsplits 10 --static no --pretrained  yes  --dataset='clip_cropped_MIDV2020' --model='efficientnet-b3'
```

### 5. Run different model

To test different model, repeat step 4, changing the --model flag to: resnet, vit_large_patch16_224, trans_fg and coatten_fcn_model.

You can choose the model with the flag --model, but be careful to write without typo the model name that correspond to your chosen model:  
+ EfficientNet -> 'efficientnet-b3'
+ ResNet -> 'resnet50'
+ ViT -> 'vit_large_patch16_224'
+ Trans FG -> 'trans_fg'
+ CoAARC 'coatten_fcn_model' 

### 6. Inference options: flags

-- name : important to choose a different name to each experiment your are working on. It will allow to save weight models and results in different files. Also, between train and test, it should be the same name for the test than the training you performed earlier in order to use the same trained models.
-- dataset : You can choose to test on the dataset of your choice with this flag. The name of the dataset must be the exact same name as the one of the folder that store the dataset you want to use. 
--nsplits : You can choose the number of partition you want to perform inference on.
--device cpu : run inference with CPU. Default option is the use of GPU.
--pretrained : If no, use *your* trained models. Else, if yes, use *our* pretrained models.
--static : If yes, use static CSV file to reproduce results. Else, if no, use *your own* CSV file with image paths.
-td or --type_data : If static flag is set to 'yes', choose type of SIDTD data to perform inference on. If pretrained flag is set to 'yes', choose the type of SIDTD the model has been trained on. Choose among: "templates", "clips", "clips_cropped".
-ts, --type_split : Choose the type of partition chosen to perform inference among: "cross" and "kfold". "kfold", stands for k-fold cross validation. "cross" stands for train, validation and test split.

## Other type of inference

Other inference options are also possible: inference with trained model and reproduce results. For that type of inference, no needs to do the first three steps, just adapt the inference flags according to your needs. Hence, for inference with trained models, set --pretrained flag as no and for reproducing results, set --static flag to yes. 


# Few shot setting

It is also possible to train and test with few-shot setting. 
