
# Data Loader

Here you have the source code to download different datasets and work with our own benchmark models in a real/fake binary classification approach.

## Documentation

The structure is organized as follows:
```
DataLoader
│   Datasets.py
│   Loader_Modules.py
|   __init__.py    
```

## __Datasets.py__

This file is the core of the differents datasets that we have worked with.

The different datasets availables are [ SIDTD (Default), Banknotes, Findit, Fungus, Dogs]

All this benchamarks have been changed in order to fit our task and they will be available to download with our changes. To get the original datasets all the classes have the flag *download_original* which must be True to get the original dataset.

## Run Example 

To download the datasets we have the following code examples:

```python
        data = SIDTD(download_original=False).download_dataset("templates")  #["you  can choose among templates, clips, clips_cropped and "all" of them]
```

Once you have installed the package as described in the main **Readme** you can call the different functionalities as follows:

If you just want to get the benchmark you only need to go to the datasets and choice among [SIDTD, Dogs, Fungus, Findit, Banknotes]:

```python
    from SIDTD.data.DataLoader.Datasets import *
    
    SIDTD(download_original=...)
```


## __Loader_Modules.py__


Our Data Loader allows you to download and prepare datasets for training models. It provides options for different types of data splits and preprocessing.

## Usage

```bash
python data_loader.py [--dataset DATASET] [--kind KIND] [--download_static] [--type_split TYPE_SPLIT] [--unbalanced] [-c|--cropped]
```

## Arguments
        --dataset: Specify the dataset to download. Available options: "SIDTD", "Dogs", "Fungus", "Findit", "Banknotes". (default: "SIDTD")

        --kind: Specify the type of benchmark data to download. Available options: "templates", "clips", "clips_cropped", "no". If "no" is selected, the dataset will not be downloaded. (default: "templates")

        --download_static: Set this flag if you want to download the static CSV for reproducibility of the models.

        --type_split: Specify the type of data split to train the models. Available options: "hold_out", "kfold", "few_shot". (default: "hold_out")

        --unbalanced: Flag to prepare the unbalanced partition.

        -c|--cropped: Flag to use the cropped version of clips.

## Split Types
### Hold-out Split

*   If --type_split hold_out is selected, the following argument is required:

        --hold_out_split: Define the split ratios for hold-out validation.
                            This argument should be a list containing three elements that sum up to 1.  
                            For example, --hold_out_split 0.8 0.1 0.1 will result in a split of 80% training, 10% validation, and 10% testing.Few-shot Split


*    If --type_split few_shot is selected, the following arguments are required:

    --few_shot_split: Define the split behavior for few-shot learning. 
                      The first value must be either "random" or "ranked", and the following values specify the proportions.
                      For example, --few_shot_split random 0.75 0.25 will perform a random few-shot split with 75% metatrain and 25% metatest.
    
    --metaclasses: Define the second level to group the metatrain and the metatest. This argument should be a list of metacategories. 
                   For example, --metaclasses id passport will group the metatrain and metatest based on the "id" and "passport" metacategories.

### K-fold Split

* If --type_split kfold is selected, the following argument is required:

        --kfold_split: Define the number of folds for k-fold validation.
                       For example, --kfold_split 10 will perform 10-fold validation.

## Run Example

Templates
```python   
    #templates
    python data/DataLoader/Loader_Modules.py --dataset SIDTD --kind templates --type_split hold_out --hold_out_split 0.8 0.1 0.1
    
    #clips
    python data/DataLoader/Loader_Modules.py --dataset SIDTD --kind templates --type_split kfold --kfold_split 10
    
    #cropped clips
    python data/DataLoader/Loader_Modules.py --dataset SIDTD --kind templates --type_split few_shot --few_shot_split random 0.75 0.25 --metaclasses id passport
```

Clips
```python   
    #templates
    python data/DataLoader/Loader_Modules.py --dataset SIDTD --kind clips --type_split hold_out --hold_out_split 0.8 0.1 0.1 --unbalanced

    #clips
    python data/DataLoader/Loader_Modules.py --dataset SIDTD --kind clips --type_split kfold --kfold_split 10 --unbalanced

    #cropped clips
    python data/DataLoader/Loader_Modules.py --dataset SIDTD --kind clips --type_split few_shot --few_shot_split random 0.75 0.25 --metaclasses id passport -c
```

Clips Cropped
```python   
    #templates
    python data/DataLoader/Loader_Modules.py --dataset SIDTD --kind clips_cropped --type_split hold_out --hold_out_split 0.8 0.1 0.1

    #clips
    python data/DataLoader/Loader_Modules.py --dataset SIDTD --kind clips_cropped --type_split kfold --kfold_split 10

    #cropped clips
    python data/DataLoader/Loader_Modules.py --dataset SIDTD --kind clips_cropped --type_split few_shot --few_shot_split random 0.75 0.25 --metaclasses id passport -c
```


Statics


```python   
    #templates
    python data/DataLoader/Loader_Modules.py -ts kfold --kind templates --download_static

    #clips with background
    python data/DataLoader/Loader_Modules.py -ts kfold --kind clips --download_static

    #cropped clips
    python data/DataLoader/Loader_Modules.py -ts kfold --kind clips_cropped --download_static
```

to download the other statics, follow the same structure as depicted above


## Further explanation

Here's an explanation of the functionalities provided by the `DataLoader` class:

1. The `_Data` class:
   - This is a nested class within `DataLoader` that represents a single data instance. It has four attributes: `_sample`, `_gt`, `_class`, and `_metaclass`, which correspond to the sample data, ground truth data, class label, and metaclass label, respectively.
   - It provides properties (`sample`, `gt`, `clas`, `metaclass`) to access the attributes.
   - It also implements the `__getitem__` method, but its implementation is not provided in the code.

3. `change_path` method:
   - It takes a `path` parameter and reads the CSV file at the specified path using pandas.
   - It updates the "image_path" column in the DataFrame by appending the current working directory to the image paths.
   - It saves the updated DataFrame back to the CSV file.

4. `set_static_path` method:
   - It sets the paths of static CSV files by calling the `change_path` method for each CSV file in the "split_normal" and "split_kfold" directories.

5. `_kfold_partition` method:
   - It performs k-fold partitioning of the dataset.
   - It creates a directory for the k-fold split.
   - It splits the dataset into training, validation, and testing sets for each fold.
   - It saves the split CSV files in the corresponding directories.

6. `_shot_partition` method:
   - It performs few-shot partitioning of the dataset.
   - It creates a directory for the few-shot split.
   - It divides the dataset into meta_train and meta_test sets based on the specified proportion and metaclasses.
   - It saves the split CSV files in the corresponding directory.

7. `_ranking_shot_partition` method:
   - This method is not implemented and raises a NotImplementedError.

8. `_hold_out_partition` method:
   - It performs hold-out partitioning of the dataset.
   - It creates a directory for the hold-out split.
   - It splits the dataset into training, validation, and testing sets based on

9. `load_dataset` method:
    - This method is used to load the dataset from a CSV file into a list of `DataLoader._Data` instances.
    - It takes a `path` parameter that specifies the path to the CSV file.
    - It reads the CSV file using pandas and retrieves the label, image path, and class information from the DataFrame.
    - It creates a list of `DataLoader._Data` instances, populating each instance with the retrieved information.
    - It returns the list of `DataLoader._Data` instances.

10. `make_batches` method:
    - This static method is used to create batches from a list of `DataLoader._Data` instances.
    - It takes two parameters: `structure` (a list of `DataLoader._Data` instances) and `batch_size` (the desired size of each batch).
    - If the length of `structure` is 1, it calculates the number of batches required and creates a list of tuples, where each tuple represents the start and end indices of a batch.
    - If the length of `structure` is greater than 1 (indicating multiple folds), it creates a nested list of batches for each fold.
    - Each batch is represented by a tuple of start and end indices.
    - It returns the list of batches.
