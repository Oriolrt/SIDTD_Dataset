
# Data Loader

Here you have the font code to download and work with different benchmarks and our own benchmark
for the real/fake binary classification approach.

There exist 5 different type of benchmarks whose behaviour has been changed in order to fit with our task.

## Documentation

And the structure is decripted as follows:
```
Loader
│   Datasets.py
│   Loader_Modules.py
|   __init__.py    
```


Mainly we have to .py scripts [__Loader_Modules.py__ & __Datasets.py__].


## __Datasets.py__

This file is the core of the differents datasets that we have work with

The different datasets tho chose are [ SIDTD (Default), Banknotes, Findit, Fungus, Dogs]

All this benchamarks have been changed in order to fit our task and they will be available to download with our changes. To get the original datasets all the classes have the flag *conditioned* which must be True to get the original dataset.

## Run Example 

**TODO change code example and list the different possibilities to download the dataset (partially or entirely)**
To download the datasets we have the following code examples:

```python
        #load the videos depending of the nationality and also replicate the structure of the original benchmark (about 50G)
        data = SIDTD(type_download="videos", conditioned=True, download_original=True)
```
```python
        #load the templates depending of the nationality and also replicate the structure of the original benchmark (about 1.2G)
        data = SIDTD(type_download="templates", conditioned=True, download_original=True)
```

```python
        #load the frames of the videos depending of the nationality and also replicate the structure of the original benchmark (about 12G)
        data = SIDTD(type_download="clips", conditioned=True, download_original=True)
```

```python
        #load all the information #not implemented condition and original download
        data = SIDTD(type_download="all")
```
Once you have installed the package as described in the main **Readme** you can call the different functionalities as follows:

If you just want to get the benchmark you only need to go to the datasets and choice among [SIDTD, Dogs, Fungus, Findit, Banknotes]:

```python
    import Loader.Datasets as ld

    ld.SIDTD(conditioned=True, download_original=True)
```

To download pretrained models on SIDTD and reproduce results you can download all the models at once (type_download_models="all_trained_models") or download only the pretrain model you need (type_download_models="effnet" or "resnet" or "vit" or "transfg" or "arc"). However, if you want to download all the trained models at once, keep in mind that you need enough space as the size of all models is 28,1GB. The pretrained models will be stored in code_examples, in the folder pretrained_models. By default, only the pretrained Trans FG on ImageNet is downloaded. If you do not choose "all_trained_models" or "transfg", the pretrained Trans FG on ImageNet will not downloaded.

```python
    import Loader.Datasets as ld
    
    # load all trained models including pretrained transfg on ImageNet
    ld.SIDTD(type_download_models="all_trained_models", conditioned=True, download_original=True)

    # load trained EfficientNet models
    ld.SIDTD(type_download_models="effnet", conditioned=True, download_original=True)

    # load trained TransFG models including pretrained transfg on ImageNet
    ld.SIDTD(type_download_models="transfg", conditioned=True, download_original=True)
```


## __Loader_Modules.py__

Inside this file you will see the DataLoader class who takes 7 different inputs (for now)

    dataset -->    Define what kind of the different datasets do you want to download [Midv, Dogs, Fungus, Findit, Banknotes].
    
                    The datasets have been changed in order to the different approach we are working on
    
    kind --> in case of benchmark SIDTD define ("templates", "clips", "videos), else set to None
    
    Type_split --> Diferent kind of split for train the models. The diferents splits are [kfold, normal or few_shot]

    batch_size --> define the batch of the training set

    kfold_, normal, few_shot_ (split) --> define the behaviour of the split based on what kind of split did you put

    metaclasses --> in case the type of the split is few shot learning is the way to define the groups to generate the metatraing and the metatest

    conditioned --> flag to define if you want to train with the metaclasses inside the dataset thath downloaded 

    balanced --> boolean tho define the kin of partition to generate. if set to true the csv to generate is imbalanced by fakes [80% reals, 20% fakes]


The class will search if the dataset that you want to work with is downloaded in your computer, if not it will create the folder dataset with it inside


Depend of the partition you define, besides getting the train, val, test arrays with the data atacched in memory, you will get the CSV of the partition for more flexible train.

The code will also provide the batch based on the amount of batches did you define (default=1).

Below you have some examples for the different kin of partitions:

## Run Example

To load dataset and make the few shot partition
```python   
    #templates
    python3 Loader/Loader_Modules.py -ts few_shot --few_shot_split random 0.6 0.4 --kind templates

    #clips
    python3 Loader/Loader_Modules.py -ts few_shot --few_shot_split random 0.6 0.4 --kind clips

    #cropped clips
    python3 Loader/Loader_Modules.py -ts few_shot --few_shot_split random 0.6 0.4 --kind clips_cropped
```

To load dataset and make the kfold partition
```python   
    #templates
    python3 Loader/Loader_Modules.py -ts kfold --kfold_split 10 --kind templates

    #clips
    python3 Loader/Loader_Modules.py -ts kfold --kfold_split 10 --kind clips

    #cropped clips
    python3 Loader/Loader_Modules.py -ts kfold --kfold_split 10 --kind clips_cropped
```

To load dataset and make the cross val partition
```python   
    #templates
    python3 Loader/Loader_Modules.py -ts cross --cross_split 0.8 0.1 0.1--kind templates
    
    #clips
    python3 Loader/Loader_Modules.py -ts cross --cross_split 0.8 0.1 0.1 --kind clips
    
    #cropped clips
    python3 Loader/Loader_Modules.py -ts cross --cross_split 0.8 0.1 0.1 --kind clips_cropped
```

To load all pretrained models on SIDTD at once without downloading datasets
```python   
    #templates
    python3 Loader/Loader_Modules.py --kind_models all_trained_models --kind no

    #clips with background
    python3 Loader/Loader_Modules.py --kind_models all_trained_models --kind no --unbalanced

    #cropped clips
    python3 Loader/Loader_Modules.py --kind_models all_trained_models --kind no --unbalanced --cropped
```

To load dataset and static csv on kfold partition to reproduce results
```python   
    #templates
    python3 Loader/Loader_Modules.py -ts kfold --kind templates --download_static

    #clips with background
    python3 Loader/Loader_Modules.py -ts kfold --kind clips --download_static

    #cropped clips
    python3 Loader/Loader_Modules.py -ts kfold --kind clips_cropped --download_static
```


