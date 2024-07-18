# Introduction to each branch of the package

All the names in this structure are folders. Every folder have his own Readme to make clearer all the structure, except folders models/arc_pytorch, models/transfg, models/Baseline and utils, as it is only piece of code used in other folders.

## data
Inside the data folder, you will find all the directories and scripts containing information related to the benchmark data. This information is divided into generating fake documents and downloading our different partitions of the benchmark itself.

In the *data/DataGenerator* folder are the scripts to create your own dataset with different variability.
In the *data/DataLoader* you can download the whole SIDTD or only one type of SIDTD data: templates, clips, cropped clips.
In the *data/explore* folder are the scripts to generate new forged identity document and then explore the different forgeries our script can do.

## models
In the *models* folder are the main functionalities to download the benchmark models (EfficientNet, ResNet, ViT, TransFG and Co-Attention ARC) that we used to train on SIDTD. 
In the *models/explore* folder are the main functionalities to train or test the benchmark models on SIDTD or a custom dataset. Hence, it can be used to reproduce the results. Depending on your needs, you should first download data in *data/DataLoader* and/or download models in *models*.
The folder *models/arc_pytorch* give more information about the Co-Attention ARC model's implementation.
The folder *models/transfg* give more information about the TransFG model's implementation.
The folder *models/Baseline* give more information about the EfficientNet, ResNet and ViT models' implementation.

## utils
In utils folder you will find the code that is used to perform document forgeries and batch generator function for model training and test.