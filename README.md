# About the project

The purpose of this project is to generate a new synthetic identity document dataset with forged ID based on MIDV2020 dataset. This is constructed in order to build models that will classify genuine and forged identity document without the need to use real-world identity document.

## Structure

The structure of the project is organized as follows:
```
SIDTD
│
└───data
|   |
│   └───DataGenerator 
│   |
|   └───DataLoader 
|   |
|   └───explore
|       
└───models
|   |
│   └───arc_pytorch 
│   |
|   └───Baseline 
|   |
│   └───explore 
│   |
|   └───transfg
|
|___utils
```


All the names in this structure are folders. Every folder have his own Readme to make clearer all the structure, except folders *models/arc_pytorch*, *models/transfg*, *models/Baseline* and *utils*, as it is only piece of code used in other folders.

In the *data/DataGenerator* folder are the scripts to create your own dataset with different variability.
In the *data/DataLoader* you can download the whole SIDTD or only one type of SIDTD data: templates, clips, cropped clips.
In the *data/explore* folder are the scripts to generate new forged identity document and then explore the different forgeries our script can do.

In the *models* folder are the main functionalities to download the benchmark models (EfficientNet, ResNet, ViT, TransFG and Co-Attention ARC) that we used to train on SIDTD. 
In the *models/explore* folder are the main functionalities to train or test the benchmark models on SIDTD or a custom dataset. Hence, it can be used to reproduce the results. Depending on your needs, you should first download data in *data/DataLoader* and/or download models in *models*.
The folder *models/arc_pytorch* give more information about the Co-Attention ARC model's implementation.
The folder *models/transfg* give more information about the TransFG model's implementation.
The folder *models/Baseline* give more information about the EfficientNet, ResNet and ViT models' implementation.

In *utils* folder you will find the code that is used to perform document forgeries and batch generator function for model training and test.


## Installation

To get the Dataloader package you will need to:

```
python setup.py install --user
```

Enter to any python terminal and now you can
```bash
  import SIDTD.data.DataLoader
```
It shouldnt exist any type of error


#### To resolve any doubt  

+ cboned@cvc.uab.cat
+ oriolrt@cvc.uab.cat
+ mtalarmain@cvc.uab.cat
## Acknowledgments
SOTERIA has received funding from the European Union’s Horizon 2020 	research and innovation programme under grant agreement No 101018342 

Disclaimer: This content reflects only the author's view. The European Agency is not responsible for any use that may be made of the information it contains. 