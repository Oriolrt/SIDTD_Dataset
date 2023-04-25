# About the project

The purpose of this project is to generate a new data partition to recognize DNI without the need to use completely real data. To achieve this, different databases have been modified. In addition to these modifications, a database of its own has been created as a result of varying and changing the structure of the images that are in it.

## Structure

The structure of the project is decripted as follows:
```
SIDTD
│   LICENSE
│   setup.py
|   requirements.txt   
│
└───Fake_Generator
|   |
│   └───Fake_Loader 
│   |
|   |
|   |___Test_Examples
|       
└───Loader
|
|___code_examples
|
|___models
```


All the names in this structure are folders. Every folder have his own Readme to make clearer all the structure.

In the *Fake_Generator* folder are the scripts to create your own dataset with different variability

In the *Loader* folder are the main functionalities to download the benchmark  that we used to train the models. Furthermore you can download the different datasets that we used to train and pretrain the models.

In the *code examples* folder are 4 different scripts that reproduce some functionalities that we have done in order to create the dataset and, train and/or test with different models. You should run the scripts locally.

Finally in the *models*' folder store the models' code. This is the folder to go if you want more information about the models' implementation. The models are seperate in 3 different folders *arc_pytorch* for Co-Attention ARC, *transfg* for Trans FG model and, *Baseline* for EfficientNet, ResNet and Visition Transformer models.


## Installation

To get the Dataloader package you will need to:

```bash
    pip install git+https://github.com/Oriolrt/SIDTD_Dataset.git
```

Enter to any python terminal and now you can
```bash
  import DataLoader
```
It shouldnt exist any type of error


#### To resolve any doubt  

+ cboned@cvc.uab.cat
+ oriolrt@cvc.uab.cat
+ mtalarmain@cvc.uab.cat
## Acknowledgments
SOTERIA has received funding from the European Union’s Horizon 2020 	research and innovation programme under grant agreement No 101018342 

Disclaimer: This content reflects only the author's view. The European Agency is not responsible for any use that may be made of the information it contains. 