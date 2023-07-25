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