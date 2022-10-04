# Data Loader

Here you have the font code to download and work with different benchmarks and our own benchmark
for the real/fake binary classification approach.

There exist 5 different type of benchmarks whose behaviour has been changed in order to fit with our task.




## Documentation

Mainly we have to .py scripts [Loader_Modules & Datasets]. The main file is the first One

Inside this file you will see the DataLoader class who takes 7 different inputs (for now)

    dataset -->    Define what kind of the different datasets do you want to download [Midv, Dogs, Fungus, Findit, Banknotes].
    
                    The datasets have been changed in order to the different approach we are working on
    
    Type_split --> Diferent kind of split for train the models. The diferents splits are [kfold, normal or few_shot]

    batch_size --> define the batch of the training set

    kfold_, normal, few_shot_ (split) --> define the behaviour of the split based on what kind of split did you put

    conditioned --> flag to define if you want to train with the metaclasses inside the dataset thath downloaded 


The class will search if the dataset that you want to work with is downloaded in your computer, if not it will create the folder dataset with it inside


Depend of the partition you define besides get the train, val, test arrays with the data atacched in memory you will get the CSV of the partition for more flexible train.

The code will also provide the batch based on the amount of batches did you define (default=1)


If you just want to download the dataset and dont want to get any type of training implementation you can download the different datasets from the Datasets file.

Inside the dataset you have the different Classes to download the different datasets, as the example below shows


## Run Locally

To get the Dataloader package you will need to:

```bash
    pip install git+https://github.com/Oriolrt/SIDTD_Dataset.git
```

Enter to any python terminal and now you can
```bash
  import Loader
```
It shouldnt exist any type of error

If you just want to get the benchmark you only need to go to the datasets and choice among [MIDV2020, Dogs, DF20M, Findit, Banknotes]:

```python
    import Loader.Datasets as ld

    ld.Midv(conditioned=True, download_original=True).dowload_dataset()
```

if you want to download the dataset to train it with our partitions the Loader_Modules is the file you have to call
in this case the example is:

```python
    import Loader.Loader_Modules as lm

    dt = lm.Dataloader(dataset="Midv", type_split="normal", batch_size="64",normal_split=[0.8,0.1,0.1], conditioned=True)

    train, val, test, batch_index = dt.get_structures()
```

# Models

### Cross-validation training

For each models trained we performed a 10-fold cross-validation training (8 folds for training, 1 fold for validation and 1 fold for testing). The method involved creating csv files with the image path, class name, class id, label number, and label name for each k-fold partition of the training, validation, and test sets.

### Models

For the Benchmark study 5 different models were trained: EfficientNet, ResNet50, ViT, TransFG and Attentive Recurrent Comparators(ARC). You can find the models implementation in the 'models' directory. The first three models (EfficientNet, ResNet50 and ViT) are grouped in the same directory ('Baseline'). The fourth model (TransFG) can be found in 'transfg' directory and the last model (ARC) is located in the 'arc_pytorch' directory. 

You can train or test the models in the 'code_examples' directory, by running the bash file exectrain.sh (for training) and exectest.sh for testing. You can add or modify the flags already wrote in the bash files. The name flag can be customized depending on your needs. The dataset flag should correspond to the dataset folder name. You can choose the model among a list of 5 different models: vit_large_patch16_224, efficientnet-b3, resnet50, trans_fg, coatten_fcn_model. Also you can modify the networks parameters to perform your own fine-tuning. 


### Results

Results performance by fold for validation and test set can be seen in the 'results_files' directory (loss, accuracy, ROC AUC). 

In 'plots' directory, you can see the training history for the training/validation loss and the training/validation accuracy.


### Training features

Python version used for training : Python 3.10. 

All models are coded with the Pytorch and were trained on GPU. If you want, you can find out more information about how to install Pytorch for CUDA here: https://pytorch.org/get-started/locally/.

For each models, a '--cuda' parameter can be set to pass from CPU to GPU training.

In each model directory, you can find a README explaining how to train the models.

/!\ Be careful to have enough CUDA memory on server when training ViT and TransFG models. A CUDA out of memory error could happen.

### Packages dependency

+ torchvision
+ Pytorch for CUDA
+ efficientnet_pytorch
+ albumentations
+ timm
+ tqdm
+ opencv-python
+ scikit-learn
+ ml_collections 
+ numpy
+ pandas
+ PIL
+ scipy
+ matplotlib

## FAQ

#### To resolve any doubt  

cboned@cvc.uab.cat
mtalarmain@cvc.uab.cat

## Acknowledgments
SOTERIA has received funding from the European Union’s Horizon 2020 	research and innovation programme under grant agreement No 101018342 

Disclaimer: This content reflects only the author's view. The European Agency is not responsible for any use that may be made of the information it contains. 