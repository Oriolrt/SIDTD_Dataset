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
## FAQ

#### To resolve any doubt  

cboned@cvc.uab.cat



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

If you just want to get the benchmark you only need to go to the datasets and choice among [Midv, Dogs, Fungus, Findit, Banknotes]:

```python
    import Loader.Datasets as ld

    ld.Midv(conditioned=True, download_original=True).dowload_dataset()
```

if you want to download the dataset to train it with our partitions the Loader_Modules is the file you  have to call
in this case the example is:

```python
    import Loader.Loader_Modules as lm

    dt = lm.Dataloader(dataset="Midv", type_split="normal", batch_size="64",normal_split=[0.8,0.1,0.1], conditioned=True)

    train, val, test, batch_index = dt.get_structures()
```
