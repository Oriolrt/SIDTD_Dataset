
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

## FAQ

#### To resolve any doubt  

cboned@cvc.uab.cat



## Run Locally

Clone the project

```bash
  git clone https://link-to-project
```

Go to the project directory

```bash
  cd my-project
```

Install dependencies

```bash
  pip install requirements.txt
```

Execute 

```bash
    python3 Loader/Loader_Modules.py --dataset Midv --conditioned 1 -ts kfold --kfold_split 10 --batch_size 10
```


