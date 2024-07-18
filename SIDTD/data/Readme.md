# Data Section

```
   data
   |
   └───DataGenerator 
   |    |
   |    |_Midv500
   |    |
   |    |_Midv2020
   |    |
   |    |_ Midv.py
   |
   └───DataLoader 
   |    |
   |    |_Datasets.py
   |    |_Loader_Modules.py
   |
   └───explore
   |    |_fake_id_ex.py
   |    |_generate_fake_dataset.py
```


In this folder, you will find all the scripts related to the dataset. Each folder has its own readme with a tutorial and an explanation of the structure in a more localized manner.

The *DataGenerator* contains functions to generate more fake data, while the Dataloader contains functions to download our partitions, as already mentioned in the different readme files.

Finally the explore contain some toy examples in order to make it easy to reproduce and use the different functionalities that are this section.