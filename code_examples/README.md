## Usage examples
This folder contains code examples to use the functions used to generate fake id documents. It also contains the scripts used to generate the SIDTD dataset and to train the benchmark models.

```
code_examples
|   fake_id_ex.py 
|   generate_fake_dataset.py
|   train.py
|   test.py
```

# Generate fake Id images

This file is divided in 4 different functions:

     The first two are the reproduction of the calls that we are doing inside all the core of the main script to create the two different variations (Inpaint and Crop and Replace) (__make_inpaint__ and __make_crop_and_replace__). These two fucntions are being called in the same way than the main code so the params are the same. To use the fucntion you must pass the information needed to access to the images and their metadata.

    There are two more functions called __custo_crop_and_replace__ and __custom_inpaint_and_rewrite__  that are the main operation to create the two transformations. They are impleemented if you want to generate some crop and replace or some inpaint with your own data. 

    to make more sure abount how to use this functions you can check the descriptions inside them or by calling to the "help" function from python

    ```python

        help(__function__)
    ```

# Generate dataset

The script __generate_fake_dataset.py__ have the two functionalities to recreate the generation that we have done to create the new benchmark. As far as are some params that are random to create more variability, a new generation may be different in the number of images for each class.

# Train Models

The script __train.py__ train a model on the training partitions and save it in the _location_ folder.

# Test Models

The script __test.py__ load a trained model and evaluated it on the test partitions.
