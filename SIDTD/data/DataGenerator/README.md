
# Generating the Benchmark

Within this folder, you will find the functionalities necessary to generate more images with some falsifications.

To create more information you will have some other functionalities that you can play with to create more variety of information.

To generate information you need to follow certain structure: 

```
Folder_path
|   Images
|   | reals ( There are inside the list of the different images)
|   | fakes 
|   
|   Annotations
|   | reals ( There are inside the list of the different annotations)
|   | fakes     
```

This structure follows the Midv2020 structure. It can be generalized to each kind of image paths

### The structure of this section is depicted as follows:

```
DataGenerator
|   __init__.py    
│   Midv.py (where this class is the core class of the generator)
│   utils.py
|   
│─────Midv2020
│       │   Template_Generator.py
│       │   Video_Generator.py (beta)
|
|______Midv500 (beta)
|       |   Template_Generator
|       |   Video_Generator.py (beta)
```

Once the structure of this section is explained lets show some example to use it.

## Re-generate the Benchmark

To generate more new fake data you need to have the structure that is depicted above and call this different functions:

A test example with our downloading function is...
```python

    from  SIDTD.data.DataGenerator.Midv2020.Template_Generator import Template_Generator
    from SIDTD.data.DataLoader.Datasets import *

    ## Downloadign our data just as an example
    data = SIDTD(download_original=False).download_dataset("templates")

    # get the abosulte path where the data is stored following the structure depicted above
    
    # you need to go where the data is 
    path_dataset = "path to the downloaded dataset"
    
    # generating the data
    gen = Template_Generator.Template_Generator(absolute_path=path_dataset))

    gen.create(sample=10)
    
    gen.store_generated_dataset(path_store=None) #[None for dedault]

```

**We strongly recomend to generate new data with  Templates**

*You can also specify a concrete metadata structure. If not the default metadata is used. This default can be found in Midv.py*
