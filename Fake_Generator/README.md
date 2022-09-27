
# Generating the Benchmark

In this we add the functions to create more fake data. To use this scripts you must have the midv2020 or midv500 dataset downloaded.

To create more information you will have some functionalities that you can play with to create more variety of information.

First of all the structure of this section is:

```
Fake_Generator
│   Main.py
│   Options.py --> this files are if you are wondering to copy the repo
|   __init__.py    
│
└───Fake_Loader
│   │   Midv.py (where this class is the core class of the generator)
│   │   utils.py
│   │   __init__.py
|   |
│   └───Midv2020
│       │   Template_Generator.py
│       │   Video_Generator.py (beta)
│   |
|   |
|   |___Midv500
|       |   Template_Generator
|       |   Video_Generator.py (beta)
│   
└───Test_Example
    │   generate_fake_test.py
    │   __init__.py
    |
    |___Samples_Test (with naive images to show the functionality)
```

Once the structure of this section is explained lets show some example to use it.

## Re-generate the Benchmark

Insert gif or link to demo

To generate more new fake data 
```python

    from Fake_Generator.Fake_Loader.Midv2020 import Template_Generator

    gen = Template_Generator.Template_Generator(path=(path_to_parent Midv2020 folder ["/home/cboned/MIDV2020/dataset"]))


```

you can also specify a concrete metadata structure. If not the default metadata is used. This default can be found in Midv.py

```python

    gen.fit(sample=1000) -> No return , stored in var(self._fake_img_loader) 

    gen.store_generated_dataset()

```

## Try Crop and replace and Inpaint separately

In the Test Example you have one the functions to know how the functions call each others inside

you can try to download the Samples Folder and use it to do the crop and replace and the Inpaint

```python
    
    from Fake_Generator.Test_Example.generate_fake_test import *

    #Crop and Replace
    img1, img2, field1, field2 = generate_crop_and_replace("Test_Example/Samples_Test/MIDV2020/alb_id/images/00.jpg",
                                                           "Test_Example/Samples_Test/MIDV2020/aze_passport/images/01.jpg","Test_Example/Samples_Test/MIDV2020/alb_id/annotations/alb_id.json",
                                                           additional_annotation_path="Test_Example/Samples_Test/MIDV2020/aze_passport/annotations/aze_passport.json")
    
    #The inpainting
    img, field = generate_inpaint("Test_Example/Samples_Test/MIDV2020/alb_id/images/00.jpg","Test_Example/Samples_Test/MIDV2020/alb_id/annotations/alb_id.json")


```