## Usage examples
This folder contains code examples to use the functions used to generate fake id documents.


# Generate fake Id images

This file is divided in 4 different functions:


Two functions called __custom_crop_and_replace__ and __custom_inpaint_and_rewrite__  that are the main operation to create the two transformations. They are impleemented if you want to generate some crop and replace or some inpaint with your own data. 

Two functions to plot the results obtained.

To be sure about how to use the functions you can check the descriptions inside them or by calling 

```
    # Help from Crop and Replace
    python fake_id_ex.py --transformation 'cr' --help_func

    #Help from Inpaint and Rewrite
    python fake_id_ex.py --transformation 'ir' --help_func
```

if you want more information about the arguments 
```
    python fake_id_ex.py --help
```
## How to execute the custom transformations?
```
    # CR
    python fake_id_ex.py --transformation 'cr' --src_path 'Samples_Test/alb00.jpg' --src_annotations 'Samples_Test/alb_0_id.json' --trg_path 'Samples_Test/alb01.jpg' --trg_annotations 'Samples_Test/alb_1_id.json' --shift_boundary 10

    # Inpaint
    python fake_id_ex.py --transformation 'ir' --src_path 'Samples_Test/alb00.jpg' --src_annotations 'Samples_Test/alb_0_id.json' --field_to_change 'name'

```

# Generate dataset

The script __generate_fake_dataset.py__ have the two functionalities to recreate the generation that we have done to create the new benchmark. As far as are some params that are random to create more variability, a new generation may be different in the number of images for each class.

## How to execute the regeneration of the SIDTD?

```
python generate_fake_dataset.py --dataset 'Midv2020' --src 'SIDTD_Dataset/dataset/SIDTD' --sample 1000
```
