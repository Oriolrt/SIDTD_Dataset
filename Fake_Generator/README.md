
# Generating the Benchmark

In this we add the functions to create more fake data. To use this scripts you must have the midv2020 or midv500 dataset downloaded.

To create more information you will have some functionalities that you can play with to create more variety of information.

The structure of this section is depicted as follows:

```
Fake_Generator
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
```

Once the structure of this section is explained lets show some example to use it.

## Re-generate the Benchmark

To generate more new fake data 
```python

    from Fake_Generator.Fake_Loader.Midv2020 import Template_Generator

    gen = Template_Generator.Template_Generator(path=(path_to_parent Midv2020 folder ["/home/cboned/MIDV2020/dataset"]))


```

You can also specify a concrete metadata structure. If not the default metadata is used. This default can be found in Midv.py
