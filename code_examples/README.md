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

 The first two are the reproduction of the calls that we are doing inside all the core of the main script to create the two different variations (Inpaint and Crop and Replace) (__make_inpaint__ and __make_crop_and_replace__). These two functions are being called in the same way than the main code so the parameters are the same. To use the fucntion you must pass the information needed to access to the images and their metadata.

There are two more functions called __custom_crop_and_replace__ and __custom_inpaint_and_rewrite__  that are the main operation to create the two transformations. They are impleemented if you want to generate some crop and replace or some inpaint with your own data. 

To be sure about how to use the functions you can check the descriptions inside them or by calling to the "help" function from python

```python
    help(__function__)
```

# Generate dataset

The script __generate_fake_dataset.py__ have the two functionalities to recreate the generation that we have done to create the new benchmark. As far as are some params that are random to create more variability, a new generation may be different in the number of images for each class.

# Train Models

The script __train.py__ train one of the five models implemented in the Benchmark study: EfficientNet, ResNet, Vision Transformer (ViT), Trans FG and Co-Attention Attentive Recurrent Comparator (Co-Attention ARC). You can choose the model with the flag --model, but be careful to write without typo the model name that correspond to your chosen model:  
+ EfficientNet -> 'efficientnet-b3'
+ ResNet -> 'resnet50'
+ ViT -> 'vit_large_patch16_224'
+ Trans FG -> 'trans_fg'
+ Co-Attention ARC 'coatten_fcn_model' 

You can choose to train on the dataset of your choice with the flag --dataset. The name of the dataset must be the exact same name as the one of the folder that store the dataset you want to use. 

You can choose to train on a specific partition or on every partition with the flag --all. If --all='yes', then you train on all partition. If --all='no', then you train on one partition. You need to specify the partition with the flag --partition. To train on multiple partition but not on all partition it is necessary to run each time the same code and changing the flag --partition.

During training, the model is saved in the *trained_models* folder. It is the model with the highest accuracy that is saved. You can change the location of the folder to another path with the flag --save_model_path.

If you need to fine-tune the parameters you can modify the models' parameter thanks to the flags such as the --epochs (number of epochs), --batch_size (batch size for training and validation), --learning rate (learning rate chosen for the training of EfficientNet, ResNet and ViT). Flag model parameter can be different from one model to another, so flag model paramet are grouped by model in train.py (see code).

Remember to choose a name for your experiment with the flag --name. It could help you to classify your different experiment or to remembe the training parameter you chose. The name you are choosing is free of choice.

In the bash file *exectrain.sh* you can find an exemple of the minimum --flag you should mention to make your training working. If you do not have CUDA and GPU on your computer, you must remove *CUDA_VISIBLE_DEVICES* from the bash file and inform the model to use cpu device instead of CUDA with the flag --device='cpu'.

# Test Models

The script __test.py__ load a trained model and evaluated it on the test partitions. You can train the same model as mentionned in the Train models section. You can choose the model with the flag --model with the same name as described on the previous section.

You can test a model with your trained weight or with our trained models to reproduce results on the chosen dataset. You can choosE it with the flag --pretrained. If --pretrained='yes', use trained network on MIDV2020 to reproduce results. If --pretrained='no', use the custom trained network on your own partitions.

You can choose to test on the dataset of your choice with the flag --dataset. The name of the dataset must be the exact same name as the one of the folder that store the dataset you want to use. 

You can choose to test on a specific partition or on every partition with the flag --all. If --all='yes', then you test on all partition. If --all='no', then you test on one partition. You need to specify the partition with the flag --partition. To test on multiple partition but not on all partition it is necessary to run each time the same code and changing the flag --partition.

Remember to choose a name for your experiment with the flag --name. It should be the same name for the test than the training you performed earlier in order to use the correct trained models.

In the bash file *exectest.sh* you can find an exemple of the minimum --flag you should mention to make your training working. If you do not have CUDA and GPUs on your computer, you must remove *CUDA_VISIBLE_DEVICES* from the bash file and inform the model to use cpu device instead of CUDA with the flag --device='cpu'.