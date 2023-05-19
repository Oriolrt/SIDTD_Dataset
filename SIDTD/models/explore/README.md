## Usage examples
This folder contains scripts used to train and test the benchmark models.

# Train Models

The script __train.py__ train one of the five models implemented in the Benchmark study: EfficientNet, ResNet, Vision Transformer (ViT), Trans FG and Co-Attention Attentive Recurrent Comparator (Co-Attention ARC).

### How to train the models?

During training, the model is saved in the *trained_models* folder. It is the model with the highest accuracy that is saved. You can change the location of the folder to another path with the flag --save_model_path.

To train one of the five models with CUDA, you should run the line that correspond to your model with the corresponding name of the dataset used:
```
# Train EfficientNet model
CUDA_VISIBLE_DEVICES=0 python train.py --name='EfficientNet' --dataset='dataset_raw' --model='efficientnet-b3' 

# Train ResNet50 model
CUDA_VISIBLE_DEVICES=0 python train.py --name='ResNet50' --dataset='dataset_raw' --model='resnet50'

# Train ViT model
CUDA_VISIBLE_DEVICES=0 python train.py --name='vit_large_patch16' --dataset='dataset_raw' --model='vit_large_patch16_224'

# Train Trans FG model
CUDA_VISIBLE_DEVICES=0 python train.py --name='trans_fg' --dataset='dataset_raw' --model='trans_fg'

# Train Co-Attention ARC model
CUDA_VISIBLE_DEVICES=0 python train.py --name='coatten_fcn_model' --dataset 'dataset_raw' --model='coatten_fcn_model'
```
Remember to choose a name for your experiment with the flag --name. It could help you to classify your different experiment or to remember the training parameter you chose. The name you are choosing is free of choice.

You can choose to train on the dataset of your choice with the flag --dataset. The name of the dataset must be the exact same name as the one of the folder that store the dataset you want to use. 

You can choose the model with the flag --model, but be careful to write without typo the model name that correspond to your chosen model:  
+ EfficientNet -> 'efficientnet-b3'
+ ResNet -> 'resnet50'
+ ViT -> 'vit_large_patch16_224'
+ Trans FG -> 'trans_fg'
+ CoAARC 'coatten_fcn_model' 

If you have downloaded the static csv, you can reproduce the training by adding the flag --static=='yes'. Please, remember to train according to the static csv you have downloaded. For instance, if you have downloaded the k-fold cross validation version of cropped clips you should add the corresponding flag, hence, here it would be: --type_data='clips_cropped' --type_split='kfold'.

More flags can be added to modify the default parameters. If you need to fine-tune the parameters you can modify the models' parameter thanks to the flags such as the --epochs (number of epochs), --batch_size (batch size for training and validation), --learning rate (learning rate chosen for the training of EfficientNet, ResNet and ViT). Flag model parameter can be different from one model to another, so flag model paramet are grouped by model in train.py (see code).

You can choose to train on a specific partition or on every partition with the flag --all. If --all='yes', then you train on all partition. If --all='no', then you train on one partition. You need to specify the partition with the flag --partition. To train on multiple partition but not on all partition it is necessary to run each time the same code and changing the flag --partition.

### Results

Results by fold are generated in csv file in the 'results_files' directory. The training history (loss and accuracy) can be found in the 'plots' directory. Trained models will be located in the trained_models directory.


# Test Models

The script __test.py__ load a trained model and evaluate it on the test partitions. It is possible to test the same models as mentionned in the Train models section. 

### Test your trained model on custom 10-fold partitions with CUDA
To test one of the five models with *CUDA* after training your own model, you should run the line that correspond to your model with the corresponding name of the dataset used:
```
# Test EfficientNet model with CUDA
CUDA_VISIBLE_DEVICES=0 python test.py --name='EfficientNet' --dataset='dataset_raw' --model='efficientnet-b3'

# Test ResNet50 model with CUDA
CUDA_VISIBLE_DEVICES=0 python test.py --name='ResNet50' --dataset='dataset_raw' --model='resnet50'

# Test ViT model with CUDA
CUDA_VISIBLE_DEVICES=0 python test.py --name='vit_large_patch16' --dataset='dataset_raw' --model='vit_large_patch16_224' 

# Test Trans FG model with CUDA
CUDA_VISIBLE_DEVICES=0 python test.py --name='trans_fg' --dataset='dataset_raw' --model='trans_fg'

# Test Co-Attention ARC model with CUDA
CUDA_VISIBLE_DEVICES=0 python test.py --name='coatten_fcn_model' --dataset 'dataset_raw' --model='coatten_fcn_model'
```

### Test your trained model on custom 10-fold partitions without CUDA
To test one of the five models with *your CPU* after training your own model, you should run the line that correspond to your model with the corresponding name of the dataset used:
```
# Test EfficientNet model without CPU
python test.py --name='EfficientNet' --dataset='dataset_raw' --model='efficientnet-b3' --device='cpu'

# Test ResNet50 model without CPU
python test.py --name='ResNet50' --dataset='dataset_raw' --model='resnet50' --device='cpu'

# Test ViT model without CPU
python test.py --name='vit_large_patch16' --dataset='dataset_raw' --model='vit_large_patch16_224' --device='cpu'

# Test Trans FG model without CPU
python test.py --name='trans_fg' --dataset='dataset_raw' --model='trans_fg' --device='cpu'

# Test Co-Attention ARC model without CPU
python test.py --name='coatten_fcn_model' --dataset 'dataset_raw' --model='coatten_fcn_model' --device='cpu'
```

### Test 10-fold partitions with our trained model on custom 10-fold partitions with CUDA
To test one of *our trained model* with CUDA, you should run the line that correspond to your model with the corresponding name of the dataset used:
```
# Test EfficientNet model
CUDA_VISIBLE_DEVICES=0 python test.py --name='EfficientNet' --dataset='dataset_raw' --model='efficientnet-b3' --pretrained='yes'

# Test ResNet50 model
CUDA_VISIBLE_DEVICES=0 python test.py --name='ResNet50' --dataset='dataset_raw' --model='resnet50' --pretrained='yes'

# Test ViT model
CUDA_VISIBLE_DEVICES=0 python test.py --name='vit_large_patch16' --dataset='dataset_raw' --model='vit_large_patch16_224' --pretrained='yes'

# Test Trans FG model
CUDA_VISIBLE_DEVICES=0 python test.py --name='trans_fg' --dataset='dataset_raw' --model='trans_fg' --pretrained='yes'

# Test Co-Attention ARC model
CUDA_VISIBLE_DEVICES=0 python test.py --name='coatten_fcn_model' --dataset 'dataset_raw' --model='coatten_fcn_model' --pretrained='yes'
```

### Reproduce our results on SIDTD on our 10-fold partitions on templates SIDTD
To *reproduce our results using our trained model* with CUDA, you should run the line that correspond to your model with the corresponding name of the dataset used:
```
# Test EfficientNet model
CUDA_VISIBLE_DEVICES=0 python test.py --name='EfficientNet' --dataset='SIDTD' --model='efficientnet-b3' --pretrained='yes' --static='yes'

# Test ResNet50 model
CUDA_VISIBLE_DEVICES=0 python test.py --name='ResNet50' --dataset='SIDTD' --model='resnet50' --pretrained='yes' --static='yes'

# Test ViT model
CUDA_VISIBLE_DEVICES=0 python test.py --name='vit_large_patch16' --dataset='SIDTD' --model='vit_large_patch16_224' --pretrained='yes' --static='yes'

# Test Trans FG model
CUDA_VISIBLE_DEVICES=0 python test.py --name='trans_fg' --dataset='SIDTD' --model='trans_fg' --pretrained='yes' --static='yes'

# Test Co-Attention ARC model
CUDA_VISIBLE_DEVICES=0 python test.py --name='coatten_fcn_model' --dataset 'SIDTD' --model='coatten_fcn_model' --pretrained='yes' --static='yes'
```

### Reproduce our results on SIDTD on our 10-fold partitions on video clips SIDTD
To *reproduce our results using our trained model* with CUDA, you should run the line that correspond to your model with the corresponding name of the dataset used:
```
# Test EfficientNet model
CUDA_VISIBLE_DEVICES=0 python test.py --name='EfficientNet' --dataset='SIDTD' --model='efficientnet-b3' --pretrained='yes' --static='yes' -td clips

# Test ResNet50 model
CUDA_VISIBLE_DEVICES=0 python test.py --name='ResNet50' --dataset='SIDTD' --model='resnet50' --pretrained='yes' --static='yes' -td clips

# Test ViT model
CUDA_VISIBLE_DEVICES=0 python test.py --name='vit_large_patch16' --dataset='SIDTD' --model='vit_large_patch16_224' --pretrained='yes' --static='yes' -td clips

# Test Trans FG model
CUDA_VISIBLE_DEVICES=0 python test.py --name='trans_fg' --dataset='SIDTD' --model='trans_fg' --pretrained='yes' --static='yes' -td clips

# Test Co-Attention ARC model
CUDA_VISIBLE_DEVICES=0 python test.py --name='coatten_fcn_model' --dataset 'SIDTD' --model='coatten_fcn_model' --pretrained='yes' --static='yes' -td clips
```

### Reproduce our results on SIDTD on our 10-fold partitions on cropped video clips SIDTD
To *reproduce our results using our trained model* with CUDA, you should run the line that correspond to your model with the corresponding name of the dataset used:
```
# Test EfficientNet model
CUDA_VISIBLE_DEVICES=0 python test.py --name='EfficientNet' --dataset='SIDTD' --model='efficientnet-b3' --pretrained='yes' --static='yes' -td clips_cropped

# Test ResNet50 model
CUDA_VISIBLE_DEVICES=0 python test.py --name='ResNet50' --dataset='SIDTD' --model='resnet50' --pretrained='yes' --static='yes' -td clips_cropped

# Test ViT model
CUDA_VISIBLE_DEVICES=0 python test.py --name='vit_large_patch16' --dataset='SIDTD' --model='vit_large_patch16_224' --pretrained='yes' --static='yes' -td clips_cropped

# Test Trans FG model
CUDA_VISIBLE_DEVICES=0 python test.py --name='trans_fg' --dataset='SIDTD' --model='trans_fg' --pretrained='yes' --static='yes' -td clips_cropped

# Test Co-Attention ARC model
CUDA_VISIBLE_DEVICES=0 python test.py --name='coatten_fcn_model' --dataset 'SIDTD' --model='coatten_fcn_model' --pretrained='yes' --static='yes' -td clips_cropped
```

You can test a model with your trained weight or with our trained models to reproduce results on the chosen dataset. You can choose it with the flag --pretrained. If --pretrained='yes', use trained network on MIDV2020 to reproduce results. If --pretrained='no', use the custom trained network on your own partitions.

Remember to choose a name for your experiment with the flag --name. It should be the same name for the test than the training you performed earlier in order to use the correct trained models.

You can choose to test on the dataset of your choice with the flag --dataset. The name of the dataset must be the exact same name as the one of the folder that store the dataset you want to use. 

You can choose the model with the flag --model with the same name as described on the previous section.

As the mentionned in the previous section, you can choose to test on a specific partition or on every partition with the flag --all.