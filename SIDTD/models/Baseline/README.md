# EfficientNet, ResNet50 and ViT

## Framework

Install Pytorch for CUDA.
For Linux: 
```
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch
```


For Windows: 
```
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

More info on this [link](https://pytorch.org/get-started/locally/)

## Usage

We proposed different configurations for training: it is possible to change the loss from Cross Entropy Loss to Smooth AP Loss or to train (un-)conditionnaly the networks (suggesting sub-class information or not). You should note that in order to train with Smooth AP Loss the models are slightly changed regarding the output layer. Indeed, the models output a feature vector of size 512 (that can be modified with the flag --embed_dim) which is then classified by a simple classifier (SVM, k-NN or K-Means that you can choose depending on your preference with the flagclassification_model --). Also in this configuration, the default value for the batch size is increased to 112 as the loss is mostly effective when the batch size is high so it is advised to keep it as high as possible; you can also choose the optimizer (Adam or SGD) and the scheduler (MultiStepLR or ReduceLROnPlateau) witht the respective flags --opt and --scheduler.


### 1. 10-fold partition

If not already performed split the dataset into 10-fold:
```
cd ..
python kfold.py
cd Baseline
```
For more information about the 10-fold partition, check the README in root directory.

### 2. Train

To train **unconditionally** EfficientNet, ResNet50 and ViT for 100 epochs with **Cross Entropy Loss** on 'dataset_raw' write in exectrain.sh the following text:

```
CUDA_VISIBLE_DEVICES=0 python training_kfold_rectified.py --name='EfficientNet' --dataset='write_dataset_name_here' --model='efficientnet-b3'

CUDA_VISIBLE_DEVICES=0 python training_kfold_rectified.py --name='ResNet50' --dataset='write_dataset_name_here' --model='resnet50'

CUDA_VISIBLE_DEVICES=0 python training_kfold_rectified.py --name='vit_large_patch16' --dataset='write_dataset_name_here' --model='vit_large_patch16_224'
```

To train **unconditionally** EfficientNet, ResNet50 and ViT for 100 epochs with **Smooth AP Loss** on 'dataset_raw' write in exectrain.sh the following text:

```
CUDA_VISIBLE_DEVICES=0 python training_kfold_rectified.py --name='EfficientNet' --dataset='write_dataset_name_here' --model='efficientnet-b3'

CUDA_VISIBLE_DEVICES=0 python training_kfold_rectified.py --name='ResNet50' --dataset='write_dataset_name_here' --model='resnet50'

CUDA_VISIBLE_DEVICES=0 python training_kfold_rectified.py --name='vit_large_patch16' --dataset='write_dataset_name_here' --model='vit_large_patch16_224'
```

To train **conditionally** EfficientNet, ResNet50 and ViT for 100 epochs with **Cross Entropy Loss** on 'dataset_raw' write in execconditionedtrain.sh the following text:

```
CUDA_VISIBLE_DEVICES=0 python training_kfold_conditioned.py --name='conditionned_EfficientNet' --dataset='write_dataset_name_here' --model='efficientnet-b3'

CUDA_VISIBLE_DEVICES=0 python training_kfold_conditioned.py --name='conditionned_ResNet50' --dataset='write_dataset_name_here' --model='resnet50'

CUDA_VISIBLE_DEVICES=0 python training_kfold_conditioned.py --name='conditionned_vit_large_patch16' --dataset='write_dataset_name_here' --model='vit_large_patch16_224'
```

The dataset name must be written exaclty the same way as the name of the dataset directory with the '--dataset' parameter.
Then run the bash files.

If you don't want to train it successively, you can run each line independently in different bash files.


### 3. Results

Results by fold can be seen in the 'results_files' directory. In 'plots' directory, you can see the training history for the training/validation loss and the validation accuracy.


