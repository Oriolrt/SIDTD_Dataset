# Attentive Recurrent Comparators with ResNet and CoAttention Model

PyTorch implementation of Attentive Recurrent Comparators (ARC) by Shyam et al.


A [blog](https://medium.com/@sanyamagarwal/understanding-attentive-recurrent-comparators-ea1b741da5c3) explaining Attentive Recurrent Comparators

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
### 1. 10-fold partition

If not already performed split the dataset into 10-fold:
```
cd ..
python kfold.py
cd arc_pytorch
```
For more information about the 10-fold partition, check the README in root directory.

### 2. Download data

After splitting the data path into csv files. Let's load the data into array to make faster the Batch loader during the training phase. For the unconditioned training you need to run this file:

```
python download_data.py
```

For the conditioned model you need to run this file:

```
python download_data_multi_class.py --nb_class 10
```
Do not forget to specify the number of meta-classes that the dataset has, here we set the number of meta-classes at 10 with the parameter "--nb_class". Depending on the dataset size the loading can take from a few minutes to some hours.

### 3. Train

To train unconditionally Co-Attention ARC with ResNet for 5000 steps, write in exectrain.sh the following text:

```
CUDA_VISIBLE_DEVICES=0 python train_coAttn_binary.py --name='arc_coatten_fcn_model' --dataset 'write_dataset_name_here'
```

To train conditionally Co-Attention ARC with ResNet for 5000 steps, write in execconditionedtrain.sh the following text:

```
CUDA_VISIBLE_DEVICES=1 python train_coAttn_binary_class.py --name 'conditionned_coatten_fcn_model' --dataset 'dataset_raw' --nb_class 10
```

The dataset name must be written exaclty the same way as the name of the dataset directory with the '--dataset' parameter. Do not forget to specify the number of meta-classes that the dataset has, here we set the number of meta-classes at 10 with the parameter "--nb_class". Then run the bash file.


### 4. Results

Results by fold can be seen in the 'results_files' directory. In 'plots' directory, you can see the training history for the training/validation loss and the training/validation accuracy.


