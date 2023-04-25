# TransFG: A Transformer Architecture for Fine-grained Recognition

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/fine-grained-image-classification-on-cub-200)](https://paperswithcode.com/sota/fine-grained-image-classification-on-cub-200?p=transfg-a-transformer-architecture-for-fine) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/fine-grained-image-classification-on-nabirds)](https://paperswithcode.com/sota/fine-grained-image-classification-on-nabirds?p=transfg-a-transformer-architecture-for-fine) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/fine-grained-image-classification-on-stanford-1)](https://paperswithcode.com/sota/fine-grained-image-classification-on-stanford-1?p=transfg-a-transformer-architecture-for-fine) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/transfg-a-transformer-architecture-for-fine/image-classification-on-inaturalist)](https://paperswithcode.com/sota/image-classification-on-inaturalist?p=transfg-a-transformer-architecture-for-fine)

Official PyTorch code for the paper:  [*TransFG: A Transformer Architecture for Fine-grained Recognition*](https://arxiv.org/abs/2103.07976)  

## Framework

![](./TransFG.png)

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
### 1. Pre-trained ViT models

A pre-trained ViT models is used and is located in the 'transfg\_pretrained' directory. If necessary it can be moved manually by the user. To be located by the models the user need to indicate correctly the new location folder with the '--pretrained_dir' parameter. 

### 2. 10-fold partition

If not already performed split the dataset into 10-fold by running:
```
cd ..
python kfold.py
cd transfg
```
For more information about the 10-fold partition, check the README in root directory.

### 3. Train

To train unconditionally TransFG for 100 epochs write in exectrain.sh the following text:

```
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python train_kfold_transfg.py --name='trans_fg' --dataset 'write_dataset_name_here'
```

To train conditionally TransFG for 100 epochs write in exectrain.sh the following text:

```
CUDA_VISIBLE_DEVICES=0 CUDA_LAUNCH_BLOCKING=1 python train_conditionned_kfold_transfg.py --name='conditioned_trans_fg' --dataset='dataset_raw'
```

The dataset name must be written exactly the same way as the name of the dataset directory with the '--dataset' parameter.
Then run the bash file.


### 4. Results

Results by fold can be seen in the 'results\_files' directory. In 'plots' directory, you can see the training history for the training/validation loss and the training/validation accuracy.

## Citation

If you find TransFG helpful in your research, please cite it as:

```
@article{he2021transfg,
  title={TransFG: A Transformer Architecture for Fine-grained Recognition},
  author={He, Ju and Chen, Jie-Neng and Liu, Shuai and Kortylewski, Adam and Yang, Cheng and Bai, Yutong and Wang, Changhu and Yuille, Alan},
  journal={arXiv preprint arXiv:2103.07976},
  year={2021}
}
```

## Acknowledgement

Many thanks to [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) for the PyTorch reimplementation of [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

