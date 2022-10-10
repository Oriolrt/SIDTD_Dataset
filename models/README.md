This folder contains the models used to evaluate the SIDTD dataset. Every model is coded in Pytorch. 

# EfficientNet, ResNet50 and ViT

The EfficientNet-B3, ResNet50 and ViT are built-in models from pytorch packages, respectively in order efficientnet_pytorch, torchvision and timm. Each model is pretrained on ImageNet/1k (2012) at resolution 224x224x3 for ViT and 299x299x3 for EfficientNet-B3 and ResNet50.  

ResNet and EfficientNet are typical CNN models widely used by the deep learning community. EfficientNet is a network conceived to use the parameters more efficiently to not saturate the model too fast during the training. The ViT is a more complex model used also for image classification. This model employs a Transformer encoder architecture that includes Multi-Head Attention, Scaled Dot-Product Attention and other architectural features seen in the Transformer architecture traditionally used for NLP (see Figure 7). The ViT architecture we chose is the ViT-L/16. 

# Transformer (TransFG)

The TransFG network is a model derived from ViT model. What is new with this model is the addition of a part selection module that guide the network to effectively select discriminative image patches and compute their relations by applying contrastive feature learning to enlarge the distance of representations between confusing sub-categories. 

We took the same ViT model as the backbone network for the TransFG model, ViT-L/16, pretrained on ImageNet21k and ImageNet1k (2012). 

The code is adapted from the official PyTorch code of the paper:  [*TransFG: A Transformer Architecture for Fine-grained Recognition*](https://arxiv.org/abs/2103.07976)  

### Pre-trained ViT models

A pre-trained ViT models is used and is located in the 'transfg\_pretrained' directory. If necessary it can be moved manually by the user. To be located by the models the user need to indicate correctly the new location folder with the '--pretrained_dir' parameter. 

### Citation

If you find TransFG helpful in your research, please cite it as:

```
@article{he2021transfg,
  title={TransFG: A Transformer Architecture for Fine-grained Recognition},
  author={He, Ju and Chen, Jie-Neng and Liu, Shuai and Kortylewski, Adam and Yang, Cheng and Bai, Yutong and Wang, Changhu and Yuille, Alan},
  journal={arXiv preprint arXiv:2103.07976},
  year={2021}
}
```

### Acknowledgement

Many thanks to [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) for the PyTorch reimplementation of [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)

# Co-attention Attention Recurrent Comparator (Co-Attention ARC)

ARC is an algorithmic imitation of the human way that compares alternatively two images with a model based on a recurrent neural network controller and a CNN to exctract the features. The iteration is performed over glimpses from pairs of images to focus on region of interest to decide if the two images are from the same class or not. In addition to this model, we join a co-attention module to focus on identifying the most relevant and crucial parts of the images.

We chose to use ResNet pretrained on ImageNet1k (2012) as the CNN network. 

The code is derived from PyTorch implementation of Attentive Recurrent Comparators (ARC) by Shyam et al.

A [blog](https://medium.com/@sanyamagarwal/understanding-attentive-recurrent-comparators-ea1b741da5c3) explaining Attentive Recurrent Comparators

### Preprocessing to fit Co-Attention ACR

We load the data in memory to make faster the Batch loader during the training phase. The data will be stored as numpy array in the omniglot directory.

### Acknowledgement

Many thanks to Shyam et al. for the PyTorch implementation of [Attentive Recurrent Comparators](https://arxiv.org/abs/1703.00767)