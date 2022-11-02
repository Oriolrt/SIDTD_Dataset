This folder contains the models used to evaluate the SIDTD dataset. Every model is coded in Pytorch. 

# EfficientNet and ResNet

ResNet and EfficientNet are widely used CNN models by the deep learning community. ResNet is short for Residual Network and was designed in order to avoid vanishing gradient problem which enables then to build deeper network. We chose to implement ResNet50, a Residual Network with 23 million parameters. EfficientNet is a network conceived to use the parameters more efficiently, good accuracy with a lower number of parameters. They build this network focusing on optimizing both accuracy and FLOPS (calculation speed) and it resulted with balanced networks in terms of width, depth, and resolution. We chose to implement EfficientNet-b3, a residual network with 10 million parameters, the most accurate efficientnet with a low number of parameters. 

The EfficientNet-B3 and ResNet50 are built-in models from pytorch packages, respectively in order efficientnet_pytorch and torchvision. Both model are pretrained on ImageNet/1k (2012) at resolution 299x299x3 for EfficientNet-B3 and ResNet50.  

# Transformer (ViT and TransFG)

Vision Transformer (ViT) is a recent innovation in computer vision inspired from Transformer architecture in Natural Language Processing. ViT split images into image patches and add position embedding, and input patch + position embeding into a Transformer encoder architecture were images are treated like tokens for NLP tasks. The ViT architecture implemented in this code is the ViT-L/16. The ViT-L/16 model is implemented in pytorch with a built-in model from timm package. The model is pretrained on ImageNet/1k (2012) at resolution 224x224x3 for EfficientNet-B3 and ResNet50.  

The TransFG network is a model derived from ViT model. The innovation with this model is the addition of a Part Selection module between Transformer Encoder and the Transformer Layer. This new module aims to guide the network during the selection of the relevant image patches and to learn only from the discriminative image patches. We took the same ViT model as the backbone network for the TransFG model, ViT-L/16, pretrained on ImageNet21k and ImageNet1k (2012).

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

The code is adapted from the official PyTorch code of the paper:  [*TransFG: A Transformer Architecture for Fine-grained Recognition*](https://arxiv.org/abs/2103.07976). Many thanks to [ViT-pytorch](https://github.com/jeonsworld/ViT-pytorch) for the PyTorch reimplementation. 

# Co-attention Attention Recurrent Comparator (Co-Attention ARC)

ARC is an algorithmic imitation of the human way that compares alternatively two images with a model based on a recurrent neural network controller and a CNN model that perform features exctraction. This algorithm iterates over different glimpses from pairs of images in order to search the region of interest which would help to decide if the two images are from the same class or not. In addition to this model, we join a co-attention module to focus on identifying the most relevant and crucial parts of the images.

We chose to use ResNet50 pretrained on ImageNet1k (2012) as the CNN network. 

The code is derived from PyTorch implementation of Attentive Recurrent Comparators (ARC) by Shyam et al.

A [blog](https://medium.com/@sanyamagarwal/understanding-attentive-recurrent-comparators-ea1b741da5c3) explaining Attentive Recurrent Comparators

### Preprocessing to fit Co-Attention ACR

We load the data as array files in the "omniglot" folder and then load the data in memory as a python dictionnary. This procedure is executed in order to make the training phase faster

### Acknowledgement

The code is adapted from [Attentive Recurrent Comparators](https://arxiv.org/abs/1703.00767). Many thanks to Shyam et al. for the PyTorch implementation