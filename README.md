# About the project

The purpose of this project is to generate a new synthetic identity document dataset with forged ID based on MIDV2020 dataset. This is constructed in order to build models that will classify genuine and forged identity document without the need to use real-world identity document.

## Structure

The structure of the project is organized as follows:
```
SIDTD
│
└───data
|   |
│   └───DataGenerator 
│   |
|   └───DataLoader 
|   |
|   └───explore
|       
└───models
|   |
│   └───arc_pytorch 
│   |
|   └───Baseline 
|   |
│   └───explore 
│   |
|   └───transfg
|
|___utils
```


All the names in this structure are folders. Every folder have his own Readme to make clearer all the structure, except folders *models/arc_pytorch*, *models/transfg*, *models/Baseline* and *utils*, as it is only piece of code used in other folders.


## Installation

To get the Dataloader package you will need to:

```
python setup.py install
```

Enter to any python terminal and now you can
```bash
  import SIDTD
```
It shouldnt exist any type of error

## Technical Validation

Data is partitioned to allow three model validation techniques: hold-out, k-fold cross-validation and few-shot. The code147
provided allow users to define any partition for these three model validation schemes. Data is randomly sampled and it is,148
by default, split into 80%-10%-10% for the hold-out validation and split it into 10 folds for the k-fold cross-validation. For149
few-shot, 6 nationalities ID documents are randomly chosen for the meta-training and the remainder 4 for the meta-testing.150
However, for fair comparison between models we also provide a predefined partition in which training, validation and test151
instances are always the same for both, the hold-out and the k-fold cross-validation.

To evaluate the SIDTD dataset, we have trained five state-of-the-art deep learning models for three tasks: template-based153
ID documents, video-based ID documents and few-shot

The following five models were used to evaluate the SIDTD dataset: EfficientNet-B37, ResNet508, Vision Transformer Large161
Patch 16 (ViT-L/16)9, TransFG10 and Co-Attention Attentive Recurrent Network (CoAARC)11, 12. EfficientNet, ResNet, and162
ViT models are general purpose models while TransFG is a model for fine-grained classification task while the CoAARC model163
was designed to detect forged ID documents.


## Results
## Models Performance in Terms of Accuracy and ROC AUC Scores with Standard Deviation

Models are trained and tested regarding a 10-fold cross-validation for the template-based ID document and the video-based ID document tasks. Model performance is also reported for the private dataset from IDNow.

| Dataset        | Templates-based task                    | Video-based task                    | Few-shot task                     | Private dataset                    |
|----------------|----------------------|------------------|------------------|------------------|----------------|------------------|-----------------|------------------|
|                | accuracy             | ROC AUC          | accuracy         | ROC AUC          | accuracy       | ROC AUC          | accuracy        | ROC AUC          |
| EfficientNet   | 0.994 ± 0.010        | 1.000 ± 0.000    | 0.999 ± 0.001    | 1.000 ± 0.000    | 0.820 ± 0.026  | 0.8991 ± 0.0253  | 0.7758 ± 0.0342 | 0.8648 ± 0.0344  |
| ResNet         | 0.981 ± 0.012        | 0.999 ± 0.011    | 0.999 ± 0.002    | 1.000 ± 0.000    | 0.906 ± 0.026  | 0.9634 ± 0.0168  | 0.8246 ± 0.0187 | 0.8929 ± 0.0141  |
| ViT            | 0.552 ± 0.023        | 0.501 ± 0.042    | 0.975 ± 0.020    | 0.989 ± 0.015    | 0.910 ± 0.021  | 0.9690 ± 0.0108  | 0.5265 ± 0.0266 | 0.5436 ± 0.0317  |
| TransFG        | 0.966 ± 0.015        | 0.992 ± 0.004    | 0.999 ± 0.002    | 1.000 ± 0.000    | 0.836 ± 0.033  | 0.9069 ± 0.0298  | 0.6371 ± 0.0311 | 0.7407 ± 0.0179  |
| CoAARN         | 0.986 ± 0.016        | 0.999 ± 0.003    | 0.992 ± 0.012    | 0.999 ± 0.003    | 0.6279 ± 0.0549| 0.6754 ± 0.0684  | 0.7587 ± 0.0293 | 0.8255 ± 0.0246  |






#### To resolve any doubt  

+ cboned@cvc.uab.cat
+ oriolrt@cvc.uab.cat
+ mtalarmain@cvc.uab.cat
## Acknowledgments
SOTERIA has received funding from the European Union’s Horizon 2020 	research and innovation programme under grant agreement No 101018342 

Disclaimer: This content reflects only the author's view. The European Agency is not responsible for any use that may be made of the information it contains. 