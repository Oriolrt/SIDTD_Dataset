import argparse
import os

def BaseOptions():

    """
    This class includes options that are used by the five models and for train or test file.
    """

    parser = argparse.ArgumentParser()
    # flag for defining training or testing parameters
    parser.add_argument("--nsplits", default = 10, type=int, help="Number of k-fold partition")
    parser.add_argument("--device", default = 'cuda', type=str, help='Use CPU or CUDA')
    parser.add_argument("--nclasses", default = 2, type=int, help="Number of class in the dataset")
    parser.add_argument("--model", choices = ['vit_large_patch16_224', 'efficientnet-b3', 'resnet50', 'trans_fg', 'coatten_fcn_model'], default = 'resnet50', type=str, help= "Model used to perform the training. The model name will also be used to identify the csv/plot results for each model.")
    parser.add_argument("--dataset", default = 'dataset_raw', type=str, help='Name of the dataset to use. Must be the exact same name as the dataset directory name')
    parser.add_argument("--dataset_source", default = 'dataset_raw', type=str, help='Name of the source dataset that model are trained on. Flag for test inference.')
    parser.add_argument("--save_model_path", default =  os.getcwd() + '/trained_models/', type=str, help="Path where you wish to store the trained models")
    parser.add_argument("--save_results", default = True, type=bool, help="Save results performance in csv or not.")
    parser.add_argument("--csv_dataset_path", default = os.getcwd() + '/split_kfold/', type=str, help="Path where are located the image paths for each partition")
    parser.add_argument("--results_path", default = os.getcwd() + '/results_files/', type=str, help="Path where are located the performance of the models in csv file")
    parser.add_argument("--plot_path", default = os.getcwd() + '/plots/', type=str, help="Path where are located the plot graphs for the loss and accuracy performance")
    parser.add_argument("--name", default='ResNet50', type=str, help='Name of the experiment')
    parser.add_argument("-ts","--type_split",default="kfold",nargs="?", choices=["cross", "kfold"], help="Diferent kind of split to train the models.")
    parser.add_argument("-td","--type_data",default="templates",nargs="?", choices=["templates", "clips", "clips_cropped"], help="Diferent kind of data to train the models.")
    parser.add_argument("--static",default="no",nargs="?", choices=["yes", "no"], help="If 'yes', use static csv. If 'no', use your custom csv partition.")
    parser.add_argument("--pretrained",default="no",nargs="?", choices=["yes", "no"], help="If 'yes', use our trained network. If 'no', use your custom trained network.")
    parser.add_argument("--inf_domain_change",default="no",nargs="?", choices=["yes", "no"], help="If 'yes', it means you are doing inference on another dataset. If 'no', it means you perform inference on the same dataset.")
    parser.add_argument('--faker_data_augmentation', action='store_true', help='Apply data augmentation with generation of new fakes on-the-fly')
    parser.add_argument("--shift_crop", "-scr", type=int, default=10, help= "shifting constant for crop and replace")
    parser.add_argument("--shift_copy", "-sco", type=int, default=10, help= "shifting constant for copy paste")
    return parser