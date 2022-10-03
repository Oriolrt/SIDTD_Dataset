import argparse
import os
from models.Baseline.training_kfold_baseline import *
from models.Baseline.test_kfold_baseline import *
from models.arc_pytorch.train_coAttn import *
from models.arc_pytorch.test_coAttn import *
from models.transfg.train_kfold_transfg import *
from models.transfg.test_kfold_transfg import *


def init_logger(log_file='train.log'):
    from logging import getLogger, DEBUG, FileHandler, Formatter, StreamHandler
        
    log_format = '%(asctime)s %(levelname)s %(message)s'
        
    stream_handler = StreamHandler()
    stream_handler.setLevel(DEBUG)
    stream_handler.setFormatter(Formatter(log_format))
        
    file_handler = FileHandler(log_file)
    file_handler.setFormatter(Formatter(log_format))
        
    logger = getLogger('Herbarium')
    logger.setLevel(DEBUG)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
        
    return logger 

def main(args):

    if args.model in ['vit_large_patch16_224', 'efficientnet-b3', 'resnet50']:
        
        # Choose or not to train model 
        if args.training == 'yes':
            
            # train model on all partition
            if args.all =='yes':
                for iteration in range(args.nsplits):
                    train_baseline_models(args, LOGGER, iteration)
            
            # train model on a specific partition
            else:
                iteration = args.partition
                train_baseline_models(args, LOGGER, iteration)
        
        # Choose or not to test model 
        if args.test == 'yes':
            
            # test model on all partition
            if args.all =='yes':
                for iteration in range(args.nsplits):
                    test_baseline_models(args, LOGGER, iteration)
            
            # test model on a specific partition
            else:
                iteration = args.partition
                test_baseline_models(args, LOGGER, iteration)

    if args.model == 'trans_fg':
        
        # Choose or not to train model 
        if args.training == 'yes':
            
            # train model on all partition
            if args.all =='yes':
                for iteration in range(args.nsplits):
                    train_transfg_models(args, LOGGER, iteration)
            
            # train model on a specific partition
            else:
                iteration = args.partition
                train_transfg_models(args, LOGGER, iteration)
        
        # Choose or not to test model 
        if args.test == 'yes':
            
            # test model on all partition
            if args.all =='yes':
                for iteration in range(args.nsplits):
                    test_transfg_models(args, LOGGER, iteration)
            
            # test model on a specific partition
            else:
                iteration = args.partition
                test_transfg_models(args, LOGGER, iteration)

    if args.model == 'coatten_fcn_model':
        
        # Choose or not to train model 
        if args.training == 'yes':
            
            # train model on all partition
            if args.all =='yes':
                for iteration in range(args.nsplits):
                    train_coAttn_models(args, iteration)
            
            # train model on a specific partition
            else:
                iteration = args.partition
                train_coAttn_models(args, iteration)
        
        # Choose or not to test model 
        if args.test == 'yes':
            
            # test model on all partition
            if args.all =='yes':
                for iteration in range(args.nsplits):
                    test_coAttn_models(args, LOGGER, iteration)
            
            # test model on a specific partition
            else:
                iteration = args.partition
                test_coAttn_models(args, LOGGER, iteration)



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    # flag for defining training or testing parameters
    parser.add_argument("--partition", default = 0, type=int, help="train or test on a specific partition")
    parser.add_argument("--all", choices = ['yes', 'no'], default = 'yes', type=int, help="train or test on all partitions")
    parser.add_argument("--nsplits", default = 10, type=int, help="Number of k-fold partition")
    parser.add_argument("--nclasses", default = 2, type=int, help="Number of class in the dataset")
    parser.add_argument("--model", choices = ['vit_large_patch16_224', 'efficientnet-b3', 'resnet50', 'trans_fg', 'coatten_fcn_model'], default = 'resnet50', type=str, help= "Model used to perform the training. The model name will also be used to identify the csv/plot results for each model.")
    parser.add_argument("--training", choices = ['yes', 'no'], default = 'yes', type=str, help= "Choose if the user want to train chosen model on the chosen dataset.")
    parser.add_argument("--test", choices = ['yes', 'no'], default = 'yes', type=str, help= "Choose if the user want to test chosen model on the chosen dataset.")
    parser.add_argument("--save_model_path", default =  os.getcwd() + '/trained_models/', type=str, help="Path where you wish to store the trained models")
    parser.add_argument("--save_results", default = True, type=bool, help="Save results performance in csv or not.")

    # flag for baseline code
    parser.add_argument("--name", default='ResNet50', type=str, help='Name of the experiment')
    parser.add_argument("--dataset", default = 'dataset_raw', type=str, help='Name of the dataset to use. Must be the exact same name as the dataset directory name')
    parser.add_argument("--device", default = 'cuda', type=str, help='Use CPU or CUDA')
    parser.add_argument("--batch_size", default = 32, type=int)
    parser.add_argument("--accumulation_steps", default = 2, type=int)
    parser.add_argument("--epochs", default = 100, type=int)
    parser.add_argument("--workers", default = 4, type=int)
    parser.add_argument("--learning_rate", default = 0.01, type=float)

    # flag for Trans FG
   # Required parameters
    
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-L_16", help="Which variant to use.")
    
    parser.add_argument("--pretrained_dir", type=str, 
default= os.getcwd() + "/transfg_pretrained/imagenet21k+imagenet2012_ViT-L_16.npz",
                        help="Where to search for pretrained ViT models.")
    
    parser.add_argument("--img_size", default=299, type=int,
                        help="Resolution size")
    
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    
    parser.add_argument("--eval_batch_size", default=32, type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--learning_rate_sgd", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
                        
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    
    parser.add_argument('--gradient_accumulation_steps', type=int, default=2,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")

    parser.add_argument('--smoothing_value', type=float, default=0.0,
                        help="Label smoothing value\n")
    
    parser.add_argument('--split', type=str, default='non-overlap',
                        help="Split method")
    
    parser.add_argument('--slide_step', type=int, default=12,
                        help="Slide step for overlap split")

    # flag for CoAttention Network
    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to ARC')
    parser.add_argument("--npy_dataset_path", default = os.getcwd() + '/omniglot/', type=str, help="Path where are located the image arrays for each label and partition")
    parser.add_argument("--csv_dataset_path", default = os.getcwd() + '/split_kfold/', type=str, help="Path where are located the image paths for each partition")
    parser.add_argument('--glimpseSize', type=int, default=8, help='the height / width of glimpse seen by ARC')
    parser.add_argument('--numStates', type=int, default=128, help='number of hidden states in ARC controller')
    parser.add_argument('--numGlimpses', type=int, default=6, help='the number glimpses of each image in pair seen by ARC')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--cuda', default='True', help='enables cuda')
    parser.add_argument('--n_its', type=int, default=5000, help='number of iterations for training')
    parser.add_argument('--apply_fcn', type=bool, default=True, help='apply a resnet to input prior to ARC')
    parser.add_argument('--use_coAttn', type=bool, default=True, help='apply the coattention mechanism before discrimination')
    args = parser.parse_args()
    
    #global
    LOG_FILE = '{}_{}.log'.format(args.name, args.dataset) 
    LOGGER = init_logger(LOG_FILE)
        
    main(args)
