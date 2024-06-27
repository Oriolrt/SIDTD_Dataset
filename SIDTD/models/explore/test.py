from SIDTD.models.Baseline.test_kfold_baseline import *
from SIDTD.models.arc_pytorch.test_coAttn import *
from SIDTD.models.transfg.test_kfold_transfg import *

from options.base_options import BaseOptions
from options.baseline_options import BaselineOptions
from options.trans_fg_options import TransFGOptions
from options.coaarn_options import CoAARNOptions


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
            
        # test model on all partition
        if args.type_split in ['kfold', 'unbalanced']:
            for iteration in range(args.nsplits):
                test_baseline_models(args, LOGGER, iteration)
        
        # test model on a specific partition
        else:
            test_baseline_models(args, LOGGER)

    if args.model == 'trans_fg':

        # test model on all partition
        if args.type_split in ['kfold', 'unbalanced']:
            for iteration in range(args.nsplits):
                test_transfg_models(args, LOGGER, iteration)
        
        # test model on a specific partition
        else:
            test_transfg_models(args, LOGGER)

    if args.model == 'coatten_fcn_model':
        print('Hello')
            
        # test model on all partition
        if args.type_split in ['kfold', 'unbalanced']:
            for iteration in range(args.nsplits):
                test_coAttn_models(args, iteration)
        
        # test model on a specific partition
        else:
            test_coAttn_models(args)


if __name__ == "__main__":

    parent_parser = BaseOptions()
    base_args, rem_args = parent_parser.parse_known_args()

    if base_args.model in ['vit_large_patch16_224', 'efficientnet-b3', 'resnet50']:
        args = BaselineOptions(parent_parser)

    elif base_args.model == 'trans_fg':
        args = TransFGOptions(parent_parser)

    elif base_args.model == 'coatten_fcn_model':
        args = CoAARNOptions(parent_parser)
    
    
    #global
    LOG_FILE = '{}_{}.log'.format(args.name, args.dataset) 
    LOGGER = init_logger(LOG_FILE)
        
    main(args)
