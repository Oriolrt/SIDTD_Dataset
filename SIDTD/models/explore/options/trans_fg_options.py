import os
import argparse

def TransFGOptions(parser):
    """This class includes options for Trans Fg model.

    It also includes shared options defined in BaseOptions.
    """

    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14"],
                        default="ViT-L_16", help="Which variant to use.")
    
    complete_path = os.path.join(os.getcwd(), '..', 'transfg', 'transfg_pretrained', 'imagenet21k+imagenet2012_ViT-L_16.npz')
    parser.add_argument("--pretrained_dir", type=str, default= complete_path,
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
    
    args = parser.parse_args()

    return args