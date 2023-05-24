import argparse

def BaselineOptions(parser):
    """This class includes options for models in Baseline folder (Efficientnet-b3, ResNet50 and ViT).

    It also includes shared options defined in BaseOptions.
    """

    parser.add_argument("--batch_size", default = 32, type=int)
    parser.add_argument("--accumulation_steps", default = 2, type=int)
    parser.add_argument("--workers", default = 4, type=int)
    parser.add_argument("--learning_rate", default = 0.01, type=float)

    #Only for training
    parser.add_argument("--epochs", default = 100, type=int)

    args = parser.parse_args()

    return args



