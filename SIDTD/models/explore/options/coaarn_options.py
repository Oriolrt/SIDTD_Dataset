import argparse

def CoAARNOptions(parser):
    """This function includes options for CoAARN model.

    It also includes shared options defined in BaseOptions.
    """

    parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=224, help='the height / width of the input image to ARC')
    parser.add_argument('--glimpseSize', type=int, default=8, help='the height / width of glimpse seen by ARC')
    parser.add_argument('--numStates', type=int, default=128, help='number of hidden states in ARC controller')
    parser.add_argument('--numGlimpses', type=int, default=6, help='the number glimpses of each image in pair seen by ARC')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--n_its', type=int, default=5000, help='number of iterations for training')
    parser.add_argument('--cuda', default='True', help='enables cuda')
    
    args = parser.parse_args()

    return args