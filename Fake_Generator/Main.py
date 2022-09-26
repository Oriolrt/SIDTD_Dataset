from Options import Midv500 as m5
from Options import Midv2020 as m20
from Fake_Loader.Midv500.Template_Generator import *
from Fake_Loader.Midv2020.Template_Generator import Template_Generator as T2
import sys

#from Transformations import Transformations


def read_metadata(args):
    """

    Parameters
    ----------
    args

    Returns
    -------
        De moment aquesta funció hauria de retornar el constant valu amb els que treballarem les transformacions
    """
    return args.delta_boundary


def read_sampling(args):
    """

    Return the differents options of the argparse cleaned and controled

    -------

    """
    default_sampling = 10000

    if args.sampling_number != None:

        return args.sampling_number

    return default_sampling


def create_info(args):

    with open(args.info) as file:
        info = json.load(file)

    return info

def generation_type(args):
    return args.type

def get_path(args):
    return args.dataset_path


def main():
    """

    Returns the data augmentation done
    -------

    """
    opts500 = m5()
    args5 = opts500.parse()
    print(args5)

    opts2020 = m20()
    args20 = opts2020.parse()
    print(args20)



    #MIDV2020
    absolute_path_20 = get_path(args20)
    typ_20 = generation_type(args20)

    if typ_20 == "image":
        fake_meta20 = create_info(args20)
        delta20 = read_metadata(args20)
        sampling20 = read_sampling(args20)

        Midv500Object_Gen = T2(absolute_path_20,sampling20,fake_meta20,delta20)
        #h = Template_Generator(absolute_path,sampling,fake_meta)



if __name__ == '__main__':
    main()
