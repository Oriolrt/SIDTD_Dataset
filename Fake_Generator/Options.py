import argparse
import json
import os
import sys

__author__ = "Carlos Boned Riera"
__email__ = "cboned@cvc.uab.cat"

from argparse import HelpFormatter
from operator import attrgetter


class SortingHelpFromatter(HelpFormatter):
    def add_arguments(self, actions):
        actions = sorted(actions, key=attrgetter("option_strings"))
        super(SortingHelpFromatter, self).add_arguments(actions)




class Midv500(argparse.ArgumentParser):

    def __init__(self):

        super().__init__(
            description="This script Create the database that we will use to train the IA for fake NID with the Midv500 dataset ",
            formatter_class=SortingHelpFromatter
        )

        super().add_argument("-n", "--sampling_number", type=int, default=None,required=False ,help="The number of fake elements to generate, if this extension is up the proportion of the different type of data augmentation are required")
        super().add_argument("-i", "--info", type=str, required=True, help= "Path with the json file with the template of every fake image info")
        super().add_argument("-t", "--type", type=str,default="image", const=None,help="Define to create fake imgs or fake videos")
        super().add_argument("-p", "--dataset_path", nargs="?" ,type=str,required=True, const=os.getcwd(),help="absolute path with the root directory of the datasets")
        super().add_argument("-d", "--delta_boundary", type=int, default=10,required=False, help= "shifting constant for crop and replace")

    def parse(self):
        return super().parse_args()



class Midv2020(argparse.ArgumentParser):

    def __init__(self):

        super().__init__(
            description="This script Create the database that we will use to train the IA for fake NID with the Midv2020 dataset ",
            formatter_class=SortingHelpFromatter
        )

        super().add_argument("-n", "--sampling_number", type=int, default=None,required=False ,help="The number of fake elements to generate, if this extension is up the proportion of the different type of data augmentation are required")
        super().add_argument("-i", "--info", type=str, required=True, help= "Path with the json file with the template of every fake image info")
        super().add_argument("-t", "--type", type=str,default="image", const=None,help="Define to create fake imgs or fake videos")
        super().add_argument("-p", "--dataset_path", type=str,required=True,nargs="?" ,const=os.getcwd(),help="absolute path with the root directory of the datasets")
        super().add_argument("-d", "--delta_boundary", type=int, default=10,required=False, help= "shifting constant for crop and replace")


    def parse(self):
        return super().parse_args()
