import argparse
from SIDTD.data.DataGenerator.Midv2020 import Template_Generator
from SIDTD.data.DataGenerator.Midv500 import Template_Generator


def Midv2020_generator(path_to_templates:str, sample:int=1000):
    
    generator = Template_Generator(path_to_templates)
    generator.fit(sample)
    generator.store_generated_dataset()
    
    
def Midv500_generator(path_to_templates:str, sample:int=1000):
    
    generator = Template_Generator(path_to_templates)
    generator.fit(sample)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Main for the execution to generate ")
    parser.add_argument("--dataset", "-dt",choices=["Midv2020", "Midv500"], required=True, type=str, help="Mode")    
    parser.add_argument("--src", "-s", default="MIDV2020/dataset/SIDTD",required=True, type=str, help="path of the absolute path of the imatges")
    parser.add_argument("--sample", "-smpl", type=int, default=1000, help= "Number of the iterations to create new imatges")
    
    
    args = parser.parse_args()
    if args.dataset == "Midv2020":
        Midv2020_generator(args.src, args.sample)
    else:
        raise NotImplementedError
    
