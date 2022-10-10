# To generate the fake templates Dataset with the Midv2020 folder
from email import generator
from importlib.resources import path
from Fake_Generator.Fake_Loader.Midv2020 import Template_Generator


def Midv2020_generator(path_to_templates:str, sample:int=1000):
    
    generator = Template_Generator(path_to_templates)
    generator.fit(sample)
    generator.store_generated_dataset()
    
    
def Midv500_generator(path_to_templates:str, sample:int=1000):
    
    generator = Template_Generator(path_to_templates)
    generator.fit(sample)