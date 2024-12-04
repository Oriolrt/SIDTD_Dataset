from setuptools import setup, find_packages
import codecs
import os
#print(find_packages(include=['SIDTD', 'SIDTD.*', '*']))
here = os.path.abspath(os.path.dirname(__file__))

with codecs.open(os.path.join(here, "README.md"), encoding="utf-8") as fh:
    long_description = "\n" + fh.read()

ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # This is your Project Root
VERSION = '0.0.1'
DESCRIPTION = 'Download Dataloader for fake/real benchmarks approach with our own Benchmark'
LONG_DESCRIPTION = 'This package allows to download 5 different benchmarks included our own for the fake/real classification task.'

# Setting up
setup(
    name="SIDTD",
    version=VERSION,
    author="Carlos Boned, Oriol Ramos, Maxime Talarmain",
    author_email="{cboned,oriolrt,mtalarmain}@cvc.uab.cat",
    description=DESCRIPTION,
    long_description_content_type="text/markdown",
    long_description=long_description,
    package_dir={},
    packages=find_packages(exclude='SIDTD/models'),
    install_requires=[line.strip() for line in open("requirements.txt").readlines()],
    keywords=['python', 'benchmarks', "documents","classification", "binary", "fakes", "reals"],
    classifiers=[
        "Development Status :: 1 - Planning",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Operating System :: Unix",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
    ]
)
