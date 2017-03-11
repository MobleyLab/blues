import os
from os.path import relpath, join
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

def find_package_data(data_root, package_root):
    files = []
    for root, dirnames, filenames in os.walk(data_root):
        for fn in filenames:
            files.append(relpath(join(root, fn), package_root))
    return files

setup(
    name = "blues",
    version = "0.0.1",
    author = "Samuel Gill, David Mobley, and others",
    description = ("NCMC moves in OpenMM to enhance ligand sampling"),
    license = "MIT",
    platforms = ['Linux-64', 'Mac OSX-64', 'Unix-64'],
    keywords = "Nonequilibrium candidate Monte Carlo sampling of ligand binding modes",
    url = "https://github.com/MobleyLab/blues",
    packages=find_packages()+['blues', 'blues/tests'],
    include_package_data = True,
    zip_safe = False,
    long_description=read('README.md'),
    classifiers=[
        "Development Status :: 1 - Alpha",
        "Topic :: Utilities",
    ],
)
