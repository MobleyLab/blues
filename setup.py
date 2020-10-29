"""
This setup.py script and other related installation scripts are adapted from
https://github.com/choderalab/yank/blob/master/setup.py
"""

from setuptools import setup, find_packages
from os.path import relpath, join
import os, sys
import versioneer

try:
    import pypandoc
    long_description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
    long_description = "Error reading README"

#from Cython.Build import cythonize
DOCLINES = __doc__.split("\n")

CLASSIFIERS = """\
Development Status :: 1 - Alpha
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved :: The MIT License (MIT)
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Scientific/Engineering :: Chemistry
Operating System :: Unix
"""


def find_package_data(data_root, package_root):
    files = []
    for root, dirnames, filenames in os.walk(data_root):
        for fn in filenames:
            files.append(relpath(join(root, fn), package_root))
    return files

################################################################################
# SETUP
################################################################################

setup(
    name='blues',
    author="Samuel C. Gill, Nathan M. Lim, Kalistyn Burley, David L. Mobley, and others",
    author_email='dmobley@uci.edu',
    description=DOCLINES[0],
    long_description=long_description,
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='MIT',
    python_requires=">=3.6",
    url='https://github.com/MobleyLab/blues',
    platforms=['Linux-64', 'Mac OSX-64', 'Unix-64'],
    classifiers=CLASSIFIERS.splitlines(),
    packages=['blues', "blues.tests", "blues.tests.data"] +
    ['blues.{}'.format(package) for package in find_packages('blues')],
    package_data={
        'blues':
        find_package_data('blues/tests/data', 'blues') + ['notebooks/*.ipynb'] + ['images/*'] + ['examples/*.yml']
    },
    package_dir={'blues': 'blues'},
    extras_require={
        'docs': [
            'sphinx',  # autodoc was broken in 1.3.1
            'sphinxcontrib-napoleon',
            'sphinx_rtd_theme',
            'numpydoc',
            'nbsphinx'
        ],
        'tests': [
            'pytest',
            'pytest-cov',
            'pytest-pep8',
            'tox',
        ],
    },
    tests_require=[
        'pytest',
        'pytest-cov',
        'pytest-pep8',
        'tox',
    ],
    install_requires=[
        'openmmtools==0.15.0',
        'oeommtools>=0.1.16',
        'openmm>=7.2.2',
        'numpy>=1.15.2',
        'cython>=0.28.5'
    ],
    zip_safe=False,
    include_package_data=True)
