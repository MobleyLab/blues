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

def write_meta_yaml(filename='devtools/conda-recipe/meta.yaml'):

    #with open('blues/version.py') as f:
    #    data = f.read()
    #lines = data.split('\n')

    version = versioneer.get_version()
    (short_version, build_number) = version.split('+')
    version_numbers = {'short_version': short_version,
                       'build_number' : build_number }

    with open(filename, 'r') as meta:
        yaml_lines = meta.readlines()

    a = open(filename, 'w')
    try:
        for k, v in version_numbers.items():
            a.write("{{% set {} = '{}' %}}\n".format(k, v))
        #Replace top 2 header lines that contain the package version
        a.writelines(yaml_lines[2:])
    finally:
        a.close()


################################################################################
# USEFUL SUBROUTINES
################################################################################
#def read(fname):
#    return open(os.path.join(os.path.dirname(__file__), fname)).read()


def find_package_data(data_root, package_root):
    files = []
    for root, dirnames, filenames in os.walk(data_root):
        for fn in filenames:
            files.append(relpath(join(root, fn), package_root))
    return files


def check_dependencies():
    from distutils.version import StrictVersion
    found_openmm = True
    found_openmmtools = True
    found_openmm_720_or_earlier = True
    found_numpy = True

    try:
        from simtk import openmm
        openmm_version = StrictVersion(openmm.Platform.getOpenMMVersion())
        if openmm_version < StrictVersion('7.2.0'):
            found_openmm_720_or_earlier = False
    except ImportError as err:
        found_openmm = False

    try:
        import numpy
    except:
        found_numpy = False

    try:
        import openmmtools
    except:
        found_openmmtools = False

    msg = None
    bar = ('-' * 70) + "\n" + ('-' * 70)
    if found_openmm:
        if not found_openmm_720_or_earlier:
            msg = [
                bar,
                '[Unmet Dependency] BLUES requires OpenMM version > 7.2.0. You have version %s.'
                % openmm_version, bar
            ]
    else:
        msg = [
            bar,
            '[Unmet Dependency] BLUES requires the OpenMM python package. Please install with `conda install -c omnia openmm=7.2.2` ',
            bar
        ]

    if not found_numpy:
        msg = [
            bar,
            '[Unmet Dependency] BLUES requires the numpy python package. Refer to <http://www.scipy.org/scipylib/download.html> for numpy installation instructions.',
            bar
        ]

    if not found_openmmtools:
        msg = [
            bar,
            '[Unmet Dependency] BLUES requires the openmmtools python package. Please install with `conda install -c omnia openmmtools=0.15.0`',
            bar
        ]

    if msg is not None:
        import textwrap
        print()
        print(
            os.linesep.join([line for e in msg for line in textwrap.wrap(e)]),
            file=sys.stderr)
        #print('\n'.join(list(textwrap.wrap(e) for e in msg)))


################################################################################
# SETUP
################################################################################

write_meta_yaml('devtools/conda-recipe/meta.yaml')
setup(
    name='blues',
    author=
    "Samuel C. Gill, Nathan M. Lim, Kalistyn Burley, David L. Mobley, and others",
    author_email='dmobley@uci.edu',
    description=DOCLINES[0],
    long_description=long_description,
    #long_description="\n".join(DOCLINES[2:]),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='MIT',

    python_requires= ">=3.5",
    url='https://github.com/MobleyLab/blues',
    platforms=['Linux-64', 'Mac OSX-64', 'Unix-64'],
    classifiers=CLASSIFIERS.splitlines(),
    packages=['blues', "blues.tests", "blues.tests.data"] + ['blues.{}'.format(package) for package in find_packages('blues')],
    package_data={'blues': find_package_data('blues/tests/data', 'blues') + ['notebooks/*.ipynb'] + ['images/*'] + ['examples/*.yml'] },
    package_dir={'blues': 'blues'},
    #install_requires=[
    #    'numpy', 'cython', 'scipy', 'pandas',
    #    'netCDF4', 'pyyaml', 'pytest',
    #],

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
    zip_safe=False,
    include_package_data=True)
#check_dependencies()
