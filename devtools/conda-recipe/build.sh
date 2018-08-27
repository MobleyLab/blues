#!/bin/bash

# Build the python package, don't let setuptools/pip try to get packages
# $PYTHON setup.py develop --no-deps
#pip install . --no-deps
#$PYTHON setup.py install --single-version-externally-managed --record=record.txt  # Python command to install the script.
$PYTHON -m pip install . --no-deps --ignore-installed --no-cache-dir -vvv
