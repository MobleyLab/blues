#!/bin/bash
set -e

#export CONDA_BLD_FILE=${HOME}/conda-bld/noarch/blues-0.2.3-py35_0.tar.bz2
CONDA_BLD_FILE=${HOME}/conda-bld/noarch/blues-**_0.tar.bz2

#echo "Converting conda package..."
#conda convert --platform all ${CONDA_BLD_FILE} --output-dir conda-bld/

echo "Deploying $CONDA_BLD_FILE to Anaconda.org..."
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USERNAME -l dev ${CONDA_BLD_FILE} --force

echo "Successfully deployed to Anaconda.org."
exit 0
