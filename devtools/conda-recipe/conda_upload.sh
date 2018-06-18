#!/bin/bash

CONDA_BLD_FILE=${HOME}/miniconda/conda-bld/noarch/blues-*.tar.bz2

#echo "Converting conda package..."
#conda convert --platform all ${CONDA_BLD_FILE} --output-dir conda-bld/

echo "Deploying $CONDA_BLD_FILE to Anaconda.org..."
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USERNAME -l dev ${CONDA_BLD_FILE} --force

echo "Successfully deployed to Anaconda.org."
exit 0
