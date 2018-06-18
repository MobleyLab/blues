#!/bin/bash
#set -e

echo $HOME
echo $TRAVIS_BRANCH
echo $TRAVIS_BUILD_DIR
echo $TRAVIS_PULL_REQUEST_BRANCH


echo "BUILD DIR IS" && ls $TRAVIS_BUILD_DIR
ls $TRAVIS_BUILD_DIR/

ls $HOME
CONDA_BLD_FILE="${HOME}/miniconda/conda-bld/noarch/blues-0.2.3-py35_0.tar.bz2"

#export CONDA_BLD_FILE=${HOME}/conda-bld/noarch/blues-0.2.3-py35_0.tar.bz2
ls $HOME/miniconda/conda-bld/noarch/blues-*.tar.bz2
CONDA_BLD_FILE1=$HOME/miniconda/conda-bld/noarch/blues-*.tar.bz2
echo $CONDA_BLD_FILE1
#echo "Converting conda package..."
#conda convert --platform all ${CONDA_BLD_FILE} --output-dir conda-bld/

echo "Deploying $CONDA_BLD_FILE to Anaconda.org..."
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USERNAME -l dev ${CONDA_BLD_FILE} --force

echo "Successfully deployed to Anaconda.org."
exit 0
