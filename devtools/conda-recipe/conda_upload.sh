#!/bin/bash
set -e
export CONDA_BLD_FILE="$CONDA_BLD_PATH/noarch/$PKG_NAME-0.2.3-py${CONDA_PY}_${PKG_BUILDNUM}.tar.bz2"
echo "Converting conda package..."
conda convert --platform all ${CONDA_BLD_FILE} --output-dir conda-bld/

echo "Deploying to Anaconda.org..."
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER -l dev ${CONDA_BLD_FILE} --force

echo "Successfully deployed to Anaconda.org."
exit 0

#export CONDA_BLD_FILE="$CONDA_BLD_PATH/$OS/$PKG_NAME-$VERSION-py${CONDA_PY}_${PKG_BUILDNUM}.tar.bz2"
#conda build .
#anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER -l dev $CONDA_BLD_FILE --force
