#!/bin/bash
set -e

echo "Converting conda package..."
conda convert --platform all $CONDA_BLD_PATH/linux-64/$PKG_NAME-*.tar.bz2 --output-dir conda-bld/

echo "Deploying to Anaconda.org..."
anaconda -t $CONDA_UPLOAD_TOKEN $CONDA_BLD_PATH/**/$PKG_NAME-*.tar.bz2

echo "Successfully deployed to Anaconda.org."
exit 0

#export CONDA_BLD_FILE="$CONDA_BLD_PATH/$OS/$PKG_NAME-$VERSION-py${CONDA_PY}_${PKG_BUILDNUM}.tar.bz2"
#conda build .
#anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER -l dev $CONDA_BLD_FILE --force
