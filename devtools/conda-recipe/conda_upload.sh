# Only need to change these two variables
PKG_NAME=blues
USER=nathanmlim

OS=$TRAVIS_OS_NAME-64
conda config --set anaconda_upload no
export CONDA_BLD_FILE="$CONDA_BLD_PATH/$OS/$PKG_NAME-$VERSION.tar.bz2"
conda build . --output $CONDA_BLD_FILE
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER -l nightly $CONDA_BLD_FILE --force
