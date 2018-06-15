# Only need to change these two variables
PKG_NAME=blues
USER=nathanmlim

OS=$TRAVIS_OS_NAME-64
mkdir ~/conda-bld
conda config --set anaconda_upload no
conda build .
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER -l nightly $CONDA_BLD_PATH/$OS/$PKG_NAME-$VERSION-`date +%Y.%m.%d`.tar.bz2 --force
