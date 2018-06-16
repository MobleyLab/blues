# Only need to change these two variables
PKG_NAME=blues
USER=nathanmlim

OS=$TRAVIS_OS_NAME-64
conda config --set anaconda_upload no
conda build . --output $CONDA_BLD_FILE
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USER -l nightly $CONDA_BLD_FILE --force
