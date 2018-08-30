#!/bin/bash

# Parse Github Tag into BASH array
IFS="- " read -r -a GIT <<< `git describe --tags`

# Check for release flag
if ${RELEASE} ; then
    echo 'Deploying RELEASE build' && LABEL='main'
else
    echo 'Deploying DEV build' && LABEL='dev'
fi

# Export the conda environment
CONDA_ENV_FILE=${PKG_NAME}-${GIT[0]}_py${CONDA_PY}-${TRAVIS_OS_NAME}.yml
conda env export -n ${CONDA_ENV} -f $CONDA_ENV_FILE --no-builds

# Upload the conda environment
ENV_URL="https://anaconda.org/${USERNAME}/${CONDA_ENV}/files"
echo "Uploading conda environment $CONDA_ENV_FILE to $ENV_URL..."
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USERNAME $CONDA_ENV_FILE -l ${LABEL} --force

# Get the conda package
#CONDA_BLD_FILE=`conda build --python=${TRAVIS_PYTHON_VERSION} devtools/conda-recipe --output`

PKG_URL="https://anaconda.org/${USERNAME}/${PKG_NAME}/files"
echo "Deploying $CONDA_BLD_FILE to ${PKG_URL}..."
anaconda -t $CONDA_UPLOAD_TOKEN upload -u $USERNAME -l ${LABEL} ${CONDA_BLD_FILE} --force
