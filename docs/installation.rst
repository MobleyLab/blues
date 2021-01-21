Installation
==================

BLUES is compatible with MacOSX/Linux with Python=3.6.

This is a python tool kit with a few dependencies. We recommend installing
`miniconda <http://conda.pydata.org/miniconda.html>`_. Then you can create an
environment with the following commands:

.. code-block:: bash

   conda create -n blues python=3.6
   conda activate blues

Stable Releases
---------------
The recommended way to install BLUES would be to install from conda.


.. code-block:: bash

    # Create a clean environment (python 3.6 is required)
    conda create -n blues python=3.6
    conda activate blues

    # Install OpenEye toolkits and related tools first
    conda install -c openeye/label/Orion -c omnia oeommtools
    conda install -c openeye openeye-toolkits

    # Install some dependencies
    conda install -c omnia -c conda-forge openmmtools=0.15.0 openmm=7.4.2 numpy cython

    # Then install BLUES
    conda install -c mobleylab blues    conda install -c mobleylab blues


Source Installation
-------------------
Although we do NOT recommend it, you can also install directly from the
source code.

.. code-block:: bash

    git clone https://github.com/MobleyLab/blues.git ./blues
    conda install -c omnia -c conda-forge openmmtools=0.15.0 openmm=7.4.2 numpy cython
    conda install -c openeye/label/Orion -c omnia oeommtools
    conda install -c openeye openeye-toolkits
    pip install -e .

To validate your BLUES installation run the tests.

.. code-block:: bash

    cd blues/tests
    pytest -v -s
