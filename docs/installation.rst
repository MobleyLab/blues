Installation
==================

BLUES is compatible with MacOSX/Linux with Python>=3.6 (blues<1.1 still works with Python 2.7)

This is a python tool kit with a few dependencies. We recommend installing
`miniconda <http://conda.pydata.org/miniconda.html>`_. Then you can create an
environment with the following commands:

.. code-block:: bash

    conda create -n blues python=3.6
    source activate blues

Stable Releases
---------------
The recommended way to install BLUES would be to install from conda.

.. code-block:: bash

    conda install -c mobleylab blues


Development Builds
------------------
Alternatively, you can install the latest development build. Development builds
contain the latest commits/PRs not yet issued in a point release.

.. code-block:: bash

    conda install -c mobleylab/label/dev blues


In order to use the `SideChainMove` class you will need OpenEye Toolkits and
some related tools.

.. code-block:: bash

    conda install -c openeye/label/Orion -c omnia oeommtools packmol
    conda install -c openeye openeye-toolkits


Source Installation
-------------------
Although we do NOT recommend it, you can also install directly from the
source code.

.. code-block:: bash

    git clone https://github.com/MobleyLab/blues.git
    conda install -c omnia -c conda-forge openmmtools openmm numpy cython
    pip install -e .

To validate your BLUES installation run the tests.

.. code-block:: bash

   pip instal -e .[tests]
   pytest -v -s
