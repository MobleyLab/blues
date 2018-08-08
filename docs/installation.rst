Installing chemper
==================

The only way to install ``chemper`` right now is to clone or download
the `GitHub repo <https://github.com/Mobleylab/chemper/>`_.
When we're ready to release version 1.0.0 we'll also make it conda installable.

Prerequisites
-------------

We currently test with Python 3.5, though we expect anything 3.5+ should work.

This is a python tool kit with a few dependencies. We recommend installing
`miniconda <http://conda.pydata.org/miniconda.html>`_. Then you can create an
environment with the following commands:

.. code-block:: bash

    conda create -n [my env name] python=3.5 numpy networkx pytest
    source activate [my env name]

This command will install all dependencies besides a toolkit for cheminformatics or storing of molecule
information. We seek to keep this tool independent of cheminformatics toolkit, but currently only support
`RDKit <http://www.rdkit.org/docs/index.html>`_ and `OpenEye Toolkits <https://www.eyesopen.com/>`_.
If you wish to add support please feel free to submit a pull request.
Make sure one of these tool kits is installed in your environment before installing chemper.

RDKit environment
^^^^^^^^^^^^^^^^^

Conda installation according to `RDKit documentation <http://www.rdkit.org/docs/Install.html>`_:

.. code-block:: bash

    conda install -c rdkit rdkit

OpenEye environment
^^^^^^^^^^^^^^^^^^^

Conda installation according to `OpenEye documentation <https://docs.eyesopen.com/toolkits/python/quickstart-python/linuxosx.html>`_

.. code-block:: bash

    cond install -c openeye openeye-toolkits

Installation
------------

Hopefully chemper will be conda installable in the near future, but for now the best option
is to download or clone from `GitHub <https://github.com/Mobleylab/chemper/>`_
and then from inside the ``chemper`` directory install with the command:

.. code-block:: bash

    pip install -e .
