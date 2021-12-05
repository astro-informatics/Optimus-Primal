.. _install:

Installation
============
We recommend installing ``optimusprimal`` through `PyPi <https://pypi.org>`_, however in some cases one may wish to install ``optimusprimal`` directly from source, which is also relatively straightforward. 

Quick install (PyPi)
--------------------
Install ``optimusprimal`` from PyPi with a single command

.. code-block:: bash

    pip install optimusprimal 

Check that the package has installed by running 

.. code-block:: bash 

	pip list 

and locate optimusprimal.


Install from source (GitHub)
----------------------------

When installing from source we recommend working within an existing conda environment, or creating a fresh conda environment to avoid any dependency conflicts,

.. code-block:: bash

    conda create -n optimusprimal_env python=3.8.0
    conda activate optimusprimal_env

Once within a fresh environment ``optimusprimal`` may be installed by cloning the GitHub repository

.. code-block:: bash

    git clone https://github.com/astro-informatics/Optimus-Primal
    cd Optimus-Primal

and running the install script, within the root directory, with one command 

.. code-block:: bash

    bash build_optimusprimal.sh

To check the install has worked correctly run the unit tests with 

.. code-block:: bash

	pytest --black optimusprimal/tests/

.. note:: For installing from source a conda environment is required by the installation bash script, which is recommended, due to a pandoc dependency.
