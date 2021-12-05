.. image:: https://img.shields.io/badge/GitHub-optimusprimal-brightgreen.svg?style=flat
    :target: https://github.com/astro-informatics/Optimus-Primal
.. image:: https://github.com/astro-informatics/Optimus-Primal/actions/workflows/python.yml/badge.svg
    :target: https://github.com/astro-informatics/Optimus-Primal/actions/workflows/python.yml
.. image:: https://codecov.io/gh/astro-informatics/Optimus-Primal/branch/master/graph/badge.svg?token=AJIQGKU8D2
    :target: https://codecov.io/gh/astro-informatics/Optimus-Primal
.. image:: https://badge.fury.io/py/optimusprimal.svg
    :target: https://badge.fury.io/py/optimusprimal
.. image:: https://img.shields.io/badge/License-GPL-blue.svg
    :target: http://perso.crans.org/besson/LICENSE.html
.. image:: http://img.shields.io/badge/arXiv-XXXX.XXXX-orange.svg?style=flat
    :target: https://arxiv.org/abs/XXXX.XXXX

.. image:: /docs/assets/logo2.png
    :width: 450
    :align: center

``optimusprimal`` is a light weight proximal splitting Forward Backward Primal Dual based solver for convex optimization problems. 
The current version supports finding the minimum of f(x) + h(A x) + p(B x) + g(x), where f, h, and p are lower semi continuous and have proximal operators, and g is differentiable. A and B are linear operators.
To learn more about proximal operators and algorithms, visit `proximity operator repository <http://proximity-operator.net/index.html>`_. We suggest that users read the tutorial `"The Proximity Operator Repository. User's guide" <http://proximity-operator.net/download/guide.pdf>`_.

QUICK INSTALL
==============================================
You can install ``optimusprimal`` from PyPi by running

.. code-block:: bash

    pip install optimusprimal

INSTALL FROM SOURCE
==============================================
Alternatively, you can install ``optimusprimal`` from GitHub by first cloning the repository 

.. code-block:: bash

    git clone git@github.com:astro-informatics/Optimus-Primal.git
    cd Optimus-Primal

and running the build script 

.. code-block:: bash 

    bash build_optimusprimal.sh 

Following which unit tests can be run 

.. code-block:: bash

    pytest --black optimusprimal/tests/


CONTRIBUTORS
==============================================
`Luke Pratley <https://www.lukepratley.com>`_, `Matthjis Mars <https://www.linkedin.com/in/matthijs-mars/>`_, `Matthew Price <https://scholar.google.com/citations?user=w7_VDLQAAAAJ&hl=en&authuser=1>`_.

LICENSE
==============================================

``optimusprimal`` is released under the GPL-3 license (see `LICENSE.txt <https://github.com/astro-informatics/Optimus-Primal/blob/master/LICENSE>`_), subject to 
the non-commercial use condition.

.. code-block::

     optimusprimal
     Copyright (C) 2021 Luke Pratley & contributors

     This program is released under the GPL-3 license (see LICENSE.txt), 
     subject to a non-commercial use condition (see LICENSE_EXT.txt).

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.