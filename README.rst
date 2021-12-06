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

|logo| Optimus-Primal: A Lightweight primal-dual solver
========================================================

.. |logo| raw:: html

   <img src="./docs/assets/animated_logo.png" align="center" height="70" width="70">

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

and running the build script and run install tests by

.. code-block:: bash 

    bash build_optimusprimal.sh 
    pytest --black optimusprimal/tests/

BASIC USAGE
==============================================
After installing ``optimusprimal`` one can *e.g.* perform an constrained proximal primal dual reconstruction by

.. code-block:: python 

    import numpy as np 
    import optimusprimal.primal_dual as primal_dual
    import optimusprimal.linear_operators as linear_ops 
    import optimusprimal.prox_operators as prox_ops 

    options = {'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False}

    # Load some data
    y = np.load('Some observed signal y')                                 # Load a file of observed data.
    epsilon = sigma * np.sqrt(y.size + 2 np.sqrt(y.size))                 # where sigma is your noise std.

    # Define a forward model i.e. y = M(x) + n
    M = np.ones_like(y)                                                   # Here M = Identity for simplicity.
    p = prox_ops.l2_ball(epsilon, y, linear_ops.diag_matrix_operator(M))  # Create a l2-ball data-fidelity.

    # Define a regularisation i.e. ||W(x)||_1
    wav = ['db1', 'db3', 'db4']                                           # Select some wavelet dictionaries.
    psi = linear_operators.dictionary(wav, levels=6, y.shape)             # Define multi-dictionary wavelets.
    h = prox_ops.l1_norm(gamma=1, psi)                                    # Create an l1-norm regulariser.

    # Recover an estiamte i.e. x_est = min[h(x)] s.t. p(x) <= epsilon
    x_est, = primal_dual.FBPD(y, options, None, None, h, p, None)         # Recover an estimate of x.

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