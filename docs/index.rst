|GitHub| |Build Status| |CodeCov| |PyPI| |GPL license|


.. |GitHub| image:: https://img.shields.io/badge/GitHub-optimusprimal-brightgreen.svg?style=flat
    :target: https://github.com/astro-informatics/Optimus-Primal
.. |Build Status| image:: https://github.com/astro-informatics/Optimus-Primal/actions/workflows/python.yml/badge.svg
    :target: https://github.com/astro-informatics/Optimus-Primal/actions/workflows/python.yml
.. |CodeCov| image:: https://codecov.io/gh/astro-informatics/Optimus-Primal/branch/master/graph/badge.svg?token=AJIQGKU8D2
    :target: https://codecov.io/gh/astro-informatics/Optimus-Primal
.. |PyPi| image:: https://badge.fury.io/py/optimusprimal.svg
    :target: https://badge.fury.io/py/optimusprimal
.. |GPL license| image:: https://img.shields.io/badge/License-GPL-blue.svg
    :target: http://perso.crans.org/besson/LICENSE.html
.. |ArXiV| image:: http://img.shields.io/badge/arXiv-XXXX.XXXX-orange.svg?style=flat
    :target: https://arxiv.org/abs/XXXX.XXXX

Optimus-Primal: Lightweight primal-dual solver
======================================================

``optimusprimal`` is a light weight proximal splitting Forward Backward Primal Dual based solver for convex optimization problems. 
The current version supports finding the minimum of combinations of at most four functions simultaneously denoted throughout as 
:math:`f(x), h(A x), p(B x)` and :math:`g(C x)` where :math:`f, h`, and :math:`p` are lower semi continuous and have proximal operators, 
and :math:`g` is differentiable. Note that here :math:`A`, :math:`B`, and :math:`C` are linear operators. 
Combinations of these functions can result in optimisations in both the constrained setting, *i.e.*

.. math:: x^{\star} = \min{h(A x)}_x \quad s.t. \quad p(B x) \leq \epsilon

where :math:`\epsilon` is typically an iso-contour of the log-likelihood ball, and the unconstrained setting *e.g.*

.. math:: x^{\text{map}} = \text{argmin}_x \big [ g(C x)) + \lambda h(A x) \big ]

where map denotes the *maximum a posteriori* solution, and :math:`\lambda` is a Lagrangian multiplier balancing data-fidelity against 
*a priori* assumed knowledge. To learn more about proximal operators and algorithms, visit `proximity operator repository <http://proximity-operator.net/index.html>`_. We suggest that users read the tutorial 
`"The Proximity Operator Repository. User's guide" <http://proximity-operator.net/download/guide.pdf>`_.

How to use this guide
---------------------
To get started, follow the :ref:`installation guide <Installation>`.  For a brief background of the implementation of the proximal primal-dual algorithm  please see the :ref:`background <Background>` 
section of this guide, which provides sufficient background information. We have also provided a variety of pedagogical examples, including a number of interactive notebooks 
that provide a step-by-step guide to get ``optimusprimal`` up and running for your particular application.  An up-to-date catalog of the software functionality can be found on the :ref:`API <Namespaces>` page. 

Basic Usage
------------
First you will need to install ``optimusprimal`` PyPi by running

.. code-block:: bash

    pip install optimusprimal

Following this you can, for example, perform an constrained proximal primal dual reconstruction by

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


Contributors 
--------------------
`Luke Pratley <https://www.lukepratley.com>`_, `Matthjis Mars <https://www.linkedin.com/in/matthijs-mars/>`_, `Matthew Price <https://scholar.google.com/citations?user=w7_VDLQAAAAJ&hl=en&authuser=1>`_.


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   user_guide/install


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Background

   background/Proximal-Primal-Dual/index

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Tutorials
   
   auto_examples/index


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: API

   api/index



