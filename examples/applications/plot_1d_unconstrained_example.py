"""
======================================
Unconstrained (1D)
======================================

How to run a basic 1D unconstrained proximal primal-dual solver. 
We consider the canonical problem :math:`y = x + n` where :math:`n \sim \mathcal{N}`. 
This inverse problem can be solved via the unconstrained optimisation 

.. math::

    \min_x ( ||(x-y)/\sigma||^2_2 + \lambda ||\Psi^{\dagger} x||_1 )

where :math:`x \in \mathbb{R}` is an a priori ground truth 1D signal, :math:`y \in \mathbb{R}` 
are simulated noisy observations, and :math:`\lambda` is the regularisation parameter which acts as 
a Lagrangian multiplier, balancing between data-fidelity and prior information. Before we begin, we 
need to import ``optimusprimal`` and some example specific packages
"""
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm as normal_dist 

import optimusprimal.primal_dual as primal_dual
import optimusprimal.grad_operators as grad_operators
import optimusprimal.linear_operators as linear_operators
import optimusprimal.prox_operators as prox_operators

##############################################################################
# First, we need to define some heuristics for the solver, these include:
#       - tol: convergence criteria for the iterations
#       - iter: maximum number of iterations
#       - update_iter: iterations between logging iteration diagnostics
#       - record_iters: whether to record the full diagnostic information

options = {"tol": 1e-5, "iter": 5000, "update_iter": 50, "record_iters": False}

##############################################################################
# Next, we simulate a standard de-noising setting by contaminating a known
# signal :math:`x`` with some Gaussianly distributed noise. Note that for simplicity the
# measurement operator here is taken to be the identity operator.

size = 2048                                              # Dimension of the 1D vector
ISNR = 20.0                                              # Input signal to noise ratio
sigma = 10 ** (-ISNR / 20.0)                             # Noise standard deviation
reg_param = 4                                            # Regularisation parameter 

x = normal_dist(0, 0.5).pdf(np.linspace(-2, 2, size))    # Ground truth signal x
y = x + np.random.normal(0, sigma, size)                 # Simulated observations y

##############################################################################
# For the unconstrained problem with Gaussian noise the data-fidelity constraint
# is given by the gradient of the :math:`\ell_2`-norm. Here we set up a gradient operator
# corresponding to a gradient of the :math:`\ell_2`-norm.

g = grad_operators.l2_norm(sigma, y, linear_operators.identity())
g.beta = 1. / sigma**2

##############################################################################
# We regularise this inverse problem by adopting a wavelet sparsity :math:`\ell_1`-norm prior.
# To do this we first define what wavelets we wish to use, in this case a
# combination of Daubechies family wavelets, and which levels to consider.
# Any combination of wavelet families available by the pywavelet package may be
# selected -- see https://tinyurl.com/5n7wzpmb

wav = ["db1", "db4", "db6"]                               # Wavelet dictionaries to combine
levels = 6                                                # Wavelet levels to consider [1-6]
shape = (size,)                                           # Shape of nd-wavelets
psi = linear_operators.dictionary(wav, levels, shape)     # Wavelet linear operator

##############################################################################
# Next we construct the :math:`\ell_1`-norm proximal operator which we pass the wavelets
# (:math:`\Psi`) as a dictionary in which to compute the :math:`\ell_1`-norm. We also add an
# additional reality constraint f for good measure, as we know a priori our
# signal :math:`x` is real.

h = prox_operators.l1_norm(np.max(np.abs(psi.dir_op(y))) * reg_param, psi)
h.beta = 1.
f = prox_operators.real_prox()

##############################################################################
# Finally we run the optimisation...

best_estimate, diagnostics = primal_dual.FBPD(y, options, g, f, h)

##############################################################################
# ...and plot the results!

plt.plot(np.real(y), "o", markersize=1)
plt.plot(np.real(x), linewidth=2)
plt.plot(np.real(best_estimate), linewidth=2)
plt.legend(["data", "true", "fit"])

def eval_snr(x, x_est):
    return round((
        np.log10(
            np.sqrt(np.sum(np.abs(x) ** 2))
            / np.sqrt(np.sum(np.abs(x - x_est) ** 2))
        )
        * 20.0
    ), 2)

SNR_est = eval_snr(x, best_estimate)
SNR_data = eval_snr(x, y)

plt.title("Data SNR: {}dB, Reconstruction SNR: {}dB".format(SNR_data, SNR_est), fontsize=16)
plt.show()
