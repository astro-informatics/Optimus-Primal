import sys
sys.path.insert(0,'..')
import optimusprimal.prox_operators as prox_operators
import optimusprimal.linear_operators as linear_operators
import optimusprimal.grad_operators as grad_operators
import optimusprimal.primal_dual as primal_dual
import numpy as np

def test_l1_constrained():
    options = {'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False}
    ISNR = 20.
    sigma = 10**(-ISNR/20.)
    size = 1024
    epsilon = np.sqrt(size + 2. * np.sqrt(size)) * sigma
    x = np.linspace(0, 1 * np.pi, size)

    W = np.ones((size,))

    y = W * x + np.random.normal(0, sigma, size)

    p = prox_operators.l2_ball(epsilon, y, linear_operators.diag_matrix_operator(W))

    wav = ["db1", "db4"]
    levels = 6
    shape = (size,)
    psi = linear_operators.dictionary(wav, levels, shape)
    
    h = prox_operators.l1_norm(np.max(np.abs(psi.dir_op(y))) * 1e-3, psi)
    h.beta = 1.
    f = prox_operators.real_prox()
    z, diag = primal_dual.FBPD(y, options, f, h, p, None)
    assert(np.linalg.norm(z - W * y) < epsilon * 1.05)
    assert(diag['max_iter'] < 500)

def test_l1_unconstrained():
    options = {'tol': 1e-5, 'iter': 5000, 'update_iter': 50, 'record_iters': False}
    ISNR = 20.
    sigma = 10**(-ISNR/20.)
    size = 1024
    epsilon = np.sqrt(size + 2. * np.sqrt(size)) * sigma
    x = np.linspace(0, 1 * np.pi, size)

    W = np.ones((size,))

    y = W * x + np.random.normal(0, sigma, size)


    g = grad_operators.l2_norm(sigma, y, linear_operators.diag_matrix_operator(W))

    wav = ["db1", "db4"]
    levels = 6
    shape = (size,)
    psi = linear_operators.dictionary(wav, levels, shape)
    
    h = prox_operators.l1_norm(np.max(np.abs(psi.dir_op(y))) * 5e-3, psi)
    h.beta = 1.
    f = prox_operators.real_prox()
    z, diag = primal_dual.FBPD(y, options, f, h, None, g)
    assert(diag['max_iter'] < 500)