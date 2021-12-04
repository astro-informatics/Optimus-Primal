import numpy as np
import sys

sys.path.insert(0, "..")
import optimusprimal.primal_dual as primal_dual
import optimusprimal.grad_operators as grad_operators
import optimusprimal.linear_operators as linear_operators
import optimusprimal.prox_operators as prox_operators


def test_l1_constrained():
    options = {"tol": 1e-5, "iter": 5000, "update_iter": 50, "record_iters": False}
    ISNR = 40.0
    sigma = 10 ** (-ISNR / 20.0)
    size = 1024
    epsilon = np.sqrt(size + 2.0 * np.sqrt(size)) * sigma
    x = np.linspace(0, 1 * np.pi, size)

    W = np.ones((size,))

    y = W * x + np.random.normal(0, sigma, size)

    p = prox_operators.l2_ball(epsilon, y, linear_operators.diag_matrix_operator(W))

    wav = ["db1", "db4"]
    levels = 6
    shape = (size,)
    psi = linear_operators.dictionary(wav, levels, shape)

    h = prox_operators.l1_norm(np.max(np.abs(psi.dir_op(y))) * 1e-3, psi)
    h.beta = 1.0
    f = prox_operators.real_prox()
    z, diag = primal_dual.FBPD(y, options, None, f, h, p)
    assert np.linalg.norm(z - W * y) < epsilon * 1.05
    assert diag["max_iter"] < 500
    # testing warm start
    z1, diag1 = primal_dual.FBPD_warm_start(
        z, diag["y"], diag["z"], diag["w"], options, None, f, h, p
    )
    assert diag1["max_iter"] < diag["max_iter"]


def test_l1_unconstrained():
    options = {"tol": 1e-5, "iter": 500, "update_iter": 50, "record_iters": False}
    ISNR = 20.0
    sigma = 10 ** (-ISNR / 20.0)
    size = 1024
    epsilon = np.sqrt(size + 2.0 * np.sqrt(size)) * sigma
    x = np.linspace(0, 1 * np.pi, size)

    W = np.ones((size,))

    y = W * x + np.random.normal(0, sigma, size)

    g = grad_operators.l2_norm(sigma, y, linear_operators.diag_matrix_operator(W))

    wav = ["db1", "db4"]
    levels = 6
    shape = (size,)
    psi = linear_operators.dictionary(wav, levels, shape)

    h = prox_operators.l1_norm(np.max(np.abs(psi.dir_op(y))) * 5e-3, psi)
    h.beta = 1.0
    f = prox_operators.real_prox()
    z, diag = primal_dual.FBPD(y, options, g, f, h)
    # assert(diag['max_iter'] < 500)


def test_l1_unconstrained_fb():
    options = {"tol": 1e-6, "iter": 5000, "update_iter": 50, "record_iters": False}
    ISNR = 50.0
    sigma = 10 ** (-ISNR / 20.0)
    size = 32
    epsilon = np.sqrt(size + 2.0 * np.sqrt(size)) * sigma
    x = np.linspace(0, 30, size)

    W = np.ones((size,))

    y = W * x + np.random.normal(0, sigma, size)
    g = grad_operators.l2_norm(sigma, y, linear_operators.diag_matrix_operator(W))
    g.beta = 1 / sigma ** 2
    psi = linear_operators.diag_matrix_operator(W)
    gamma = 50
    h = prox_operators.l1_norm(gamma, psi)
    h.beta = 1.0
    f = prox_operators.real_prox()
    z, diag = primal_dual.FBPD(x, options, g, f, h)
    z_fb = x
    mu = 0.5 * sigma ** 2
    for it in range(5000):
        z_fb = h.prox(z_fb - mu * g.grad(z_fb), mu)
    h = prox_operators.l1_norm(gamma * sigma ** 2, psi)
    solution = h.prox(y, 1)
    assert diag["max_iter"] < 500
    assert np.isclose(z, z_fb, 1e-3).all()
    assert np.isclose(z, solution, 1e-3).all()
