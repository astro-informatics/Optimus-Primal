import optimusprimal.map_uncertainty as map_uncertainty
import pytest
import numpy as np


def test_bisection_1D_2D():
    def obj(x):
        return np.sum(x ** 2) - 9

    val = map_uncertainty.bisection_method(obj, [0, 10], 1e3, 1e-3)
    assert np.allclose(val, 3.0, 1e-2)

    def obj(x):
        return np.sum(x ** 2) - 9

    val = map_uncertainty.bisection_method(obj, [-10, 0], 1e3, 1e-3)
    assert np.allclose(val, -3.0, 1e-2)

    x_sol = np.zeros((128, 128))
    region_size = 16

    def obj(x):
        return np.sum(np.abs(x) ** 2)

    bound = 4.0
    iters = 1e5
    tol = 1e-12
    error_p, error_m, mean = map_uncertainty.create_local_credible_interval(
        x_sol, region_size, obj, bound, iters, tol, -5, 5
    )
    assert np.allclose(error_p.shape, (128 / region_size, 128 / region_size), 1e-3)
    assert np.allclose(error_m.shape, (128 / region_size, 128 / region_size), 1e-3)
    assert np.allclose(error_p, 2.0 / region_size, 1e-3)
    assert np.allclose(error_m, -2.0 / region_size, 1e-3)
    assert np.allclose(mean, 0.0, 1e-12)
    x_sol = np.zeros((128,))
    region_size = 16

    def obj(x):
        return np.sum(np.abs(x) ** 2)

    bound = 4
    iters = 1e5
    tol = 1e-12
    error_p, error_m, mean = map_uncertainty.create_local_credible_interval(
        x_sol, region_size, obj, bound, iters, tol, -5, 5
    )
    assert np.allclose(error_p.shape, (128 / region_size,), 1e-3)
    assert np.allclose(error_m.shape, (128 / region_size,), 1e-3)
    assert np.allclose(error_p, 2.0 / np.sqrt(region_size), 1e-3)
    assert np.allclose(error_m, -2.0 / np.sqrt(region_size), 1e-3)


def test_bisection_fast_1D_2D():
    import optimusprimal.linear_operators as lin_ops

    psi = lin_ops.identity()
    phi = lin_ops.identity()

    x_sol = np.zeros((128, 128))
    region_size = 16

    def map_loss(x):
        return np.sum(2 * np.abs(x) ** 2)

    def map_loss_fast(x, xm, w, wm):
        return (
            np.sum(np.abs(x) ** 2)
            + np.sum(np.abs(xm) ** 2)
            + np.sum(np.abs(w) ** 2)
            + np.sum(np.abs(wm) ** 2)
        )

    bound = 4.0
    iters = 1e5
    tol = 1e-12

    error_p, error_m, mean = map_uncertainty.create_local_credible_interval(
        x_sol, region_size, map_loss, bound, iters, tol, -5, 5
    )

    (
        error_p_fast,
        error_m_fast,
        mean_fast,
    ) = map_uncertainty.create_local_credible_interval_fast(
        x_sol, phi, psi, region_size, map_loss_fast, bound, iters, tol, -5, 5
    )
    assert np.allclose(error_p, error_p_fast, 1e-3)
    assert np.allclose(error_m, error_m_fast, 1e-3)
    assert np.allclose(mean, mean_fast, 1e-12)

    x_sol = np.zeros((128,))
    region_size = 16

    error_p, error_m, mean = map_uncertainty.create_local_credible_interval(
        x_sol, region_size, map_loss, bound, iters, tol, -5, 5
    )

    (
        error_p_fast,
        error_m_fast,
        mean_fast,
    ) = map_uncertainty.create_local_credible_interval_fast(
        x_sol, phi, psi, region_size, map_loss, bound, iters, tol, -5, 5
    )
    assert np.allclose(error_p, error_p_fast, 1e-3)
    assert np.allclose(error_m, error_m_fast, 1e-3)
    assert np.allclose(mean, mean_fast, 1e-12)
