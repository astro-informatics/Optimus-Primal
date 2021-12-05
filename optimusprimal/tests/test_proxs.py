import optimusprimal.prox_operators as prox_operators
import optimusprimal.linear_operators as linear_operators
import pytest

import numpy as np


def test_l2_ball_op():
    epsilon = 2.0
    inp = np.random.normal(0, 10.0, (10, 10))
    inp = inp / np.sqrt(np.sum(np.abs(inp) ** 2)) * epsilon * 2
    out = inp / epsilon
    op = prox_operators.l2_ball(epsilon, inp * 0.0)
    assert op.fun(inp) >= 0
    assert np.allclose(op.prox(inp, 1), out, 1e-6)
    epsilon = 2.0
    inp = np.random.normal(0, 10.0, (10,))
    inp = inp / np.sqrt(np.sum(np.abs(inp) ** 2)) * epsilon * 0.9
    out = inp
    op = prox_operators.l2_ball(epsilon, inp * 0.0)
    assert np.allclose(op.prox(inp, 1), out, 1e-6)
    assert op.fun(inp) >= 0


def test_l2_ball_errors():
    with pytest.raises(ValueError):
        prox_operators.l2_ball(epsilon=0, data=None)


def test_l_inf_ball_op():
    epsilon = 2.0
    inp = np.random.normal(0, 10.0, (10, 10))
    out = np.zeros_like(inp)
    out = np.minimum(epsilon, np.abs(out)) * np.exp(complex(0, 1) * np.angle(out)) + inp
    op = prox_operators.l_inf_ball(epsilon, inp)
    assert op.fun(inp) >= 0
    assert np.allclose(op.prox(inp, 0.0), out, 1e-6)

    op = prox_operators.l_inf_ball(epsilon, inp, Phi=linear_operators.identity())
    assert op.fun(inp) >= 0
    assert np.allclose(op.prox(inp, 0.0), out, 1e-6)

    x = op.dir_op(inp)
    xx = op.adj_op(inp)
    assert np.allclose(x, xx, 1e-10)
    assert np.allclose(x, inp, 1e-10)


def test_l_inf_ball_errors():
    with pytest.raises(ValueError):
        prox_operators.l_inf_ball(epsilon=0, data=None)


def test_l1_norm_op():
    gamma = 2
    inp = np.random.normal(0, 10.0, (10, 10))
    out = np.maximum(0, np.abs(inp) - gamma) * np.exp(complex(0, 1) * np.angle(inp))
    op = prox_operators.l1_norm(gamma)
    assert op.fun(inp) >= 0
    assert np.allclose(op.prox(inp, 1), out, 1e-6)

    gamma = np.abs(np.random.normal(0, 3.0, (10, 10)))
    inp = np.random.normal(0, 10.0, (10, 10))
    out = np.maximum(0, np.abs(inp) - gamma) * np.exp(complex(0, 1) * np.angle(inp))
    op = prox_operators.l1_norm(gamma)
    assert op.fun(inp) >= 0
    assert np.allclose(op.prox(inp, 1), out, 1e-6)


def test_l1_norm_op_errors():
    with pytest.raises(ValueError):
        prox_operators.l1_norm(gamma=0)


def test_l2_square_norm_op():
    sigma = tau = 2
    inp = np.random.normal(0, 10.0, (10, 10))
    out = inp / (tau / sigma ** 2 + 1.0)
    op = prox_operators.l2_square_norm(sigma)
    assert op.fun(inp) == pytest.approx(np.sum(np.abs(inp) ** 2 / (2.0 * sigma ** 2)))
    assert np.allclose(op.prox(inp, tau), out, 1e-6)

    op = prox_operators.l2_square_norm(sigma, Psi=linear_operators.identity())
    assert op.fun(inp) == pytest.approx(np.sum(np.abs(inp) ** 2 / (2.0 * sigma ** 2)))
    assert np.allclose(op.prox(inp, tau), out, 1e-6)

    x = op.dir_op(inp)
    xx = op.adj_op(inp)
    assert np.allclose(x, xx, 1e-10)
    assert np.allclose(x, inp, 1e-10)


def test_l2_square_norm_op_errors():
    with pytest.raises(ValueError):
        prox_operators.l2_square_norm(sigma=0)


def test_positive_prox_op():
    inp = np.random.normal(0, 10.0, (10, 10))
    out = np.zeros_like(inp)
    out[inp >= 0] = inp[inp >= 0]
    op = prox_operators.positive_prox()
    assert op.fun(inp) == pytest.approx(0.0)
    assert np.allclose(op.prox(inp, 1), out, 1e-6)

    x = op.dir_op(inp)
    xx = op.adj_op(inp)
    assert np.allclose(x, xx, 1e-10)
    assert np.allclose(x, inp, 1e-10)


def test_real_prox_op():
    inp = np.random.normal(0, 10.0, (10, 10)) + 1j * np.random.normal(0, 10.0, (10, 10))
    out = np.real(inp)
    op = prox_operators.real_prox()
    assert op.fun(inp) == pytest.approx(0.0)
    assert np.allclose(op.prox(inp, 1), out, 1e-6)

    x = op.dir_op(inp)
    xx = op.adj_op(inp)
    assert np.allclose(x, xx, 1e-10)
    assert np.allclose(x, inp, 1e-10)


def test_zero_prox_op():
    inp = np.random.normal(0, 10.0, (10, 10))
    indices = np.where(inp > 0)
    out = np.copy(inp)
    out[out > 0] = 0.0
    op = prox_operators.zero_prox(
        indices=indices, op=linear_operators.identity(), offset=0
    )
    assert op.fun(inp) == pytest.approx(0.0)
    assert np.allclose(op.prox(inp, 1), out, 1e-6)

    x = op.dir_op(inp)
    xx = op.adj_op(inp)
    assert np.allclose(x, xx, 1e-10)
    assert np.allclose(x, inp, 1e-10)


def test_l21_norm():
    tau = 1.0
    l2axis = 0
    inp = np.random.normal(0, 10.0, (10, 10))
    out = np.expand_dims(np.sqrt(np.sum(np.square(np.abs(inp)), axis=l2axis)), l2axis)
    out = inp * (1 - tau / np.maximum(out, tau))
    op = prox_operators.l21_norm(tau, l2axis)
    assert op.fun(inp) == pytest.approx(0.0)
    assert np.allclose(op.prox(inp, tau), out, 1e-6)

    op = prox_operators.l21_norm(tau, l2axis, Phi=linear_operators.identity())
    assert op.fun(inp) == pytest.approx(0.0)
    assert np.allclose(op.prox(inp, tau), out, 1e-6)

    x = op.dir_op(inp)
    xx = op.adj_op(inp)
    assert np.allclose(x, xx, 1e-10)
    assert np.allclose(x, inp, 1e-10)


def test_l21_norm_errors():
    with pytest.raises(ValueError):
        prox_operators.l21_norm(tau=0)


def test_translate_prox():
    inp = np.random.normal(0, 10.0, (10, 10))

    out = np.maximum(0, 2 * np.real(inp)) - inp

    op = prox_operators.translate_prox(prox_operators.positive_prox(), inp)
    assert op.fun(inp) == pytest.approx(0.0)
    assert np.allclose(op.prox(inp, 1), out, 1e-6)

    x = op.dir_op(inp)
    xx = op.adj_op(inp)
    assert np.allclose(x, xx, 1e-10)
    assert np.allclose(x, inp, 1e-10)
