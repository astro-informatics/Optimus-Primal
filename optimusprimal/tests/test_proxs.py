import optimusprimal.prox_operators as prox_operators
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
