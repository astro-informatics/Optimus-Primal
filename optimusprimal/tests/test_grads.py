import optimusprimal.grad_operators as grad_operators
import optimusprimal.linear_operators as linear_operators
import pytest
from numpy import linalg as LA
import numpy as np


def test_l2_grad():
    y = np.ones(10)
    sigma = 1
    A = np.random.normal(0, 10.0, (10, 10)) * 1j
    Phi = linear_operators.matrix_operator(A)
    op = grad_operators.l2_norm(sigma, y, Phi)
    output = op.grad(0 * y)
    assert np.allclose(np.conj(A.T) @ (-y) / sigma ** 2, output, 1e-6)
