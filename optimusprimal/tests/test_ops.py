import optimusprimal.linear_operators as linear_operators
import pytest
from numpy import linalg as LA
import numpy as np


def forward_operator(op, inp, result):
    assert np.all(op.dir_op(inp) == result)


def adjoint_operator(op, inp, result):
    assert np.all(op.adj_op(inp) == result)


def test_power_method():
    A = np.diag(np.random.normal(0, 10.0, (10)))
    op = linear_operators.matrix_operator(A)
    inp = np.random.normal(0, 10.0, (10)) * 0 + 1
    val, x_e = linear_operators.power_method(op, inp, 1e-5, 10000)
    w, v = LA.eig(A)
    expected = np.max(np.abs(w)) ** 2
    assert np.allclose(val, expected, 1e-3)


def test_id_op():
    id_op = linear_operators.identity()
    inp = np.random.normal(0, 10.0, (10, 10))
    out = inp
    forward_operator(id_op, inp, out)
    adjoint_operator(id_op, inp, out)
    inp = np.random.normal(0, 10.0, (10))
    out = inp
    forward_operator(id_op, inp, out)
    adjoint_operator(id_op, inp, out)


def test_matrix_op():
    A = np.random.normal(0, 10.0, (10, 5)) * 1j
    op = linear_operators.matrix_operator(A)
    inp = np.random.normal(0, 10.0, (5))
    out = A @ inp
    forward_operator(op, inp, out)
    inp = np.random.normal(0, 10.0, (10))
    out = np.conj(A.T) @ inp
    adjoint_operator(op, inp, out)


def test_diag_matrix_op():
    A = np.random.normal(0, 10.0, (10)) * 1j
    op = linear_operators.diag_matrix_operator(A)
    inp = np.random.normal(0, 10.0, (10))
    out = A * inp
    forward_operator(op, inp, out)
    inp = np.random.normal(0, 10.0, (10))
    out = np.conj(A) * inp
    adjoint_operator(op, inp, out)


def test_wav_op():
    wav = "dirac"
    levels = 3
    shape = (128,)
    op = linear_operators.db_wavelets(wav, levels, shape)
    inp = np.random.normal(0, 10.0, shape)
    out = op.dir_op(inp)
    forward_operator(op, inp, out)
    inp = np.random.normal(0, 10.0, out.shape)
    out = op.adj_op(inp)
    adjoint_operator(op, inp, out)
    buff1 = op.adj_op(op.dir_op(out))
    assert np.allclose(out, buff1, 1e-6)
    wav = "db1"
    levels = 3
    shape = (128,)
    op = linear_operators.db_wavelets(wav, levels, shape)
    inp = np.random.normal(0, 10.0, shape)
    out = op.dir_op(inp)
    forward_operator(op, inp, out)
    inp = np.random.normal(0, 10.0, out.shape)
    out = op.adj_op(inp)
    adjoint_operator(op, inp, out)
    buff1 = op.adj_op(op.dir_op(out))
    assert np.allclose(out, buff1, 1e-6)
    wav = ["db1", "db2", "dirac"]
    levels = 3
    shape = (128,)
    op = linear_operators.dictionary(wav, levels, shape)
    inp = np.random.normal(0, 10.0, shape)
    out = op.dir_op(inp)
    forward_operator(op, inp, out)
    inp = np.random.normal(0, 10.0, out.shape)
    out = op.adj_op(inp)
    adjoint_operator(op, inp, out)
    buff1 = op.adj_op(op.dir_op(out))
    assert np.allclose(out, buff1, 1e-6)
    wav = "db2"
    levels = 3
    shape = (128, 128)
    op = linear_operators.db_wavelets(wav, levels, shape)
    inp = np.random.normal(0, 10.0, shape)
    out = op.dir_op(inp)
    forward_operator(op, inp, out)
    inp = np.random.normal(0, 10.0, out.shape)
    out = op.adj_op(inp)
    adjoint_operator(op, inp, out)
    buff1 = op.adj_op(op.dir_op(out))
    assert np.allclose(out, buff1, 1e-6)
    wav = ["db1", "db2", "dirac"]
    levels = 3
    shape = (128, 128)
    op = linear_operators.dictionary(wav, levels, shape)
    inp = np.random.normal(0, 10.0, shape)
    out = op.dir_op(inp)
    forward_operator(op, inp, out)
    inp = np.random.normal(0, 10.0, out.shape)
    out = op.adj_op(inp)
    adjoint_operator(op, inp, out)
    buff1 = op.adj_op(op.dir_op(out))
