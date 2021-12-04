import numpy as np
import pywt
import logging
import scipy.fft

logger = logging.getLogger("Optimus Primal")


def power_method(op, x_init, tol=1e-3, iters=1000):
    """Power method which returns the operator norm^2 and the eigen vector."""
    x_old = x_init
    val_old = 1
    logger.info("Starting Power method")
    for i in range(iters):
        x_new = op.adj_op(op.dir_op(x_old))
        val_new = np.linalg.norm(x_new)
        if np.abs(val_new - val_old) < tol * val_old:
            logger.info(
                "[Power Method] Converged with norm= %s, iter = %s, tol = %s",
                val_new,
                i,
                np.abs(val_new - val_old) / np.abs(val_old),
            )
            break
        x_old = x_new / val_new
        val_old = val_new
        if i % 10 == True:
            logger.info(
                "[Power Method] iter = %s, tol = %s",
                i,
                np.abs(val_new - val_old) / np.abs(val_old),
            )
    return val_new, x_new


class identity:
    """Identity operator."""

    def dir_op(self, x):
        return x

    def adj_op(self, x):
        return x


class projection:
    """Projection wrapper for linear operator"""

    def __init__(self, linear_op, index, shape):
        self.linear_op = linear_op
        self.shape = shape
        self.index = index

    def dir_op(self, x):
        return self.linear_op.dir_op(x[self.index, ...])

    def adj_op(self, x):
        z = np.zeros(self.shape, dtype=complex)
        z[self.index, ...] = self.linear_op.adj_op(x)
        return z


class sum:
    """Sum wrapper for linear operator"""

    def __init__(self, linear_op, shape):
        self.linear_op = linear_op
        self.shape = shape

    def dir_op(self, x):
        return self.linear_op.dir_op(np.sum(x, axis=0))

    def adj_op(self, x):
        z = np.zeros(self.shape, dtype=complex)
        z[:, ...] = self.linear_op.adj_op(x)
        return z


class weights:
    """weights wrapper for linear operator"""

    def __init__(self, linear_op, weights):
        self.linear_op = linear_op
        self.weights = weights

    def dir_op(self, x):
        return self.linear_op.dir_op(x) * self.weights

    def adj_op(self, x):
        return self.linear_op.adj_op(x * np.conj(self.weights))


class function_wrapper:
    """Given direct and adjoint functions return linear operator.

     INPUTS
    ========
    dir_op  - forward operator
    adj_op  - adjoint operator
    """

    dir_op = None
    adj_op = None

    def __init__(self, dir_op, adj_op):
        self.dir_op = dir_op
        self.adj_op = adj_op


class fft_operator:
    """Applies nd fft operator to nd signal."""

    def __init__(self):
        self.dir_op = np.fft.fftn
        self.adj_op = np.fft.ifftn


class dct_operator:
    """Applies nd discrete cosine transform to nd signal"""

    def dir_op(self, x):
        return scipy.fft.dctn(x, norm="ortho")

    def adj_op(self, x):
        return scipy.fft.idctn(x, norm="ortho")


class diag_matrix_operator:
    """
    Applies diagonal matrix operator W * x

     INPUTS
    ========
    W - array of weights
    """

    def __init__(self, W):
        self.W = W

    def dir_op(self, x):
        return self.W * x

    def adj_op(self, x):
        return np.conj(self.W) * x


class matrix_operator:
    """
    Applies matrix operator A * x

     INPUTS
    ========
    A - numpy matrix
    """

    def __init__(self, A):
        self.A = A
        self.A_H = np.conj(A.T)

    def dir_op(self, x):
        return self.A @ x

    def adj_op(self, x):
        return self.A_H @ x


class db_wavelets:
    def __init__(self, wav, levels, shape, axes=None):

        if np.any(levels <= 0):
            raise Exception("'levels' must be positive")
        if axes is None:
            axes = range(len(shape))
        self.axes = axes
        self.wav = wav
        self.levels = np.array(levels, dtype=int)
        self.shape = shape
        self.coeff_slices = None
        self.coeff_shapes = None

        self.adj_op(self.dir_op(np.ones(shape)))

    def dir_op(self, x):
        if self.wav == "dirac":
            return np.ravel(x)
        if self.wav == "fourier":
            return np.ravel(np.fft.fftn(x))
        if self.wav == "dct":
            return np.ravel(scipy.fft.dctn(x, norm="ortho"))
        if self.shape[0] % 2 == 1:
            raise Exception("Signal shape should be even dimensions.")
        if len(self.shape) > 1:
            if self.shape[1] % 2 == 1:
                raise Exception("Signal shape should be even dimensions.")

        coeffs = pywt.wavedecn(
            x, wavelet=self.wav, level=self.levels, mode="periodic", axes=self.axes
        )
        arr, self.coeff_slices, self.coeff_shapes = pywt.ravel_coeffs(
            coeffs, axes=self.axes
        )
        return arr

    def adj_op(self, x):
        if self.wav == "dirac":
            return np.reshape(x, self.shape)
        if self.wav == "fourier":
            return np.fft.ifftn(np.reshape(x, self.shape))
        if self.wav == "dct":
            return scipy.fft.idctn(np.reshape(x, self.shape), norm="ortho")
        coeffs_from_arr = pywt.unravel_coeffs(
            x, self.coeff_slices, self.coeff_shapes, output_format="wavedecn"
        )
        return pywt.waverecn(
            coeffs_from_arr, wavelet=self.wav, mode="periodic", axes=self.axes
        )


class dictionary:
    sizes = []
    wavelet_list = []

    def __init__(self, wav, levels, shape, axes=None):

        self.wavelet_list = []
        self.sizes = np.zeros(len(wav))
        if axes is None:
            axes = []
            for i in range(len(wav)):
                axes.append(range(len(shape)))
        if np.isscalar(levels):
            levels = np.ones(len(wav)) * levels
        for i in range(len(wav)):
            self.wavelet_list.append(db_wavelets(wav[i], levels[i], shape, axes[i]))

    def dir_op(self, x):
        out = self.wavelet_list[0].dir_op(x)
        self.sizes[0] = out.shape[0]
        for wav_i in range(1, len(self.wavelet_list)):
            buff = self.wavelet_list[wav_i].dir_op(x)
            self.sizes[wav_i] = buff.shape[0]
            out = np.concatenate((out, buff), axis=0)
        return out / np.sqrt(len(self.wavelet_list))

    def adj_op(self, x):
        offset = 0
        out = 0
        for wav_i in range(len(self.wavelet_list)):
            size = self.sizes[wav_i]
            x_block = x[int(offset) : int(offset + size)]
            buff = self.wavelet_list[wav_i].adj_op(x_block)
            out += buff / np.sqrt(len(self.wavelet_list))
            offset += size
        return out
