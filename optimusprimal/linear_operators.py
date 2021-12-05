import numpy as np
import pywt
import logging
import scipy.fft

logger = logging.getLogger("Optimus Primal")


def power_method(op, x_init, tol=1e-3, iters=1000):
    """Power method to compute operator norms and eigenvector

    Args:

        op (Linear operator): Operator to normalise
        x_init (np.ndarray): Random vector of correct length
        tol (double): Tolerance for convergence
        iters (int): Maximum power method iterations

    Returns:

        Tuple of Operator norm^2 and largest Eigen-vector.

    """
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
    """
    Identity linear operator

    """

    def dir_op(self, x):
        """Computes the forward operator of the identity class

        Args:

            x (np.ndarray): Vector to apply identity to
        """
        return x

    def adj_op(self, x):
        """Computes the forward adjoint operator of the identity class

        Args:

            x (np.ndarray): Vector to apply identity to
        """
        return x


class projection:
    """
    Projection wrapper for linear operator
    """

    def __init__(self, linear_op, index, shape):
        """Initialises an abstract projection class

        Args:

            linear_op (Linear operator): Operator to apply e.g identity etc.
            index (int): entry of vector to operate on
            shape (list[int]): dimension of adjoint space
        """
        self.linear_op = linear_op
        self.shape = shape
        self.index = index

    def dir_op(self, x):
        """Computes the forward operator of an abstract projection class

        Args:

            x (np.ndarray): Vector to project
        """
        return self.linear_op.dir_op(x[self.index, ...])

    def adj_op(self, x):
        """Computes the forward adjoint operator of an abstract projection class

        Args:

            x (np.ndarray): Vector to adjoint project
        """
        z = np.zeros(self.shape, dtype=complex)
        z[self.index, ...] = self.linear_op.adj_op(x)
        return z


class sum:
    """
    Sum wrapper for abstract linear operator
    """

    def __init__(self, linear_op, shape):
        """Initialises an abstract sum class

        Args:

            linear_op (Linear operator): Operator to apply e.g identity etc.
            shape (list[int]): dimension of adjoint space
        """
        self.linear_op = linear_op
        self.shape = shape

    def dir_op(self, x):
        """Computes the forward operator of an abstract sum class

        Args:

            x (np.ndarray): Vector for sumnation
        """
        return self.linear_op.dir_op(np.sum(x, axis=0))

    def adj_op(self, x):
        """Computes the forward adjoint operator of an abstract sum class

        Args:

            x (np.ndarray): Vector for adjoint sumnation
        """
        z = np.zeros(self.shape, dtype=complex)
        z[:, ...] = self.linear_op.adj_op(x)
        return z


class weights:
    """
    weights wrapper for abstract linear operator
    """

    def __init__(self, linear_op, weights):
        """Initialises an abstract weights class

        Args:

            linear_op (Linear operator): Operator to apply e.g identity etc.
            weights (np.ndarray): Data-space weighting
        """
        self.linear_op = linear_op
        self.weights = weights

    def dir_op(self, x):
        """Computes the forward adjoint operator of an abstract weight class

        Args:

            x (np.ndarray): Vector to re-weight
        """
        return self.linear_op.dir_op(x) * self.weights

    def adj_op(self, x):
        """Computes the forward adjoint operator of an abstract weight class

        Args:

            x (np.ndarray): Vector to adjoint re-weight
        """
        return self.linear_op.adj_op(x * np.conj(self.weights))


class function_wrapper:
    """
    Given direct and adjoint functions return linear operator.

    """

    dir_op = None
    adj_op = None

    def __init__(self, dir_op, adj_op):
        """Constructs a linear operator from an abstract forward and adjoint transform

        Args:

            dir_op (function): Forward operation
            adj_op (function): Forward adjoint operation
        """
        self.dir_op = dir_op
        self.adj_op = adj_op


class fft_operator:
    """
    Constructs an N-dimensional FFT linear operator class
    """

    def __init__(self):
        """Constructs a linear operator for an N-dimensional FFT

        Args:

            dir_op (function): Forward N-dimensional FFT
            adj_op (function): Inverse (Forward adjoint) N-dimensional FFT
        """
        self.dir_op = np.fft.fftn
        self.adj_op = np.fft.ifftn


class dct_operator:
    """
    Constructs an N-dimensional Discrete Cosine Transform class
    """

    def dir_op(self, x):
        """Evaluates the forward discrete N-dimensional cosine transfrom

        Args:

            x (np.ndarray): Vector to transform
        """
        return scipy.fft.dctn(x, norm="ortho")

    def adj_op(self, x):
        """Evaluates the inverse (forward adjoint) discrete N-dimensional cosine transfrom

        Args:

            x (np.ndarray): Vector to transform
        """
        return scipy.fft.idctn(x, norm="ortho")


class diag_matrix_operator:
    """
    Constructs a linear operator for coefficient wise multiplication W * x
    """

    def __init__(self, W):
        """Initialises a diagonal matrix multiplication class

        Args:

            W (np.ndarray): Array of weights
        """
        self.W = W

    def dir_op(self, x):
        """Multiplies the input with weight array

        Args:

            x (np.ndarray): Vector to re-weight
        """
        return self.W * x

    def adj_op(self, x):
        """Multiplies the input with the adjoint weight array

        Args:

            x (np.ndarray): Vector to re-weight
        """
        return np.conj(self.W) * x


class matrix_operator:
    """
    Constructs a linear operator for matrix multiplication A * x
    """

    def __init__(self, A):
        """Initialises matrix multiplication linear operator class

        Args:

            A (np.ndarray): Matrix kernel to multiply
        """
        self.A = A
        self.A_H = np.conj(A.T)

    def dir_op(self, x):
        """Matrix multiplies x with A

        Args:

            x (np.ndarray): Vector to left matrix multiply by A
        """
        return self.A @ x

    def adj_op(self, x):
        """Multiplies the input with the adjoint weight array

        Args:

            x (np.ndarray): Vector to left matrix multiply by A
        """
        return self.A_H @ x


class db_wavelets:
    """
    Constructs a linear operator for abstract Daubechies Wavelets
    """

    def __init__(self, wav, levels, shape, axes=None):
        """Initialises Daubechies Wavelet linear operator class

        Args:

            wav (string): Wavelet type (see https://tinyurl.com/5n7wzpmb)
            levels (list[int]): Wavelet levels (scales) to consider
            shape (list[int]): Dimensionality of input to wavelet transform
            axes (int): Which axes to perform wavelet transform (default = all axes)

        Raises:

            ValueError: Raised when levels are not positive definite

        """

        if np.any(levels <= 0):
            raise ValueError("'levels' must be positive")
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
        """Evaluates the forward abstract wavelet transform of x

        Args:

            x (np.ndarray): Array to wavelet transform

        Raises:

            ValueError: Raised when the shape of x is not even in every dimension
        """
        if self.wav == "dirac":
            return np.ravel(x)
        if self.wav == "fourier":
            return np.ravel(np.fft.fftn(x))
        if self.wav == "dct":
            return np.ravel(scipy.fft.dctn(x, norm="ortho"))
        if self.shape[0] % 2 == 1:
            raise ValueError("Signal shape should be even dimensions.")
        if len(self.shape) > 1:
            if self.shape[1] % 2 == 1:
                raise ValueError("Signal shape should be even dimensions.")

        coeffs = pywt.wavedecn(
            x, wavelet=self.wav, level=self.levels, mode="periodic", axes=self.axes
        )
        arr, self.coeff_slices, self.coeff_shapes = pywt.ravel_coeffs(
            coeffs, axes=self.axes
        )
        return arr

    def adj_op(self, x):
        """Evaluates the forward adjoint abstract wavelet transform of x

        Args:

            x (np.ndarray): Array to adjoint wavelet transform

        """
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
    """
    Constructs class to permit sparsity averaging across a collection of wavelet dictionaries
    """

    sizes = []
    wavelet_list = []

    def __init__(self, wav, levels, shape, axes=None):
        """Initialises a linear operator for a collection of abstract wavelet dictionaries

        Args:

            wav (list[string]): List of wavelet types (see https://tinyurl.com/5n7wzpmb)
            levels (list[int]): Wavelet levels (scales) to consider
            shape (list[int]): Dimensionality of input to wavelet transform
            axes (int): Which axes to perform wavelet transform (default = all axes)

        Raises:

            ValueError: Raised when levels are not positive definite

        """
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
        """Evaluates a list of forward abstract wavelet transforms of x

        Args:

            x (np.ndarray): Array to wavelet transform

        """
        out = self.wavelet_list[0].dir_op(x)
        self.sizes[0] = out.shape[0]
        for wav_i in range(1, len(self.wavelet_list)):
            buff = self.wavelet_list[wav_i].dir_op(x)
            self.sizes[wav_i] = buff.shape[0]
            out = np.concatenate((out, buff), axis=0)
        return out / np.sqrt(len(self.wavelet_list))

    def adj_op(self, x):
        """Evaluates a list of forward adjoint abstract wavelet transforms of x

        Args:

            x (np.ndarray): Array to adjoint wavelet transform

        """
        offset = 0
        out = 0
        for wav_i in range(len(self.wavelet_list)):
            size = self.sizes[wav_i]
            x_block = x[int(offset) : int(offset + size)]
            buff = self.wavelet_list[wav_i].adj_op(x_block)
            out += buff / np.sqrt(len(self.wavelet_list))
            offset += size
        return out
