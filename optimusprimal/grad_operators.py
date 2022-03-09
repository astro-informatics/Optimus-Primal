import abc
import numpy as np
import optimusprimal.linear_operators as linear_operators
import tensorflow as tf


class Gradient(metaclass=abc.ABCMeta):
    """Base abstract class for gradient classes"""

    @abc.abstractmethod
    def __init__(self, data, Phi):
        """Constructor setting the hyper-parameters and domains of the gradient.

        Must be implemented by derived class (currently abstract).

        Args:
            data (np.ndarray): observed data
            Phi  (linear operator): sensing operator
        """

    @abc.abstractmethod
    def grad(self, x, gamma):
        """Evaluates the l2-ball prox of x

        Args:

            x (np.ndarray): Array to evaluate proximal gradient
            gamma (float): weighting of proximal gradient
        """
        return self.model(x)

    @abc.abstractmethod
    def fun(self, x):
        """Evaluates the loss of functional term

        Args:

            x (np.ndarray): Array to evaluate model loss of

        """


class l2_norm(Gradient):
    """This class computes the gradient operator of the l2 norm function.

                        f(x) = ||y - Phi x||^2/2/sigma^2

    When the input 'x' is an array. 'y' is a data vector, `sigma` is a scalar uncertainty
    """

    def __init__(self, sigma, data, Phi):
        """Initialises the l2_norm class

        Args:

            sigma (double): Noise standard deviation
            data (np.ndarray): Observed data
            Phi (Linear operator): Sensing operator

        Raises:

            ValueError: Raised when noise std is not positive semi-definite

        """

        if np.any(sigma <= 0):
            raise ValueError("'sigma' must be positive")
        self.sigma = sigma
        self.data = data
        self.beta = 1.0 / sigma ** 2
        if np.any(Phi is None):
            self.Phi = linear_operators.identity
        else:
            self.Phi = Phi

    def grad(self, x):
        """Computes the gradient of the l2_norm class

        Args:

            x (np.ndarray): Data estimate

        Returns:

            Gradient of the l2_norm expression

        """
        return self.Phi.adj_op((self.Phi.dir_op(x) - self.data)) / self.sigma ** 2

    def fun(self, x):
        """Evaluates the l2_norm class

        Args:

            x (np.ndarray): Data estimate

        Returns:

            Computes the l2_norm loss

        """
        return np.sum(np.abs(self.data - self.Phi.dir_op(x)) ** 2.0) / (
            2 * self.sigma ** 2
        )


class l2_norm_tf(tf.keras.layers.Layer):
    """This class computes the gradient operator of the l2 norm function in tensorflow.

                        f(x) = ||y - Phi x||^2/2/sigma^2

    When the input 'x' is a tensor. 'y' is a data tensor, `sigma` is a scalar uncertainty
    """

    def __init__(self, sigma, data, Phi, shape_x, shape_y):
        """Initialises the l2_norm_tf class

        Args:

            sigma (double): Noise standard deviation
            data (tf.tensor): Observed data
            Phi (Linear operator): Sensing operator

        Raises:

            ValueError: Raised when noise std is not positive semi-definite

        """

        if np.any(sigma <= 0):
            raise ValueError("'sigma' must be positive")
        self.sigma = sigma
        self.data = data
        self.beta = 1.0 / sigma ** 2
        self.Phi = Phi
        self.input_spec = [
            tf.keras.layers.InputSpec(dtype=tf.float32, shape=shape_x),
            tf.keras.layers.InputSpec(dtype=tf.complex64, shape=shape_y),
        ]
        self.depth = 1
        self.trainable = False

    def grad(self, x):
        """Wraps the layer call for gradient of the l2_norm class

        Args:

            x (tf.tensor): Data estimate

        Returns:

            Gradient of the l2_norm expression

        """
        return self.__call__(x)[0]

    def __call__(self, x):
        """Computes the gradient of the l2_norm class

        Args:

            x (tf.tensor): Data estimate

        Returns:

            Gradient of the l2_norm expression

        """
        tmp = tf.cast(x, tf.complex64)
        tmp = self.Phi.dir_op(tmp) - self.data
        return tf.cast(self.Phi.adj_op(tmp), tf.float32)

    def fun(self, x):
        """Evaluates the l2_norm class

        Args:

            x (np.ndarray): Data estimate

        Returns:

            Computes the l2_norm loss

        """
        tmp = tf.cast(x, tf.complex64)
        return np.sum(np.abs(self.data - self.Phi.dir_op(tmp)) ** 2.0) / (
            2 * self.sigma ** 2
        )
