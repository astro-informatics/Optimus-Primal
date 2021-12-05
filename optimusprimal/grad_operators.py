import numpy as np
import optimusprimal.linear_operators as linear_operators


class l2_norm:
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
