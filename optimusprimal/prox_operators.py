import optimusprimal.linear_operators as linear_operators
import numpy as np


class l2_ball:
    """This class computes the proximity operator of the l2 ball.

                        f(x) = (||Phi x - y|| < epsilon) ? 0. : infty

    When the input 'x' is an array. y is the data vector. Phi is the measurement operator.
    """

    def __init__(self, epsilon, data, Phi=None):
        """Initialises an l2_ball proximal operator class

        Args:

            epsilon (double): Radius of l2-ball
            data (np.ndarray): Data centred on the l2-ball
            Phi (Linear operator): Sensing/weighting operator

        Raises:

            ValueError: Raised if l2-ball radius is not strictly positive
        """

        if np.any(epsilon <= 0):
            raise ValueError("'epsilon' must be positive")
        self.epsilon = epsilon
        self.data = data
        self.beta = 1.0
        if Phi is None:
            self.Phi = linear_operators.identity()
        else:
            self.Phi = Phi

    def prox(self, x, gamma):
        """Evaluates the l2-ball prox of x

        Args:

            x (np.ndarray): Array to evaluate proximal projection of

        Returns:

            Proximal projection of x onto the l2-ball
        """
        xx = np.sqrt(np.sum(np.square(np.abs(x - self.data))))
        if xx < self.epsilon:
            p = x
        else:
            p = (x - self.data) * self.epsilon / xx + self.data

        return p

    def fun(self, x):
        """Evaluates loss of functional term

        Args:

            x (np.ndarray): Array to evaluate loss of

        Returns:

            0
        """
        return 0

    def dir_op(self, x):
        """Evaluates the forward sensing operator

        Args:

            x (np.ndarray): Array to transform

        Returns:

            Forward sensing operator applied to x
        """
        return self.Phi.dir_op(x)

    def adj_op(self, x):
        """Evaluates the forward adjoint sensing operator

        Args:

            x (np.ndarray): Array to adjoint transform

        Returns:

            Forward adjoint sensing operator applied to x
        """
        return self.Phi.adj_op(x)


class l_inf_ball:
    """This class computes the proximity operator of the l_inf ball.

                        f(x) = (||Phi x - y||_inf < epsilon) ? 0. : infty

    When the input 'x' is an array. y is the data vector. Phi is the measurement operator.
    """

    def __init__(self, epsilon, data, Phi=None):
        """Initialises an l2_ball proximal operator class

        Args:

            epsilon (double): Radius of l_inf-ball
            data (np.ndarray): Data centred on the l_inf-ball
            Phi (Linear operator): Sensing/weighting operator

        Raises:

            ValueError: Raised if l_inf-ball radius is not strictly postitive
        """

        if np.any(epsilon <= 0):
            raise ValueError("'epsilon' must be positive")
        self.epsilon = epsilon
        self.data = data
        self.beta = 1.0
        if Phi is None:
            self.Phi = linear_operators.identity()
        else:
            self.Phi = Phi

    def prox(self, x, gamma):
        """Evaluates the l_inf-ball prox of x

        Args:

            x (np.ndarray): Array to evaluate proximal projection of

        Returns:

            Proximal projection of x onto the l_inf-ball
        """
        z = x - self.data
        return (
            np.minimum(self.epsilon, np.abs(z)) * np.exp(complex(0, 1) * np.angle(z))
            + self.data
        )

    def fun(self, x):
        """Evaluates loss of functional term

        Args:

            x (np.ndarray): Array to evaluate loss of

        Returns:

            0
        """
        return 0

    def dir_op(self, x):
        """Evaluates the forward sensing operator

        Args:

            x (np.ndarray): Array to transform

        Returns:

            Forward sensing operator applied to x
        """
        return self.Phi.dir_op(x)

    def adj_op(self, x):
        """Evaluates the forward adjoint sensing operator

        Args:

            x (np.ndarray): Array to adjoint transform

        Returns:

            Forward adjoint sensing operator applied to x
        """
        return self.Phi.adj_op(x)


class l1_norm:
    """This class computes the proximity operator of the l2 ball.

                        f(x) = ||Psi x||_1 * gamma

    When the input 'x' is an array. gamma is a regularization term. Psi is a sparsity operator.
    """

    def __init__(self, gamma, Psi=None):
        """Initialises an l1-norm proximal operator class

        Args:

            gamma (double >= 0): Regularisation parameter
            Psi (Linear operator): Regularisation functional (typically wavelets)

        Raises:

            ValueError: Raised if regularisation parameter is not postitive semi-definite
        """

        if np.any(gamma <= 0):
            raise ValueError("'gamma' must be positive semi-definite")

        self.gamma = gamma
        self.beta = 1.0

        if Psi is None:
            self.Psi = linear_operators.identity()
        else:
            self.Psi = Psi

    def prox(self, x, tau):
        """Evaluates the l1-norm prox of x

        Args:

            x (np.ndarray): Array to evaluate proximal projection of
            tau (double): Custom weighting of l1-norm prox

        Returns:

            l1-norm prox of x
        """
        return np.maximum(0, np.abs(x) - self.gamma * tau) * np.exp(
            complex(0, 1) * np.angle(x)
        )

    def fun(self, x):
        """Evaluates loss of functional term of l1-norm regularisation

        Args:

            x (np.ndarray): Array to evaluate loss of

        Returns:

            l1-norm loss
        """
        return np.abs(self.gamma * x).sum()

    def dir_op(self, x):
        """Evaluates the forward regularisation operator

        Args:

            x (np.ndarray): Array to forward transform

        Returns:

            Forward regularisation operator applied to x
        """
        return self.Psi.dir_op(x)

    def adj_op(self, x):
        """Evaluates the forward adjoint regularisation operator

        Args:

            x (np.ndarray): Array to adjoint transform

        Returns:

            Forward adjoint regularisation operator applied to x
        """
        return self.Psi.adj_op(x)


class l2_square_norm:
    """This class computes the proximity operator of the l2 squared.

                        f(x) = 0.5/sigma^2 * ||Psi x||_2^2

    When the input 'x' is an array. 0.5/sigma^2 is a regularization term. Psi is an operator.
    """

    def __init__(self, sigma, Psi=None):
        """Initialises an l2^2-norm proximal operator class

        Args:

            sigma (double >= 0): Regularisation parameter
            Psi (Linear operator): Regularisation functional (typically wavelets)

        Raises:

            ValueError: Raised if sigma is not postitive semi-definite
        """
        if np.any(sigma <= 0):
            raise ValueError("'sigma' must be positive semi-definite")

        self.sigma = sigma
        self.beta = 1.0

        if Psi is None:
            self.Psi = linear_operators.identity()
        else:
            self.Psi = Psi

    def prox(self, x, tau):
        """Evaluates the l2^2-norm prox of x

        Args:

            x (np.ndarray): Array to evaluate proximal projection of
            tau (double): Custom weighting of prox

        Returns:

            l2^2-norm prox of x
        """
        return x / (tau / self.sigma ** 2 + 1.0)

    def fun(self, x):
        """Evaluates loss of functional term of an l2^2-norm term

        Args:

            x (np.ndarray): Array to evaluate loss of

        Returns:

            l2^2-norm loss
        """
        return np.sum(np.abs(x) ** 2 / (2.0 * self.sigma ** 2))

    def dir_op(self, x):
        """Evaluates the forward operator

        Args:

            x (np.ndarray): Array to forward transform

        Returns:

            Forward operator applied to x
        """
        return self.Psi.dir_op(x)

    def adj_op(self, x):
        """Evaluates the forward adjoint operator

        Args:

            x (np.ndarray): Array to adjoint transform

        Returns:

            Forward adjoint operator applied to x
        """
        return self.Psi.adj_op(x)


class positive_prox:
    """This class computes the proximity operator of the indicator function for
    positivity.

                        f(x) = (Re{x} >= 0) ? 0. : infty
    it returns the projection.
    """

    def __init__(self):
        """
        Initialises a positive half-plane proximal operator class
        """
        self.beta = 1.0

    def prox(self, x, tau):
        """Evaluates the positive half-plane projection of x

        Args:

            x (np.ndarray): Array to evaluate proximal projection of

        Returns:

            positive half-plane projection of x
        """
        return np.maximum(0, np.real(x))

    def fun(self, x):
        """Evaluates loss of functional term

        Args:

            x (np.ndarray): Array to evaluate loss of

        Returns:

            0
        """
        return 0.0

    def dir_op(self, x):
        """Evaluates the forward operator

        Args:

            x (np.ndarray): Array to forward transform

        Returns:

            Forward operator applied to x
        """
        return x

    def adj_op(self, x):
        """Evaluates the forward adjoint operator

        Args:

            x (np.ndarray): Array to forward adjoint transform

        Returns:

            Forward adjoint operator applied to x
        """
        return x


class real_prox:
    """This class computes the proximity operator of the indicator function for
    reality.

                        f(x) = (Re{x} == x) ? 0. : infty
    it returns the projection.
    """

    def __init__(self):
        """
        Initialises a real half-plane proximal operator class
        """
        self.beta = 1.0

    def prox(self, x, tau):
        """Evaluates the real half-plane projection of x

        Args:

            x (np.ndarray): Array to evaluate proximal projection of

        Returns:

            real half-plane projection of x
        """
        return np.real(x)

    def fun(self, x):
        """Evaluates loss of functional term

        Args:

            x (np.ndarray): Array to evaluate loss of

        Returns:

            0
        """
        return 0.0

    def dir_op(self, x):
        """Evaluates the forward operator

        Args:

            x (np.ndarray): Array to forward transform

        Returns:

            Forward operator applied to x
        """
        return x

    def adj_op(self, x):
        """Evaluates the forward adjoint operator

        Args:

            x (np.ndarray): Array to forward adjoint transform

        Returns:

            Forward adjoint operator applied to x
        """
        return x


class zero_prox:
    """This class computes the proximity operator of the indicator function for zero.

                        f(x) = (0 == x) ? 0. : infty
    it returns the projection.
    """

    def __init__(self, indices, op, offset=0):
        """Initialises a proximal operator class for the zero indicator function

        Args:

            indicies (list[int]): Indices to apply zero indicator to
            op (Linear operator): Linear operator
        """
        self.beta = 1.0
        self.indices = indices
        self.op = op
        self.offset = offset

    def prox(self, x, gamma):
        """Evaluates the indicator function (a zero-projection)

        Args:

            x (np.ndarray): Array to evaluate proximal projection of

        Returns:

            indicator applied to x
        """
        buff = np.copy(x)
        buff[self.indices] = self.offset
        return buff

    def fun(self, x):
        """Evaluates loss of functional term

        Args:

            x (np.ndarray): Array to evaluate loss of

        Returns:

            0
        """
        return 0.0

    def dir_op(self, x):
        """Evaluates the forward operator

        Args:

            x (np.ndarray): Array to forward transform

        Returns:

            Forward operator applied to x
        """
        return self.op.dir_op(x)

    def adj_op(self, x):
        """Evaluates the forward adjoint operator

        Args:

            x (np.ndarray): Array to forward adjoint transform

        Returns:

            Forward adjoint operator applied to x
        """
        return self.op.adj_op(x)


class poisson_loglike_ball:
    """This class computes the proximity operator of the log of Poisson distribution

                        f(x) = (1^t (x + b) - y^t log(x + b) < epsilon/2.) ? 0. : infty

    When the input 'x' is an array. y is the data vector. Phi is the measurement operator.
    """

    def __init__(self, epsilon, data, background, iters=20, Phi=None):
        """Initialises a proximal operator class for the poisson_ball

        Args:

            epsilon (double > 0): radius of poisson ball
            data (np.ndarray): Data that is centred on the ball
            background (np.ndarray): Background signal dataset
            iters (int): maximum proximal sub-iterations
            Phi (Linear operator): Measurement/Weighting operator

        Raises:

            ValueError: Raised if ball radius is not strictly positive.
        """
        if np.any(epsilon <= 0):
            raise ValueError("'epsilon' must be positive definite")
        self.epsilon = epsilon
        self.data = data
        self.background = background
        self.beta = 1.0
        if Phi is None:
            self.Phi = linear_operators.identity()
        else:
            self.Phi = Phi
        self.loglike = lambda x, mask: np.sum(
            x[mask]
            - self.data[mask]
            - self.data[mask] * np.log(x[mask])
            + self.data[mask] * np.log(self.data[mask])
        )
        # below are functions needed for newtons method to find the root for the prox
        self.f = (
            lambda x, delta, mask: self.loglike(
                np.abs(x - delta + np.sqrt((x - delta) ** 2 + 4 * delta * self.data))
                / 2.0,
                mask,
            )
            - epsilon / 2.0
        )
        self.df = (
            lambda x, delta, mask: np.sum(
                (
                    (delta - x[mask] + 2 * self.data[mask])
                    / np.sqrt((x[mask] - delta) ** 2 + 4 * delta * self.data[mask])
                    - 1
                )
                * (
                    1
                    - 2
                    * self.data[mask]
                    / (
                        x[mask]
                        - delta
                        + np.sqrt((x[mask] - delta) ** 2 + 4 * delta * self.data[mask])
                    )
                )
            )
            / 2.0
        )
        self.iters = iters

    def prox(self, x, gamma):
        """Evaluates the poisson ball proximal projection

        Args:

            x (np.ndarray): Array to evaluate proximal projection of

        Returns:

            Poisson ball proximal projection of x
        """
        x_buff = x + self.background
        mask = np.logical_and(self.data > 0, x_buff > 0)
        xx = self.loglike(x_buff, mask)
        p = x_buff * 0
        if xx <= self.epsilon / 2.0:
            p = x
        else:
            # below we use the prox for h(x + b) is prox_h(x + b) - b
            delta = 0
            for i in range(self.iters):
                delta = delta - self.f(x_buff, delta, mask) / self.df(
                    x_buff, delta, mask
                )
            p[mask] = (
                x_buff[mask]
                - delta
                + np.sqrt((x_buff[mask] - delta) ** 2 + 4 * delta * self.data[mask])
            ) / 2.0 - self.background[mask]
        return p

    def fun(self, x):
        """Evaluates loss of functional term

        Args:

            x (np.ndarray): Array to evaluate loss of

        Returns:

            0
        """
        return 0

    def dir_op(self, x):
        """Evaluates the forward operator

        Args:

            x (np.ndarray): Array to forward transform

        Returns:

            Forward operator applied to x
        """
        return self.Phi.dir_op(x)

    def adj_op(self, x):
        """Evaluates the forward adjoint operator

        Args:

            x (np.ndarray): Array to forward adjoint transform

        Returns:

            Forward adjoint operator applied to x
        """
        return self.Phi.adj_op(x)


class poisson_loglike:
    """This class computes the proximity operator of the log of Poisson distribution

                        f(x) = 1^t (x + b) - y^t log(x + b)

    When the input 'x' is an array. y is the data vector. Phi is the measurement operator.
    """

    def __init__(self, data, background, Phi=None):
        """Initialises a proximal operator class for the log poisson distribution

        Args:

            data (np.ndarray): Data that is centred on the ball
            background (np.ndarray): Background signal dataset
            Phi (Linear operator): Measurement/Weighting operator
        """

        self.data = data
        self.background = background
        self.beta = 1.0
        if Phi is None:
            self.Phi = linear_operators.identity()
        else:
            self.Phi = Phi

    def prox(self, x, gamma):
        """Evaluates the proximal projection of the log-poisson distribution

        Args:

            x (np.ndarray): Array to evaluate proximal projection of
            gamma (double >= 0): regularisation parameter

        Returns:

            Log-Poisson prox of x
        """
        return (
            x
            + self.background
            - gamma
            + np.sqrt((x + self.background - gamma) ** 2 + 4 * gamma * self.data)
        ) / 2.0 - self.background

    def fun(self, x):
        """Evaluates loss of functional term

        Args:

            x (np.ndarray): Array to evaluate loss of

        Returns:

            Log-Poisson loss of x
        """
        return np.sum(
            x - self.data - self.data * np.log(x) + self.data * np.log(self.data)
        )

    def dir_op(self, x):
        """Evaluates the forward operator

        Args:

            x (np.ndarray): Array to forward transform

        Returns:

            Forward operator applied to x
        """
        return self.Phi.dir_op(x)

    def adj_op(self, x):
        """Evaluates the forward adjoint operator

        Args:

            x (np.ndarray): Array to forward adjoint transform

        Returns:

            Forward adjoint operator applied to x
        """
        return self.Phi.adj_op(x)


class l21_norm:
    """This class computes the proximity operator of the l2 ball.

                        f(x) = (||Phi x - y|| < epsilon) ? 0. : infty

    When the input 'x' is an array. y is the data vector. Phi is the measurement operator.
    """

    def __init__(self, tau, l2axis=0, Phi=None):
        """Initialises a proximal operator class for the l2-ball

        Args:

            tau (double): Scaling factor
            l2axis (int): Axis for l2-ball
            Phi (Linear operator): Measurement/Weighting operator

        Raises:

            ValueError: Raised if tau is not strictly positive.
        """

        if np.any(tau <= 0):
            raise ValueError("'tau' must be positive")
        self.tau = tau
        self.l2axis = l2axis
        self.beta = 1.0
        if Phi is None:
            self.Phi = linear_operators.identity()
        else:
            self.Phi = Phi

    def prox(self, x, gamma):
        """Evaluates the proximal projection of the l2-ball

        Args:

            x (np.ndarray): Array to evaluate proximal projection of
            gamma (double >= 0): regularisation parameter

        Returns:

            Projection of x onto the l2-ball
        """
        xx = np.expand_dims(
            np.sqrt(np.sum(np.square(np.abs(x)), axis=self.l2axis)), self.l2axis
        )
        return x * (1 - self.tau * gamma / np.maximum(xx, self.tau * gamma))

    def fun(self, x):
        """Evaluates loss of functional term

        Args:

            x (np.ndarray): Array to evaluate loss of

        Returns:

            0
        """
        return 0

    def dir_op(self, x):
        """Evaluates the forward operator

        Args:

            x (np.ndarray): Array to forward transform

        Returns:

            Forward operator applied to x
        """
        return self.Phi.dir_op(x)

    def adj_op(self, x):
        """Evaluates the forward adjoint operator

        Args:

            x (np.ndarray): Array to forward adjoint transform

        Returns:

            Forward adjoint operator applied to x
        """
        return self.Phi.adj_op(x)


class translate_prox:
    """
    This class wraps an abstract proximal operator with an arbitrary translation

    Effectively this wraps a given prox class with the fundamental translation
    properties of proximal operators.
    """

    def __init__(self, input_prox, z):
        """Initialises a proximal operator wrapper class for an arbitrary translation

        Args:

            input_prox (class): Proximal class to wrap
            z (np.ndarray): Array to translate
        """
        self.z = input_prox.dir_op(z)
        self.input_prox = input_prox
        self.beta = input_prox.beta

    def prox(self, x, gamma):
        """Evaluates an arbitrarily translated proximal projection

        Args:

            x (np.ndarray): Array to evaluate the translated proximal projection of
            gamma (double >= 0): regularisation parameter

        Returns:

            Translation of input_prox of x
        """
        return self.input_prox.prox(x + self.z, gamma) - self.z

    def fun(self, x):
        """Evaluates translated loss of functional term

        Args:

            x (np.ndarray): Array to evaluate translated loss of

        Returns:

            Translation of loss function of input_prox class
        """
        return self.input_prox.fun(x + self.z)

    def dir_op(self, x):
        """Evaluates the forward operator

        Args:

            x (np.ndarray): Array to forward transform

        Returns:

            Forward operator applied to x
        """
        return self.input_prox.dir_op(x)

    def adj_op(self, x):
        """Evaluates the forward adjoint operator

        Args:

            x (np.ndarray): Array to forward adjoint transform

        Returns:

            Forward adjoint operator applied to x
        """
        return self.input_prox.adj_op(x)
