import optimusprimal.linear_operators as linear_operators
import numpy as np


class l2_ball:
    """This class computes the proximity operator of the l2 ball.

                        f(x) = (||Phi x - y|| < epsilon) ? 0. : infty

    When the input 'x' is an array. y is the data vector. Phi is the measurement operator.


     INPUTS
    ========
     x         - ND array
     epsilon   - radius of l2-ball
     data      - data that that centres the l2-ball
     Phi       - Measurement/Weighting operator
    """

    def __init__(self, epsilon, data, Phi=None):

        if np.any(epsilon <= 0):
            raise Exception("'epsilon' must be positive")
        self.epsilon = epsilon
        self.data = data
        self.beta = 1.0
        if Phi is None:
            self.Phi = linear_operators.identity()
        else:
            self.Phi = Phi

    def prox(self, x, gamma):
        xx = np.sqrt(np.sum(np.square(np.abs(x - self.data))))
        if xx < self.epsilon:
            p = x
        else:
            p = (x - self.data) * self.epsilon / xx + self.data

        return p

    def fun(self, x):
        return 0

    def dir_op(self, x):
        return self.Phi.dir_op(x)

    def adj_op(self, x):
        return self.Phi.adj_op(x)


class l_inf_ball:
    """This class computes the proximity operator of the l_inf ball.

                        f(x) = (||Phi x - y||_inf < epsilon) ? 0. : infty

    When the input 'x' is an array. y is the data vector. Phi is the measurement operator.


     INPUTS
    ========
     x         - ND array
     epsilon   - radius of l2-ball
     data      - data that that centres the l2-ball
     Phi       - Measurement/Weighting operator
    """

    def __init__(self, epsilon, data, Phi=None):

        if np.any(epsilon <= 0):
            raise Exception("'epsilon' must be positive")
        self.epsilon = epsilon
        self.data = data
        self.beta = 1.0
        if Phi is None:
            self.Phi = linear_operators.identity()
        else:
            self.Phi = Phi

    def prox(self, x, gamma):
        z = x - self.data
        return (
            np.minimum(self.epsilon, np.abs(z)) * np.exp(complex(0, 1) * np.angle(z))
            + self.data
        )

    def fun(self, x):
        return 0

    def dir_op(self, x):
        return self.Phi.dir_op(x)

    def adj_op(self, x):
        return self.Phi.adj_op(x)


class l1_norm:
    """This class computes the proximity operator of the l2 ball.

                        f(x) = ||Psi x||_1 * gamma

    When the input 'x' is an array. gamma is a regularization term. Psi is a sparsity operator.


     INPUTS
    ========
     x         - ND array
     gamma     - regularization parameter
     Phi       - Measurement/Weighting operator
    """

    def __init__(self, gamma, Psi=None):

        if np.any(gamma <= 0):
            raise Exception("'gamma' must be positive")

        self.gamma = gamma
        self.beta = 1.0

        if Psi is None:
            self.Psi = linear_operators.identity()
        else:
            self.Psi = Psi

    def prox(self, x, tau):
        return np.maximum(0, np.abs(x) - self.gamma * tau) * np.exp(
            complex(0, 1) * np.angle(x)
        )

    def fun(self, x):
        return np.abs(self.gamma * x).sum()

    def dir_op(self, x):
        return self.Psi.dir_op(x)

    def adj_op(self, x):
        return self.Psi.adj_op(x)


class l2_square_norm:
    """This class computes the proximity operator of the l2 squared.

                        f(x) = 0.5/sigma^2 * ||Psi x||_2^2

    When the input 'x' is an array. 0.5/sigma^2 is a regularization term. Psi is an operator.


     INPUTS
    ========
     x         - ND array
     sigma     - regularization parameter
     Phi       - Measurement/Weighting operator
    """

    def __init__(self, sigma, Psi=None):

        if np.any(sigma <= 0):
            raise Exception("'gamma' must be positive")

        self.sigma = sigma
        self.beta = 1.0

        if Psi is None:
            self.Psi = linear_operators.identity()
        else:
            self.Psi = Psi

    def prox(self, x, tau):
        return x / (tau / self.sigma ** 2 + 1.0)

    def fun(self, x):
        return np.sum(np.abs(x) ** 2 / (2.0 * self.sigma ** 2))

    def dir_op(self, x):
        return self.Psi.dir_op(x)

    def adj_op(self, x):
        return self.Psi.adj_op(x)


class positive_prox:
    """This class computes the proximity operator of the indicator function for
    positivity.

                        f(x) = (Re{x} >= 0) ? 0. : infty
    it returns the projection.
    """

    def __init__(self):
        self.beta = 1.0

    def prox(self, x, tau):
        return np.maximum(0, np.real(x))

    def fun(self, x):
        return 0.0

    def dir_op(self, x):
        return x

    def adj_op(self, x):
        return x


class real_prox:
    """This class computes the proximity operator of the indicator function for
    reality.

                        f(x) = (Re{x} == x) ? 0. : infty
    it returns the projection.
    """

    def __init__(self):
        self.beta = 1.0

    def prox(self, x, tau):
        return np.real(x)

    def fun(self, x):
        return 0.0

    def dir_op(self, x):
        return x

    def adj_op(self, x):
        return x


class zero_prox:
    """This class computes the proximity operator of the indicator function for zero.

                        f(x) = (0 == x) ? 0. : infty
    it returns the projection.
    """

    def __init__(self, indices, op, offset=0):
        self.beta = 1.0
        self.indices = indices
        self.op = op
        self.offset = offset

    def prox(self, x, tau):
        buff = np.copy(x)
        buff[self.indices] = self.offset
        return buff

    def fun(self, x):
        return 0.0

    def dir_op(self, x):
        return self.op.dir_op(x)

    def adj_op(self, x):
        return self.op.adj_op(x)


class poisson_loglike_ball:
    """This class computes the proximity operator of the log of Poisson distribution

                        f(x) = (1^t (x + b) - y^t log(x + b) < epsilon/2.) ? 0. : infty

    When the input 'x' is an array. y is the data vector. Phi is the measurement operator.


     INPUTS
    ========
     x         - ND array
     background - background signal
     epsilon   - radius of the ball
     data      - data that that centres the ball
     Phi       - Measurement/Weighting operator
    """

    def __init__(self, epsilon, data, background, iters=20, Phi=None):
        if np.any(epsilon <= 0):
            raise Exception("'epsilon' must be positive")
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
        return 0

    def dir_op(self, x):
        return self.Phi.dir_op(x)

    def adj_op(self, x):
        return self.Phi.adj_op(x)


class poisson_loglike:
    """This class computes the proximity operator of the log of Poisson distribution

                        f(x) = 1^t (x + b) - y^t log(x + b)

    When the input 'x' is an array. y is the data vector. Phi is the measurement operator.


     INPUTS
    ========
     x         - ND array
     epsilon   - radius of the ball
     data      - data that that centres the ball
     Phi       - Measurement/Weighting operator
    """

    def __init__(self, data, background, Phi=None):

        self.data = data
        self.background = background
        self.beta = 1.0
        if Phi is None:
            self.Phi = linear_operators.identity()
        else:
            self.Phi = Phi

    def prox(self, x, gamma):
        return (
            x
            + self.background
            - gamma
            + np.sqrt((x + self.background - gamma) ** 2 + 4 * gamma * self.data)
        ) / 2.0 - self.background

    def fun(self, x):
        return np.sum(
            x - self.data - self.data * np.log(x) + self.data * np.log(self.data)
        )

    def dir_op(self, x):
        return self.Phi.dir_op(x)

    def adj_op(self, x):
        return self.Phi.adj_op(x)


class l21_norm:
    """This class computes the proximity operator of the l2 ball.

                        f(x) = (||Phi x - y|| < epsilon) ? 0. : infty

    When the input 'x' is an array. y is the data vector. Phi is the measurement operator.


     INPUTS
    ========
     x         - ND array
     epsilon   - radius of l2-ball
     data      - data that that centres the l2-ball
     Phi       - Measurement/Weighting operator
    """

    def __init__(self, tau, l2axis=0, Phi=None):

        if np.any(tau <= 0):
            raise Exception("'tau' must be positive")
        self.tau = tau
        self.l2axis = l2axis
        self.beta = 1.0
        if Phi is None:
            self.Phi = linear_operators.identity()
        else:
            self.Phi = Phi

    def prox(self, x, gamma):
        xx = np.expand_dims(
            np.sqrt(np.sum(np.square(np.abs(x)), axis=self.l2axis)), self.l2axis
        )
        return x * (1 - self.tau * gamma / np.maximum(xx, self.tau * gamma))

    def fun(self, x):
        return 0

    def dir_op(self, x):
        return self.Phi.dir_op(x)

    def adj_op(self, x):
        return self.Phi.adj_op(x)


class translate_prox:
    def __init__(self, input_prox, z):
        self.z = input_prox.dir_op(z)
        self.input_prox = input_prox
        self.beta = input_prox.beta

    def prox(self, x, gamma):
        return self.input_prox.prox(x + self.z, gamma) - self.z

    def fun(self, x):
        return self.input_prox.fun(x + self.z)

    def dir_op(self, x):
        return self.input_prox.dir_op(x)

    def adj_op(self, x):
        return self.input_prox.adj_op(x)
