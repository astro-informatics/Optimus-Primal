import abc
import numpy as np
import tensorflow as tf


class LearntPrior(metaclass=abc.ABCMeta):
    """Base abstract class for learnt prior models"""

    @abc.abstractmethod
    def __init__(self, tf_model):
        """Constructor setting the hyper-parameters and domains of the model.

        Must be implemented by derived class (currently abstract).

        Args:
            tf_model (KerasTensor): network pretrained as a prior model
        """
        self.model = tf_model

    @abc.abstractmethod
    def prox(self, x, gamma):
        """Evaluates the l2-ball prox of x

        Args:

            x (np.ndarray): Array to evaluate proximal gradient
            gamma (float): weighting of proximal gradient
        """
        return self.model(x)

    @classmethod
    def fun(self, x):
        """Placeholder for loss of functional term

        Args:

            x (np.ndarray): Array to evaluate model loss of

        """
        return 0

    @classmethod
    def dir_op(self, x):
        """Evaluates the forward sensing operator

        Args:

            x (np.ndarray): Array to transform

        Returns:

            Forward sensing operator applied to x
        """
        return x

    @classmethod
    def adj_op(self, x):
        """Evaluates the forward adjoint sensing operator

        Args:

            x (np.ndarray): Array to adjoint transform

        Returns:

            Forward adjoint sensing operator applied to x
        """
        return x


class PnpDenoiser(LearntPrior):
    """This class integrates machine learning operators to PNP algorithms"""

    def __init__(self, tf_model, sigma):
        """Initialises a pre-trained tensorflow model

        Args:

            tf_model (KerasTensor): network trained as a denoising prior
            sigma (float): noise std of observed data
        """

        self.model = tf_model

        # Normalisation specific parameters
        self.maxtmp = 0
        self.mintmp = 0
        self.scale_range = 1.0 + sigma / 2.0
        self.scale_shift = (1 - self.scale_range) / 2.0

    def prox(self, x, gamma=1):
        """Applies a keras model as a backward projection step

        Args:

            x (np.ndarray): Array to execute learnt backward denoising step

        Returns:

            Denoising plug & play model applied to input
        """
        out = x.numpy()
        out = self.__normalise(out)
        out = self.__sigma_correction(out)
        out = self.model(out)
        out = self.__invert_sigma_correction(out)
        return self.__invert_normalise(out)

    def __normalise(self, x):
        """Maps tensor from [a,b] to [0,1]

        Args:

            x (np.ndarray): Array to normalise
        """
        self.maxtmp, self.mintmp = x.max(), x.min()
        return (x - self.mintmp) / (self.maxtmp - self.mintmp)

    def __invert_normalise(self, x):
        """Maps tensor from [0,1] to [a,b]

        Args:

            x (np.ndarray): Array to invert normalise
        """
        return x * (self.maxtmp - self.mintmp) + self.mintmp

    def __sigma_correction(self, x):
        """Corrects normalisation [a,b] onto [0,1] for noise

        Args:

            x (np.ndarray): Array to apply sigma shifting
        """
        return x * self.scale_range + self.scale_shift

    def __invert_sigma_correction(self, x):
        """Invert corrects normalisation [0,1] onto [a,b] for noise

        Args:

            x (np.ndarray): Array to invert sigma shifting
        """
        return (x - self.scale_shift) / self.scale_range
