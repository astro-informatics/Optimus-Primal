Bespoke linear operators
=================================
In this section we provide a range of pedagogical example applications of 
``optimusprimal`` both on 1D signals (*e.g.* time series datasets) and 2D 
signals (*e.g.* image processing datasets). In each case there are many 
commonalities. The canonical structure of these applications is 

**import/simulate data** :math:`\rightarrow` 
**define a data-fidelity (likelihood)** :math:`\rightarrow` 
**define a regulariser (prior)** :math:`\rightarrow` 
**run the primal-dual solver**.

Custom Measurement operators :math:`\Phi`
-----------------------------------------
For each application we adopt a standard de-noising setting, in which the 
measurement (sensing) operator is simply the identity matrix. In practice 
this operator is customized for a particular application. To do his one 
simply needs to define a linear operator class which has the following 
structure

.. code-block:: python 

    class my_custom_linear_operator:
        """A custom linear operator e.g. a custom measurement operator"""

        def __init__(self, ... ):
            """Initialise the operator with any necessary parameters"""
            self.(...) = (...)
        
        def dir_op(self, x):
            """Forward linear operator 

            Args: 

                x (np.ndarray): vector this operator should be applied to
            
            Returns:

                f(x) (np.ndarray): Forward operator applied to x
            """
            return f(x)
        
        def adj_op(self, x):
            """Adjoint linear operator 

            Args: 

                x (np.ndarray): vector this adjoint operator should be applied to
            
            Returns:

                f^T(x) (np.ndarray): Forward operator applied to x
            """
            return f^T(x)

this function is then passed to your data-fidelity term instead of 
``linear_operator.identity()`` *e.g.*

.. code-block:: python 

    # What was 
    prox_operators.l2_ball( ... , Phi=linear_operator.identity())
    
    # Becomes
    prox_operators.l2_ball( ... , Phi=my_custom_linear_operator)

Custom Wavelet Transform :math:`\Psi`
-------------------------------------
Though PyWavelet has a large `catalogue <https://tinyurl.com/5n7wzpmb>`_ of Euclidean 
wavelets available, one may wish to adopt specific (perhaps custom) wavelets. The only 
requirement of the wavelet operator, beyond functioning correctly outright, is that it 
is written in the linear operator class format above (*i.e.* with a ``dir_op`` and ``adj_op``). 
A custom wavelet transform can then be passed to the regulariser *e.g.* an :math:`\ell_1`-norm 
term by 

.. code-block:: python 

    prox_operators.l1_norm( ... , Psi=Psi)


Real Time Viewing
-----------------
Often it is desirable to view the progress of the solver during iterations to 
evaluate performance (rather than just watching a summary statistic decreasing *e.g.* the loss). 
This is supported in ``optimusprimal`` by passing the solver a viewer function *i.e.* 

.. code-block:: python

    primal_dual.FBPD( ... , viewer=my_custom_viewing_function)

The viewing function can be whatever you want, provided it accepts both 
an array (to view) and an iteration number (to count). An example might be 

.. code-block:: python

    def my_viewer(x, it):
        """A custom function to view solver progress in realtime

        Args:

            x (np.ndarray): Current estimate at iteration it
            it (int): Current iteration number
        
        """

        plt.imshow(x)
        plt.title("Solution at iteration: {}".format(it))
        plt.show()

    

    