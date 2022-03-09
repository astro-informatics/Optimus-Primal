import optimusprimal.Empty as Empty
import logging
import numpy as np
import time

logger = logging.getLogger("Optimus Primal")


def FB(x_init, options=None, g=None, f=None, h=None, alpha=1, tau=1, viewer=None):
    """Evaluates the base forward backward optimisation

    Note that currently this only supports real positive semi-definite
    fields.

    Args:

        x_init (np.ndarray): First estimate solution
        options (dict): Python dictionary of optimisation configuration parameters
        g (Grad Class): Unconstrained data-fidelity class
        f (Prox Class): Reality constraint
        h (Prox/AI Class): Proximal or Learnt regularisation constraint
        alpha (float): regularisation paremeter / step-size.
        tau (float): custom weighting of proximal operator
        viewer (function): Plotting function for real-time viewing (must accept: x, iteration)
    """
    if f is None:
        f = Empty.EmptyProx()
    if g is None:
        g = Empty.EmptyGrad()
    if h is None:
        h = Empty.EmptyProx()

    x = x_init

    if options is None:
        options = {"tol": 1e-4, "iter": 500, "update_iter": 100, "record_iters": False}

    # algorithmic parameters
    tol = options["tol"]
    max_iter = options["iter"]
    update_iter = options["update_iter"]
    record_iters = options["record_iters"]

    # initialization
    x = np.copy(x_init)

    logger.info("Running Base Forward Backward")
    timing = np.zeros(max_iter)
    criter = np.zeros(max_iter)

    # algorithm loop
    for it in range(0, max_iter):

        t = time.time()
        # forward step
        x_old = np.copy(x)
        x = x - alpha * g.grad(x)
        x = f.prox(x, tau)

        # backward step
        u = h.dir_op(x)
        x = x + h.adj_op(h.prox(u, tau) - u)

        # time and criterion
        if record_iters:
            timing[it] = time.time() - t
            criter[it] = f.fun(x) + g.fun(x) + h.fun(h.dir_op(x))

        if np.allclose(x, 0):
            x = x_old
            logger.info("[Forward Backward] converged to 0 in %d iterations", it)
            break
        # stopping rule
        if np.linalg.norm(x - x_old) < tol * np.linalg.norm(x_old) and it > 10:
            logger.info("[Forward Backward] converged in %d iterations", it)
            break
        if update_iter >= 0:
            if it % update_iter == 0:
                logger.info(
                    "[Forward Backward] %d out of %d iterations, tol = %f",
                    it,
                    max_iter,
                    np.linalg.norm(x - x_old) / np.linalg.norm(x_old),
                )
                if viewer is not None:
                    viewer(x, it)
        logger.debug(
            "[Forward Backward] %d out of %d iterations, tol = %f",
            it,
            max_iter,
            np.linalg.norm(x - x_old) / np.linalg.norm(x_old),
        )

    criter = criter[0 : it + 1]
    timing = np.cumsum(timing[0 : it + 1])
    solution = x
    diagnostics = {
        "max_iter": it,
        "times": timing,
        "Obj_vals": criter,
        "x": x,
    }
    return solution, diagnostics
