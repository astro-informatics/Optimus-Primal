import optimusprimal.Empty as Empty
import logging
import numpy as np
import time

logger = logging.getLogger("Optimus Primal")


def FBPD(x_init, options=None, g=None, f=None, h=None, p=None, r=None, viewer=None):
    """Evaluates the Primal dual forward backward optimization

    Args:

        x_init (np.ndarray): First estimate solution
        options (dict): Python dictionary of optimisation configuration parameters
        g (Grad Class): Unconstrained data-fidelity class
        f (Prox Class): Constrained data-fidelity class
        h (Prox Class): Proximal regularisation constraint
        p (Prox Class): Positivity constraint
        r (Prox Class): Reality constraint
        viewer (function): Plotting function for real-time viewing (must accept: x, iteration)
    """
    if f is None:
        f = Empty.EmptyProx()
    if g is None:
        g = Empty.EmptyGrad()
    if h is None:
        h = Empty.EmptyProx()
    if p is None:
        p = Empty.EmptyProx()
    if r is None:
        r = Empty.EmptyProx()
    x = x_init
    y = h.dir_op(x) * 0.0
    z = p.dir_op(x) * 0
    w = r.dir_op(x) * 0
    return FBPD_warm_start(x_init, y, z, w, options, g, f, h, p, r, viewer)


def FBPD_warm_start(
    x_init, y, z, w, options=None, g=None, f=None, h=None, p=None, r=None, viewer=None
):
    """Evaluates the Primal dual forward backward optimization with warm-start

    Args:

        x_init (np.ndarray): First estimate solution
        y (np.ndarray): First simulation from `h class'
        z (np.ndarray): First simulation from `p class'
        w (np.ndarray): First simulation from `r class'
        options (dict): Python dictionary of optimisation configuration parameters
        g (Grad Class): Unconstrained data-fidelity class
        f (Prox Class): Constrained data-fidelity class
        h (Prox Class): Proximal regularisation constraint
        p (Prox Class): Positivity constraint
        r (Prox Class): Reality constraint
        viewer (function): Plotting function for real-time viewing (must accept: x, iteration)
    """
    # default inputs
    if f is None:
        f = Empty.EmptyProx()
    if g is None:
        g = Empty.EmptyGrad()
    if h is None:
        h = Empty.EmptyProx()
    if p is None:
        p = Empty.EmptyProx()
    if r is None:
        r = Empty.EmptyProx()
    if options is None:
        options = {"tol": 1e-4, "iter": 500, "update_iter": 100, "record_iters": False}

    # checking minimum requrements for inputs
    assert hasattr(f, "prox")
    assert hasattr(h, "prox")
    assert hasattr(h, "dir_op")
    assert hasattr(h, "adj_op")
    assert hasattr(p, "prox")
    assert hasattr(p, "dir_op")
    assert hasattr(p, "adj_op")
    assert hasattr(g, "grad")

    # algorithmic parameters
    tol = options["tol"]
    max_iter = options["iter"]
    update_iter = options["update_iter"]
    record_iters = options["record_iters"]
    # step-sizes
    tau = 1 / (g.beta + 2)
    sigmah = (1 / tau - g.beta / 2) / (h.beta + p.beta + r.beta)
    sigmap = (1 / tau - g.beta / 2) / (h.beta + p.beta + r.beta)
    sigmar = (1 / tau - g.beta / 2) / (h.beta + p.beta + r.beta)
    # initialization
    x = np.copy(x_init)

    logger.info("Running Forward Backward Primal Dual")
    timing = np.zeros(max_iter)
    criter = np.zeros(max_iter)

    # algorithm loop
    for it in range(0, max_iter):

        t = time.time()
        # primal forward-backward step
        x_old = np.copy(x)
        x = x - tau * (g.grad(x) + h.adj_op(y) + p.adj_op(z) + r.adj_op(w))
        x = f.prox(x, tau)
        # dual forward-backward step
        y = y + sigmah * h.dir_op(2 * x - x_old)
        y = y - sigmah * h.prox(y / sigmah, 1.0 / sigmah)

        z = z + sigmap * p.dir_op(2 * x - x_old)
        z = z - sigmap * p.prox(z / sigmap, 1.0 / sigmap)

        w = w + sigmar * r.dir_op(2 * x - x_old)
        w = w - sigmar * r.prox(w / sigmar, 1.0 / sigmar)
        # time and criterion
        if record_iters:
            timing[it] = time.time() - t
            criter[it] = (
                f.fun(x)
                + g.fun(x)
                + h.fun(h.dir_op(x))
                + p.fun(p.dir_op(x))
                + r.fun(r.dir_op(x))
            )

        if np.allclose(x, 0):
            x = x_old
            logger.info("[Primal Dual] converged to 0 in %d iterations", it)
            break
        # stopping rule
        if np.linalg.norm(x - x_old) < tol * np.linalg.norm(x_old) and it > 10:
            logger.info("[Primal Dual] converged in %d iterations", it)
            break
        if update_iter >= 0:
            if it % update_iter == 0:
                logger.info(
                    "[Primal Dual] %d out of %d iterations, tol = %f",
                    it,
                    max_iter,
                    np.linalg.norm(x - x_old) / np.linalg.norm(x_old),
                )
                if viewer is not None:
                    viewer(x, it)
        logger.debug(
            "[Primal Dual] %d out of %d iterations, tol = %f",
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
        "z": z,
        "y": y,
        "w": w,
    }
    return solution, diagnostics
