import numpy as np


def pgd(
    x_init, grad, prox, step, n_iter=100, store_every=1, grad_args=(), prox_args=()
):
    """Proximal gradient descent algorithm.

    Parameters
    ----------
    x_init : array, shape (n_parameters,)
        Parameters of the optimization problem.
    grad : callable
        The gradient of the smooth data fitting term.
    prox : callable
        The proximal operator of the regularization term.
    step : float
        The size of the gradient step done on the smooth term.
    n_iter : int
        The number of iterations.
    store_every : int
        At which frequency should the current iterated be remembered.
    grad_args : tuple
        Parameters to pass to grad.
    prox_args : tuple
        Parameters to pass to prox.

    Returns
    -------
    x : array, shape (n_parameters,)
        The estimated parameters.
    x_list : list
        The list if x values along the iterations.
    """
    x = x_init.copy()
    x_list = []
    for i in range(n_iter):

        d = -grad(x, *grad_args)
        x = prox(x + step * d, step, *prox_args)

        if i % store_every == 0:
            x_list.append(x.copy())
    return x, x_list


def apgd(
    x_init, grad, prox, step, n_iter=100, store_every=1, grad_args=(), prox_args=()
):
    """Accelerated proximal gradient descent algorithm.

    Parameters
    ----------
    x_init : array, shape (n_parameters,)
        Parameters of the optimization problem.
    grad : callable
        The gradient of the smooth data fitting term.
    prox : callable
        The proximal operator of the regularization term.
    step : float
        The size of the gradient step done on the smooth term.
    n_iter : int
        The number of iterations.
    store_every : int
        At which frequency should the current iterated be remembered.
    grad_args : tuple
        Parameters to pass to grad.
    prox_args : tuple
        Parameters to pass to prox.

    Returns
    -------
    x : array, shape (n_parameters,)
        The estimated parameters.
    x_list : list
        The list if x values along the iterations.
    """
    x = x_init.copy()
    y = x_init.copy()
    t = 1.0
    x_list = []
    for i in range(n_iter):

        t = (1 + np.sqrt(1 + 4 * t**2)) / 2
        d = -grad(x, *grad_args)
        x_new = prox(x + step * d, step, *prox_args)
        x = x_new + ((t - 1) * (x_new - x)) / (t)

        if i % store_every == 0:
            x_list.append(x.copy())

    return x, x_list
