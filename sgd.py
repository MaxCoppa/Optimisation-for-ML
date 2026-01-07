import numpy as np


def sgd(x_init, random_indices, grad_i, n_iter=100, step=1.0, store_every=10, args=()):
    """Stochastic gradient descent algorithm."""
    x = x_init.copy()
    x_list = []
    for idx in range(n_iter):
        i = random_indices[idx]

        # Decreasing Step size
        x = x - step * grad_i(i, x, *args) / (np.sqrt(idx + 1))
        # Update metrics after each iteration.
        if idx % store_every == 0:
            x_list.append(x.copy())
    return x, x_list


def sgd(x_init, random_indices, grad_i, n_iter=100, step=1.0, store_every=10, args=()):
    """Stochastic gradient descent algorithm."""
    x = x_init.copy()
    x_list = []
    for idx in range(n_iter):
        i = random_indices[idx]

        # Fixed Step Size
        x = x - step * grad_i(i, x, *args)
        # Update metrics after each iteration.
        if idx % store_every == 0:
            x_list.append(x.copy())
    return x, x_list


def sag(
    x_init,
    random_indices,
    n_samples,
    grad_i,
    n_iter=100,
    step=1.0,
    store_every=10,
    args=(),
):
    """Stochastic average gradient algorithm."""
    x = x_init.copy()
    n = n_samples
    d = x.size

    # Old gradients
    gradient_memory = np.zeros((n, d))
    averaged_gradient = np.zeros(d)
    x_list = []
    for idx in range(n_iter):
        i = random_indices[idx]

        new_grad_i = grad_i(i, x, *args)
        averaged_gradient += (1 / n) * (new_grad_i - gradient_memory[i])
        x -= step * averaged_gradient

        gradient_memory[i] = new_grad_i
        # Update metrics after each iteration.
        if idx % store_every == 0:
            x_list.append(x.copy())
    return x, x_list


def svrg(
    x_init,
    random_indices,
    grad,
    grad_i,
    n_samples,
    n_iter=100,
    step=1.0,
    store_every=10,
    args=(),
):
    """Stochastic variance reduction gradient algorithm."""
    x = x_init.copy()
    n = n_samples

    x_old = x.copy()
    x_list = []
    for k in range(
        n_iter // n
    ):  # Ensures n_iter calls to grad_i as in other algorithms above.
        ### TODO
        x = x_old.copy()
        mu = grad(x_old, *args)
        for t in range(0, n):
            idx = n * k + t
            i = random_indices[idx]
            x -= step * (grad_i(i, x, *args) - grad_i(i, x_old, *args) + mu)
            # Update metrics after each iteration.
            if idx % store_every == 0:
                x_list.append(x.copy())

        x_old = x.copy()

    return x, x_list
