import numpy as np


def cgd(x_init, grad_i, step, n_iter=100, store_every=1, grad_args=()):

    x = x_init.copy()
    n_features = len(x)
    x_list = []

    for k in range(n_iter):

        # pick one coordinate uniformly
        j = np.random.randint(n_features)

        # partial derivative w.r.t. coordinate j
        g_j = grad_i(j, x, *grad_args)

        # coordinate gradient step
        x[j] -= step[j] * g_j

        if k % store_every == 0:
            x_list.append(x.copy())

    return x, x_list


def pcd(
    x_init, grad_i, prox_i, step, n_iter=100, store_every=1, grad_args=(), prox_args=()
):

    x = x_init.copy()
    n_features = len(x)
    x_list = []

    for k in range(n_iter):

        # random coordinate
        j = np.random.randint(n_features)

        # partial gradient
        g_j = grad_i(j, x, *grad_args)

        # proximal coordinate update
        x[j] = prox_i(x[j] - step[j] * g_j, step[j], *prox_args)

        if k % store_every == 0:
            x_list.append(x.copy())

    return x, x_list
