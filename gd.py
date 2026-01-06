def gradient_descent_tmp(grad_f, eta, x0, max_iter):
    x = x0
    for t in range(max_iter):
        x = x - eta(x, t) * grad_f(x)
    return x
