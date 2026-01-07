# Optimization for Machine Learning

This repository presents implementations of optimization algorithms commonly used in machine learning, together with their theoretical assumptions and convergence guarantees.

## Problem Setting

We consider optimization problems of the form

$$
\min_{x \in \mathcal{C}} F(x),
$$

where $\mathcal{C} \subseteq \mathbb{R}^d$ is a convex set and $F : \mathbb{R}^d \to \mathbb{R}$.

## Gradient Descent (GD)

### Main idea

First-order method using the full gradient to minimize a smooth objective.

### Update rule

$$
x_{k+1} = x_k - \rho \nabla F(x_k).
$$

### Assumptions and convergence

#### Convex, $L$-smooth

If $F$ is convex, differentiable, and $L$-smooth, with step size $\rho \le 1/L$:

$$
F(x_k) - F(x^{ * }) = \mathcal{O}\left(\frac{1}{k}\right),
\qquad
k = \mathcal{O}\left(\frac{1}{\varepsilon}\right).
$$

#### $\mu$-strongly convex, $L$-smooth

If $F$ is $\mu$-strongly convex and $L$-smooth, with $\rho = 1/L$:

$$
\lVert x_k - x^{ * } \rVert^2
\le
\left(1 - \frac{\mu}{L}\right)^k
\lVert x_0 - x^{ * } \rVert^2.
$$

Linear (exponential) convergence, and

$$
k = \mathcal{O}\left(\log \frac{1}{\varepsilon}\right),
\qquad
\kappa = \frac{L}{\mu}.
$$

## Proximal Gradient Descent (PGD)

### Problem

$$
F(x) = f(x) + g(x),
$$

where $f$ is convex and $L$-smooth, and $g$ is convex (possibly non-smooth).

### Update rule

$$
x_{k+1} = \mathrm{prox}_{\rho g}\left(x_k - \rho \nabla f(x_k)\right).
$$

### Convergence

If $\rho \le 1/L$:

$$
F(x_k) - F(x^{ * }) = \mathcal{O}\left(\frac{1}{k}\right).
$$

## Accelerated Gradient Descent (AGD)

### Main idea

Momentum-based method combining current and past gradients.

### Update rule

(Equivalent formulations exist; e.g. Nesterov’s scheme)

$$
\begin{aligned}
y_k &= x_k + \beta_k (x_k - x_{k-1}), \
x_{k+1} &= y_k - \frac{1}{L} \nabla F(y_k).
\end{aligned}
$$

### Assumptions and convergence

If $F$ is convex and $L$-smooth:

$$
F(x_k) - F(x^{ * }) = \mathcal{O}\left(\frac{1}{k^2}\right),
$$

which is optimal among first-order methods.

## Stochastic Gradient Descent (SGD)

### Problem

$$
F(x) = \frac{1}{n} \sum_{i=1}^n f_i(x).
$$

### Update rule

$$
x_{k+1} = x_k - \rho_k \nabla f_{i_k}(x_k),
$$

where $i_k$ is sampled uniformly from ${1,\dots,n}$.

### Fixed step size

#### Strongly convex + bounded noise

If $F$ is $\mu$-strongly convex and gradients are unbiased with bounded variance, $\rho \le 1/\mu$:

$$
\mathbb{E}\lVert x_k - x^{ * } \rVert^2 =\mathcal{O}\left((1 - \rho \mu)^k\right),
$$

with a bias of order $\mathcal{O}(\rho)$.

#### Smooth component functions

If each $f_i$ is $L_i$-smooth and $\rho \le 1/(2L_{\max})$, convergence is linear up to a noise-dependent neighborhood.

### Decreasing step size

If $\rho_k = \frac{p}{\sqrt{k+1}}$, with $p < 1/(4L_{\max})$:

$$
\mathbb{E}(F(x_k) - F(x^{ * })) = \mathcal{O}\left(\frac{1}{k}\right),
$$

with exact convergence and reduced variance.

## Newton’s Method

### Main idea

Second-order method using curvature information.

### Update rule

$$
x_{k+1} = x_k - \nabla^2 f(x_k)^{-1} \nabla f(x_k).
$$

### Assumptions

* $f \in C^2$
* $x^{ * }$ local minimizer
* $\nabla^2 f(x^{ * }) \succ 0$
* $x_0$ sufficiently close to $x^{ * }$

### Properties

* Local quadratic convergence
* One-step convergence for quadratic objectives
* Cost: $\mathcal{O}(d^3)$ per iteration

## Quasi-Newton Methods

### DFP

#### Update rule

Rank-2 update approximating the inverse Hessian and satisfying the secant condition.

#### Properties

* Smooth objective + line search
* Local superlinear convergence
* Cost: $\mathcal{O}(d^2)$

### BFGS

#### Update rule

Improved rank-2 update with better numerical stability.

#### Properties

* Robust with inexact line search
* Local superlinear convergence
* Cost: $\mathcal{O}(d^2)$

### L-BFGS

#### Update rule

Limited-memory BFGS using the last $m \ll d$ curvature pairs.

#### Properties

* Scalable to large dimensions
* No finite-step property
* Cost: $\mathcal{O}(md)$

## Conjugate Gradient (CG)

### Problem

Solve

$$
Ax = b, \quad A \succ 0.
$$

### Update rule

Iterative construction of $A$-conjugate directions using gradient information.

### Properties

* Matrix-free
* Exact solution in at most $d$ steps (exact arithmetic)

## Coordinate Descent Methods

### Exact Coordinate Descent

#### Update rule

At iteration $k$, minimize $f$ exactly with respect to one coordinate.

#### Assumptions

* $f$ continuously differentiable
* strictly convex

#### Convergence

Global convergence to the unique minimizer $x^{ * }$.

### Randomized Coordinate Descent

### Update rule

Select coordinate $i_{k+1}$ uniformly and update only that coordinate.

### Convergence

$$
F(x_k) - F(x^{ * }) =
\begin{cases}
\mathcal{O}\left(\frac{1}{k}\right), & \text{convex}, \
\mathcal{O}\left(\left(1 - \frac{\mu}{nL}\right)^k\right), & \text{strongly convex}.
\end{cases}
$$

## Proximal Coordinate Descent

### Problem

$$
F(x) = f(x) + g(x),
$$
with $f$ smooth and $g$ separable.

### Update rule

Apply a proximal step on a single randomly selected coordinate.

### Convergence

$$
\mathcal{O}\left(\frac{1}{k}\right)
\quad \text{(convex)},
\qquad
\mathcal{O}\left(\left(1 - \frac{\mu}{nL_{\max}}\right)^k\right)
\quad \text{(strongly convex)}.
$$

