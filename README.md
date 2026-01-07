# Optimization for Machine Learning

This repository presents implementations of optimization algorithms commonly used in machine learning, together with their theoretical assumptions and convergence guarantees.


## Problem Setting

We consider optimization problems of the form

$$
\min_{x \in \mathcal{C}} F(x),
$$

where $\mathcal{C} \subseteq \mathbb{R}^d$ is a convex set and
$F : \mathbb{R}^d \to \mathbb{R}$.


## Gradient Descent (GD)

**Main idea**

First-order method using the full gradient of the objective.

**Problem**

$$
\min_{x \in \mathcal{C}} F(x)
$$

**Update rule**

$$
x_{k+1} = x_k - \rho \nabla F(x_k).
$$

**Assumptions**

* $F$ differentiable
* $F$ convex or $\mu$-strongly convex
* $\nabla F$ is $L$-Lipschitz

**Convergence**

**Convex, $L$-smooth**
If $\rho \le 1/L$:

$$
F(x_k) - F(x^{ * }) = \mathcal{O}\left(\frac{1}{k}\right),
\qquad
k = \mathcal{O}\left(\frac{1}{\varepsilon}\right).
$$

**$\mu$-strongly convex, $L$-smooth**
With $\rho = 1/L$:

$$
\lVert x_k - x^{ * } \rVert^2
\le
\left(1 - \frac{\mu}{L}\right)^k
\lVert x_0 - x^{ * } \rVert^2,
$$

and

$$
k = \mathcal{O}\left(\log \frac{1}{\varepsilon}\right),
\qquad
\kappa = \frac{L}{\mu}.
$$


## Proximal Gradient Descent (PGD)

**Main idea**

Handle non-smooth terms via proximal operators.

**Problem**

$$
F(x) = f(x) + g(x),
$$

with $f$ smooth and $g$ convex.

**Update rule**

$$
x_{k+1} =\mathrm{prox}_{\rho g}\left(x_k - \rho \nabla f(x_k)\right).
$$

**Assumptions**

* $f$ convex and $L$-smooth
* $g$ convex (possibly non-smooth)

**Convergence**

If $\rho \le 1/L$:

$$
F(x_k) - F(x^{ * }) = \mathcal{O}\left(\frac{1}{k}\right).
$$


## Accelerated Gradient Descent (AGD)

**Main idea**

Use momentum to accelerate first-order methods.

**Problem**

$$
\min_x F(x)
$$

**Update rule** (Nesterov)

$$
\begin{aligned}
y_k &= x_k + \beta_k (x_k - x_{k-1}), \
x_{k+1} &= y_k - \frac{1}{L} \nabla F(y_k).
\end{aligned}
$$

**Assumptions**

* $F$ convex and $L$-smooth

**Convergence**

$$
F(x_k) - F(x^{ * }) = \mathcal{O}\left(\frac{1}{k^2}\right).
$$


## Stochastic Gradient Descent (SGD)

**Main idea**

Approximate gradients using random samples.

**Problem**

$$
F(x) = \frac{1}{n} \sum_{i=1}^n f_i(x).
$$

**Update rule**

$$
x_{k+1} = x_k - \rho_k \nabla f_{i_k}(x_k),
$$

with $i_k \sim \text{Unif}{1,\dots,n}$.

**Assumptions**

* Unbiased stochastic gradients
* Bounded variance
* Smoothness of $f_i$

**Convergence**

**Fixed step size, $\mu$-strongly convex**

$$
\mathbb{E}\lVert x_k - x^{ * } \rVert^2 = \mathcal{O}\left((1 - \rho \mu)^k\right),
$$

with bias $\mathcal{O}(\rho)$.

**Decreasing step size**
If $\rho_k = \frac{p}{\sqrt{k+1}}$:

$$
\mathbb{E}[F(x_k) - F(x^{ * })] =\mathcal{O}\left(\frac{1}{k}\right).
$$


## Newton’s Method

**Main idea**

Exploit second-order curvature information.

**Problem**

$$
\min_x f(x)
$$

**Update rule**

$$
x_{k+1} = x_k - \nabla^2 f(x_k)^{-1} \nabla f(x_k).
$$

**Assumptions**

* $f \in C^2$
* $\nabla^2 f(x^{ * }) \succ 0$
* $x_0$ close to $x^{ * }$

**Convergence**

* Local quadratic convergence
* One-step convergence for quadratic objectives
* Cost: $\mathcal{O}(d^3)$


## Quasi-Newton Methods

### DFP

**Main idea**
Approximate inverse Hessian using secant conditions.

**Update rule**
Rank-2 update of the inverse Hessian approximation.

**Assumptions**
Smooth objective + line search.

**Convergence**
Local superlinear, cost $\mathcal{O}(d^2)$.


### BFGS

**Main idea**
Numerically stable quasi-Newton method.

**Update rule**
Improved rank-2 update.

**Assumptions**
Smooth objective + Wolfe line search.

**Convergence**
Local superlinear, cost $\mathcal{O}(d^2)$.


### L-BFGS

**Main idea**
Memory-efficient quasi-Newton method.

**Update rule**
BFGS using last $m \ll d$ curvature pairs.

**Convergence** / Cost
Scalable, $\mathcal{O}(md)$ per iteration.


## Conjugate Gradient (CG)

**Main idea**

Solve linear systems using conjugate directions.

**Problem**

$$
Ax = b, \quad A \succ 0.
$$

**Update rule**

Iteratively construct $A$-conjugate search directions.

**Convergence**

Exact solution in at most $d$ steps (exact arithmetic).


## Coordinate Descent Methods

### Exact Coordinate Descent

**Main idea**
Optimize one coordinate at a time.

**Update rule**
Exact minimization along a single coordinate.

**Assumptions**
$f$ smooth and strictly convex.

**Convergence**
Global convergence to $x^{ * }$.


### Randomized Coordinate Descent

**Update rule**
Randomly select and update one coordinate.

**Convergence**

$$
F(x_k) - F(x^{ * }) =
\begin{cases}
\mathcal{O}\left(\frac{1}{k}\right), & \text{convex}, \
\mathcal{O}\left(\left(1 - \frac{\mu}{nL}\right)^k\right), & \text{strongly convex}.
\end{cases}
$$


## Proximal Coordinate Descent

**Main idea**

Combine coordinate descent with proximal operators.

**Problem**

$$
F(x) = f(x) + g(x),
$$

with separable $g$.

**Update rule**

Apply a proximal update on a single randomly selected coordinate.

**Convergence**

$$
\mathcal{O}\left(\frac{1}{k}\right)
\quad \text{(convex)},
\qquad
\mathcal{O}\left(\left(1 - \frac{\mu}{nL_{\max}}\right)^k\right)
\quad \text{(strongly convex)}.
$$



