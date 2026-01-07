# Optimisation- for- ML
This repo presents the implementation of different algorithms to solve different optimisation problem.

## Problem Setting

We consider optimization problems of the form

$$
\min_{x \in \mathcal{C}} F(x),
$$

where $\mathcal{C} \subseteq \mathbb{R}^d$ is a convex set and $F : \mathbb{R}^d \to \mathbb{R}$.


## Gradient Descent (GD)

### Main idea

Iterative first-order method using the full gradient to minimize a smooth objective.

### Update rule

$$
x_{k+1} = x_k - \rho \nabla F(x_k).
$$

### Assumptions and convergence

#### Convex, $L$-smooth function

If $F$ is convex, differentiable, and $L$-smooth, then for a fixed step size

$$
\rho \le \frac{1}{L},
$$

gradient descent satisfies

$$
F(x_k) - F(x^*) = \mathcal{O}\left(\frac{1}{k}\right),
$$

and to reach precision $\varepsilon$:

$$
k = \mathcal{O}\left(\frac{1}{\varepsilon}\right).
$$

#### $\mu$-strongly convex, $L$-smooth function

If $F$ is $\mu$-strongly convex and $L$-smooth, with step size $\rho = 1/L$, then

$$
|x_k - x^{ * }|^2  \le (1 - \mu/L)^k \lVert x_0 - x^{ * } \rVert^2
$$


i.e. **linear (exponential) convergence**.
To reach precision $\varepsilon$:

$$
k = \mathcal{O}\left(\log \frac{1}{\varepsilon}\right),
$$

with condition number $\kappa = L / \mu \ge 1$.


## Proximal Gradient Descent (PGD)

### Problem

$$
F(x) = f(x) + g(x),
$$

where:

* $f$ is convex and $L$-smooth,
* $g$ is convex (possibly non-smooth).

### Update rule

$$
x_{k+1} = \mathrm{prox}_{\rho g}\left(x_k - \rho \nabla f(x_k)\right).
$$

### Convergence

If $\rho \le 1/L$, then

$$
F(x_k) - F(x^*) = \mathcal{O}\left(\frac{1}{k}\right).
$$


## Accelerated Gradient Descent (AGD)

### Main idea

Use **momentum** by combining current and previous gradients to accelerate convergence.

### Assumptions

$F$ convex, differentiable, and $L$-smooth.

### Convergence

With step size $\rho = 1/L$,

$$
F(x_k) - F(x^*) = \mathcal{O}\left(\frac{1}{k^2}\right),
$$

which is optimal among first-order methods for smooth convex optimization.


## Stochastic Gradient Descent (SGD)

### Problem

$$
F(x) = \frac{1}{n} \sum_{i=1}^n f_i(x).
$$


### Fixed step size

#### Strongly convex case + bounded gradient noise

If:

* $F$ is $\mu$-strongly convex,
* unbiased stochastic gradients with bounded variance,
* step size $\rho \le \frac{1}{\mu}$,

then SGD exhibits **linear convergence to a neighborhood** of $x^*$:

$$
\mathbb{E}|x_k - x^*|^2 = \mathcal{O}\left((1 - \rho \mu)^k\right),
$$

with bias $O(\rho)$.

#### Smooth component functions

If:

* $F$ is $\mu$-strongly convex,
* each $f_i$ is $L_i$-smooth, $L_{\max} = \max_i L_i$,
* $\rho \le \frac{1}{2L_{\max}}$,

then SGD converges linearly up to a bias depending only on the gradient noise at $x^*$.


### Decreasing step size

If:

* $F$ is $\mu$-strongly convex,
* $f_i$ are $L_i$-smooth,
* step size $\rho_k = \frac{p}{\sqrt{k+1}}, \quad p < \frac{1}{4L_{\max}}$,

then:

* exact convergence to $x^*$,
* sublinear rate $\mathbb{E}[F(x_k) - F(x^*)] = \mathcal{O}\left(\frac{1}{k}\right)$,

* reduced noise for large $k$,
* per-iteration cost $O(d)$.


## Newton’s Method

### Update rule

$$
x_{k+1} = x_k - \nabla^2 f(x_k)^{-1} \nabla f(x_k).
$$

### Assumptions (local convergence)

* $f \in C^2$,
* $x^*$ is a local minimizer,
* $\nabla^2 f(x^*) \succ 0$,
* $x_0$ sufficiently close to $x^*$.

### Properties

* **Local quadratic convergence**
* For quadratic functions $(f(x)=\tfrac12 x^\top A x)$ with $A \succ 0$: convergence in **one iteration**
* Cost: $O(d^3)$ per iteration

### Remarks

Extremely fast near the optimum, but impractical for large-scale problems; often combined with line search or regularization.


## Quasi-Newton Methods

### DFP

* Rank-2 update satisfying the secant condition
* Requires smooth objective and line search
* Local **superlinear convergence**
* Exact convergence in fewer than $d$ iterations for quadratic problems
* Cost: $O(d^2)$ time and memory


### BFGS

* More numerically stable than DFP
* Robust with inexact line search (Wolfe conditions)
* Local superlinear convergence
* Cost: $O(d^2)$


### L-BFGS

* Limited-memory version of BFGS
* Stores last $m \ll d$ curvature pairs
* No finite-step property
* Cost: $O(md)$
* Widely used in large-scale ML


## Conjugate Gradient (CG)

### Problem

Solve linear systems

$$
Ax = b, \quad A \succ 0.
$$

### Properties

* Iterative, matrix-free method (only needs matrix-vector products)
* Search directions depend on all previous gradients
* Exact solution in at most $d$ steps (exact arithmetic)


## Coordinate Descent Methods

### Exact Coordinate Gradient Descent

#### Assumptions

* $f$ continuously differentiable
* strictly convex
* admits a unique minimizer $x^*$

#### Method

At each iteration, exactly minimize $f$ with respect to one coordinate.

#### Convergence

Global convergence to $x^*$.


### Randomized Coordinate Descent

If coordinate $i_{k+1}$ is sampled uniformly:

$$
\mathbb{P}(i_{k+1} = i) = \frac{1}{n},
$$

then:

$$
F(x_k) - F(x^*) =
\begin{cases}
\mathcal{O}\left(\frac{1}{k}\right) & \text{convex case}, \
\mathcal{O}\left(\left(1 - \frac{\mu}{nL}\right)^k\right) & \text{strongly convex case}.
\end{cases}
$$

#### Complexity

* One full pass over coordinates ≈ one GD iteration
* $\text{CCD} = O(\text{GD} / n)$


## Proximal Coordinate Descent

### Problem

$$F(x) = f(x) + g(x)$$

where:
* $f$ convex and differentiable
* $\nabla f$ coordinate-wise Lipschitz with constants $L_i$
* $g$ convex and separable

### Convergence

With uniform random coordinate sampling:

$$
\mathcal{O}\left(\frac{1}{k}\right) \quad \text{(convex)}
$$

$$
\mathcal{O}\left(\left(1 - \frac{\mu}{nL_{\max}}\right)^k\right) \quad \text{(strongly convex)}.
$$

