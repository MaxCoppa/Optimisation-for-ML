# Optimisation- for- ML
This repo presents the implementation of different algorithms to solve different optimisation problem.

obj : min F(x) x in C 

Gradient Descent algorithm : 
If F convex, differentiable and L- smooth :
GD converges to x^* with a fixed step <= 1/L in O(1/k) precision eps O(1/eps)

If F mu- strongly convex, differentiable and L- smooth :
GD converges to x^* with a fixed step = 1/L in O(e^- K/k) where K = L/mu >= 1, precision eps O(log(1/eps))

Proximal Gradient algorithm : 
If F = f + g , f L- smooth function and a convex g the PGD with step size ρ ≤ 1/L to a minimum of F with the  O(1/k) speed

AGD : 
If F convex, differentiable and L- smooth :
1GD converges to x^* with a fixed step = 1/L in O(1/k^2) (the idea of AGD is to remember previous gradients)


SGD algorithm*: $F = \frac{1}{n}\sum_i f_i$

- Fix step size :

- (F) $\mu$- strongly convex + EBSG, $\rho \le \frac{1}{\mu}$:
  Exponential linear convergence $((1- \rho\mu)^k)$ to a neighborhood of $(x^*)$; bias (O(\rho)) proportional to step size.

- (F) (\mu)- strongly convex + (f_i) (L_i)- smooth, $L_{\max}=\max_i L_i$,
  $\rho \le \frac{1}{2L_{\max}} $:
  Exponential (linear) convergence; bias $(O(\rho))$ depending only on gradient noise at the solution.
  
Not fixed step size :

- (F) (\mu)- strongly convex + (f_i) (L_i)- smooth,
  $ \rho^{(k)} \le \frac{p}{\sqrt{k+1}},; p < \frac{1}{4L_{\max}} $:
  Sublinear convergence $(O(1/k))$; exact convergence, less noise for large (k); cost (O(d)) per iteration.


Newton

- Update: (x_{k+1} = x_k - \nabla^2 f(x_k)^{- 1}\nabla f(x_k))
- Hypotheses (local convergence): (f \in C^2), (x^*) is a (local) minimizer / stationary point, and (\nabla^2 f(x^*)) is invertible (typically (\succ 0)); (x_0) sufficiently close.
- Rate: Local quadratic convergence 
- Quadratic form ((A\succ 0)): Converges in 1 iteration
- Cost: (O(d^3)) per iteration
- Performance: Very fast near optimum, impractical for large (d); often needs line search / Hessian regularization



DFP (Quasi- Newton)

- Idea: Rank- 2 update approximating inverse Hessian + secant condition
- Hypotheses (practical convergence): (f) smooth ((C^2) in practice), line search (e.g., Wolfe); start with (B_0 \succ 0).
- Rate: Superlinear convergence (locally)
- Quadratic form ((A\succ 0) + exact line search): Exact convergence in < (d) iterations
- Cost: (O(d^2)) time and memory



BFGS

- Idea: More stable rank- 2 update than DFP (better with inexact line search)
- Hypotheses (practical convergence): (f) smooth; Wolfe/Goldstein line search; (H_0 \succ 0) (or (B_0 \succ 0)).
- Rate: Superlinear convergence (locally)
- Quadratic form ((A\succ 0) + exact line search): Exact convergence in < (d) iterations
- Cost: (O(d^2)) time and memory



L- BFGS

- Idea: Limited- memory BFGS (store last (m) curvature pairs)
- Hypotheses: (f) smooth; Wolfe line search typically used.
- Rate: Superlinear (in practice)
- Quadratic form: Finite- step (< (d)) property is lost
- Cost: (O(md)), (m \ll d)


Conjugate Gradient descent : 

The conjugate gradient method is an iterative method to solve linear systems with positive definite matrices (A ≻ 0). It only needs to know how to compute Ax (operation can be implicit).

The direction dk depends on all the gradients at previous iterates. 



Coordinate Gradient Descent (CGD)

Exact coordinate gradient descent :

  - (f) continuously differentiable, strictly convex, and admits a minimizer (x^*).

  - At each iteration, exactly minimize (f) w.r.t. one coordinate.

  - Global convergence to the unique minimizer (x^*)


- Randomized coordinate selection:
  If (i_{k+1}) is sampled independently with
  [
  \mathbb{P}(i_{k+1}=i)=\frac{1}{n}, \quad n = \text{number of features},
  ]
  then

  - Expected convergence rate:
    [
    O!\left(\frac{1}{k}\right)
    \quad \text{or} \quad
    O!\left(\left(1- \frac{\mu}{nL}\right)^k\right)
    \text{ if strongly convex}
    ]


- Complexity:

  - Relation:
    [
    \text{CCD} = O!\left(\frac{\text{CGD}}{n}\right)
    ]
  - One full pass over all coordinates ≈ one GD iteration

 Proximal Coordinate Descent

- Problem:
  [
  F(x) = f(x) + g(x)
  ]

- Hypotheses:

  - (f) convex and differentiable
  - (\nabla f) coordinate- wise Lipschitz with constants (L_i)
  - (g) convex and separable across coordinates


- Randomized coordinate selection:
  If (i_{k+1}) is sampled independently with
  [
  \mathbb{P}(i_{k+1}=i)=\frac{1}{n},
  ]
  then

- Convergence rate:

  - Same rate as smooth coordinate descent
    [
    O!\left(\frac{1}{k}\right)
    \quad \text{(convex case)}
    ]
    [
    O!\left(\left(1- \frac{\mu}{nL_{\max}}\right)^k\right)
    \quad \text{(strongly convex case)}
    ]

