# Optimisation-for-ML
This repo presents the implementation of different algorithms to solve different optimisation problem.

obj : min F(x) x in C 

Gradient Descent algorithm : 
If F convex, differentiable and L-smooth :
GD converges to x^* with a fixed step <= 1/L in O(1/k) precision eps O(1/eps)

If F mu-strongly convex, differentiable and L-smooth :
GD converges to x^* with a fixed step = 1/L in O(e^-K/k) where K = L/mu >= 1, precision eps O(log(1/eps))

Proximal Gradient algorithm : 
If F = f + g , f L-smooth function and a convex g the PGD with step size ρ ≤ 1/L to a minimum of F with the  O(1/k) speed

AGD : 
If F convex, differentiable and L-smooth :
1GD converges to x^* with a fixed step = 1/L in O(1/k^2) (the idea of AGD is to remember previous gradients)


SGD algorithm*: $F = \frac{1}{n}\sum_i f_i$

- Fix step size :

- (F) $\mu$-strongly convex + EBSG, $\rho \le \frac{1}{\mu}$:
  Exponential linear convergence $((1-\rho\mu)^k)$ to a neighborhood of $(x^*)$; bias (O(\rho)) proportional to step size.

- (F) (\mu)-strongly convex + (f_i) (L_i)-smooth, $L_{\max}=\max_i L_i$,
  $\rho \le \frac{1}{2L_{\max}} $:
  Exponential (linear) convergence; bias $(O(\rho))$ depending only on gradient noise at the solution.
  
Not fixed step size :

- (F) (\mu)-strongly convex + (f_i) (L_i)-smooth,
  $ \rho^{(k)} \le \frac{p}{\sqrt{k+1}},; p < \frac{1}{4L_{\max}} $:
  Sublinear convergence $(O(1/k))$; exact convergence, less noise for large (k); cost (O(d)) per iteration.
