# 11 · Derivatives, Jacobians, the Chain Rule, and Automatic Differentiation

**Section:** S02 · **Type:** Math · **Duration:** ~25 min
**Companion notebook:** `11_derivatives_autodiff.ipynb`

## Learning objectives
- Compute a Jacobian analytically, numerically, and with reverse-mode autodiff.
- Explain backpropagation as the chain rule applied over a computation graph.
- Choose a finite-difference step size and know why it has an optimum.

---

## 1. Gradients and Jacobians
For $f:\mathbb{R}^n\to\mathbb{R}$, the gradient $\nabla f$ points uphill. For $f:\mathbb{R}^n\to\mathbb{R}^m$ the **Jacobian** collects all partial derivatives:
$$J_{ij} = \frac{\partial f_i}{\partial x_j},\qquad f(x+\delta) \approx f(x) + J\,\delta$$
2-link arm example: $x = l_1\cos q_1 + l_2\cos(q_1+q_2)$, $z = l_1\sin q_1 + l_2 \sin(q_1+q_2)$:
$$J = \begin{bmatrix} -l_1 s_1 - l_2 s_{12} & -l_2 s_{12}\\ l_1 c_1 + l_2 c_{12} & l_2 c_{12}\end{bmatrix}$$

## 2. The chain rule rules robotics
$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}\,\frac{\partial y}{\partial x}$$
- Kinematics: $\dot x = J(q)\dot q$ is the chain rule in time.
- Learning: the loss on an action flows back through the action head, the transformer, and the vision encoder.
- Differentiable simulation (MJX) chains the gradient through the physics itself.

## 3. Reverse-mode autodiff (backprop) in 40 lines
Each operation records its inputs and a local derivative rule. After the forward pass:
1. Topologically sort the graph.
2. Set $\partial L/\partial L = 1$.
3. Walk backwards; each node adds `local_grad × out.grad` to its parents (accumulating with `+=` because a value can feed several nodes).

Cost: one backward pass gives the gradient with respect to **all** inputs. That's why training billion-parameter VLAs is possible. Forward mode would need one pass per input.

## 4. Finite differences
$$\frac{\partial f}{\partial x_j} \approx \frac{f(x + h e_j) - f(x - h e_j)}{2h}$$
- Truncation error $O(h^2)$ shrinks as $h$ decreases; round-off error $O(\epsilon_{mach}/h)$ grows.
- The optimum for central differences is around $h \sim \epsilon_{mach}^{1/3} \approx 10^{-5}$. The notebook's U-shaped error curve shows this.
- Use it to **unit-test** analytic Jacobians: the notebook checks MuJoCo's `mj_jacBody` to $10^{-10}$.

## 5. MuJoCo's derivatives
- `mj_jac`, `mj_jacBody`, `mj_jacSite`: analytic kinematic Jacobians (used by IK, operational-space control).
- `mjd_transitionFD`: finite-difference derivatives of the full step $x_{t+1} = f(x_t,u_t)$, used by iLQR/MPC.
- MJX: JAX autodiff through the simulator.

## Key takeaways
- The Jacobian is the local linear map, and every robot velocity relation uses it.
- Backprop = chain rule over a graph, one backward pass for all parameters.
- Finite differences are for testing, and $h \approx 10^{-5}$–$10^{-6}$ balances the two error sources.

## References
- Baydin et al. 2018, *Automatic Differentiation in Machine Learning: a Survey*
- Karpathy, *micrograd*
