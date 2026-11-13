# 32 · Inverse Kinematics

**Section:** S04 · **Type:** Math + lab · **Duration:** ~30 min
**Companion notebook:** `32_inverse_kinematics.ipynb`

## Learning objectives
- Formulate IK as iterative nonlinear least squares.
- Compare Jacobian transpose, pseudoinverse and damped least squares.
- Use redundancy (null-space) and know why IK sometimes fails.

---

## 1. The problem
Find $q$ such that $FK(q) = T^*$. Unlike FK, IK may have **no solution** (out of reach), **one**, **several** (elbow up/down), or **infinitely many** (redundant 7-DoF arms like the Panda). The notebook finds 5 different joint solutions for the same gripper pose.

## 2. Iterative IK
Linearize around the current guess and step:
$$e = \begin{bmatrix} p^* - p(q) \\ \log(R^* R(q)^\top)\end{bmatrix},\qquad q \leftarrow q + \Delta q$$
| Method | $\Delta q$ | Behavior |
|---|---|---|
| Jacobian transpose | $\alpha J^\top e$ | gradient descent on $\tfrac12\|e\|^2$; robust but slow (2% within 300 iterations) |
| Pseudoinverse | $J^+ e$ | Gauss–Newton; fast near the solution, explodes near singularities |
| Damped least squares (Levenberg–Marquardt) | $J^\top(JJ^\top+\lambda I)^{-1}e$ | best of both; **our default** |

Joint limits: clip after each step (or use a bounded solver).

Orientation error must use the short path: if the error quaternion has $w<0$, negate it. Forgetting this was a real bug in our TableTop IK.

## 3. Choosing λ
Too small → unstable steps near singularities; too large → slow convergence. The notebook sweeps λ from $10^{-6}$ to $1$ and finds the best range around $10^{-4}$–$10^{-2}$.

## 4. Redundancy and the null space
For $n > 6$, the null-space projector $N = I - J^+J$ moves the joints without moving the end effector:
$$\Delta q = J^+_{\lambda} e + N\,k\,(q_{\text{rest}} - q)$$
The secondary objective (a comfortable posture, avoiding joint limits or obstacles) is satisfied "for free". Use the **exact** projector ($J^+J$); with the damped version the secondary task leaks into the primary one and the pose is never reached, which the notebook shows.

## 5. Local minima and restarts
DLS from a single seed solved 78% of random reachable targets. Random restarts recovered 20 of the 22 failures → **98%**. Production solvers (TRAC-IK) run several strategies in parallel for the same reason.

## 6. Optimization-based IK
`mujoco.minimize.least_squares` solves
$$\min_q \|e(q)\|^2 + \mu \|q - q_{\text{rest}}\|^2 \quad \text{s.t. bounds}$$
Constraints and costs are explicit, so it extends easily to multiple targets, collision costs and smoothness.

## 7. IK in the VLA stack
Most VLAs (RT-1/2, OpenVLA, LIBERO) output **end-effector deltas**, and an IK or operational-space controller turns them into joint commands. Consequences:
- policies transfer between arms with the same end-effector convention (lecture 41),
- unreachable or singular targets must be handled *below* the policy (TableTop rejects them),
- joint-space VLAs (ACT on ALOHA, π0) skip IK but are tied to one embodiment.

## Key takeaways
- IK = iterative least squares on the pose error; DLS is the workhorse.
- Redundant arms have a null space for secondary objectives; use the exact projector.
- Restarts handle local minima; unreachable targets must be detected.

## References
- Buss 2004, *Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse and DLS*
- Beeson & Ames 2015, *TRAC-IK*
- Kandapagari, *Action Models for Robot Learning*, §4.2
