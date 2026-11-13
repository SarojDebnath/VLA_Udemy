# 21 · Contacts in MuJoCo: Soft Constraints, Friction Cones, solref and solimp

**Section:** S03 · **Type:** Math · **Duration:** ~30 min
**Companion notebook:** `21_contacts.ipynb`

## Learning objectives
- Describe contact as a convex optimization problem instead of a complementarity problem.
- Interpret `solref` (time constant, damping ratio) and `solimp` (impedance) and predict penetration.
- Choose `condim`, friction coefficients and cone type for manipulation.

---

## 1. Why contact is hard
Rigid contact is non-smooth: forces jump from zero to large values at impact, friction switches between stick and slip, and the classical formulation (LCP / complementarity) can have no solution or many. Grasping is entirely about contact, so its fidelity directly affects whether sim-trained policies work.

## 2. MuJoCo's approach: soft, convex constraints
Each constraint (contact, joint limit, equality) gets a **reference acceleration** that pulls the violation back toward zero like a damped spring:
$$a_{\text{ref}} = -b\,v - k\,r$$
where $r$ is the violation (penetration) and $v$ its velocity. The constraint forces solve a **convex** problem (schematically):
$$f^* = \arg\min_{f \in \mathcal K}\; \tfrac12 f^\top (A + R) f + f^\top (a_u - a_{\text{ref}}),\qquad A = JM^{-1}J^\top$$
- $\mathcal K$: friction cone (normal force ≥ 0, tangential force inside the cone).
- $R$: regularizer from the impedance $d$: $R_{ii} = \frac{1-d}{d}A_{ii}$. This makes contact *soft* and the solution unique.
- Solvers: **Newton** (default, fast convergence), CG, PGS. Constraint islands are solved independently.

Consequence: simulation is smooth, deterministic and invertible, which suits learning and optimization, at the cost of a little penetration.

## 3. `solref`: how fast violations are corrected
Default `solref="0.02 1"` = (time constant $\tau_c$ [s], damping ratio $\zeta$):
$$b = \frac{2}{d_{w}\,\tau_c},\qquad k = \frac{d(r)}{d_w^2\,\tau_c^2\,\zeta^2}$$
- Smaller $\tau_c$ → stiffer contact, less penetration. It must stay ≥ 2× the timestep or the simulation goes unstable.
- **Penetration depends on $\tau_c$, not on mass**, because the constraint is defined as a *reference acceleration*. The notebook confirms this: the same 0.367 mm for 0.1 kg, 1 kg and 10 kg at $\tau_c = 0.02$. This deliberate design keeps contact behavior consistent across objects.
- Negative values give direct stiffness/damping: `solref="-10000 -100"`.

## 4. `solimp`: impedance as a function of penetration
`solimp="0.9 0.95 0.001 0.5 2"` = ($d_0$, $d_w$, width, midpoint, power). The impedance $d\in(0,1)$ rises from $d_0$ to $d_w$ as penetration grows over `width` meters. Higher $d$ = harder constraint. Setting $d_0 = 0$ gives a smooth onset of contact, which is useful for differentiable simulation.

## 5. Friction
- Coulomb friction: $\|f_t\| \le \mu f_n$. An object on an incline sticks while $\tan\theta < \mu$, which is the notebook's stick/slip threshold.
- `friction="μ_slide μ_torsion μ_roll"` (defaults 1, 0.005, 0.0001).
- **`condim`**: 1 = frictionless, 3 = sliding friction, 4 = + torsional, 6 = + rolling. A ball with `condim=6` slows from rolling friction; with 3 it rolls forever.
- **Cone:** `pyramidal` (default, faster) or `elliptic` (more accurate). Grippers often use elliptic with `impratio > 1` (Robotiq: 10) to reduce slip.
- Combining two geoms: friction takes the max (unless `priority` is set); solref/solimp are mixed by `solmix`.

## 6. Practical recipe for manipulation sims
- Fingertip/object friction 1–1.5, `condim=4` for grasps that resist twisting.
- `cone="elliptic"`, `impratio=10` for stable parallel-jaw grasps.
- Timestep 1–2 ms; `solref` time constant ≥ 2× timestep.
- Watch `data.ncon` and `mj_contactForce` to debug grasps; the notebook sums contact forces and recovers the box weight (9.81 N).

## Key takeaways
- MuJoCo solves contact as a regularized convex problem: soft, unique, invertible.
- `solref` sets stiffness through a time constant (mass-independent penetration); `solimp` sets how hard the constraint is.
- Friction cone, `condim` and `impratio` decide whether grasps hold.

## References
- MuJoCo docs: *Computation → Constraint model, Solver parameters, Contact*
- Todorov 2014, *Convex and analytically-invertible dynamics with contacts and constraints* (ICRA)
