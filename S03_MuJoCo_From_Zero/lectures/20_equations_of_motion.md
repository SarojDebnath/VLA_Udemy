# 20 · Equations of Motion in Generalized Coordinates

**Section:** S03 · **Type:** Math · **Duration:** ~30 min
**Companion notebook:** `20_equations_of_motion.ipynb`

## Learning objectives
- Write the manipulator equation and identify each term in MuJoCo's `MjData`.
- Understand forward vs inverse dynamics and when each is used.
- Verify the equation numerically on a Panda with contacts.

---

## 1. The manipulator equation
$$M(q)\,\dot v + c(q, v) = \tau + J^\top(q)\, f$$
| Symbol | Meaning | MuJoCo field |
|---|---|---|
| $q$ | generalized positions ($n_q$) | `qpos` |
| $v$ | generalized velocities ($n_v$) | `qvel` |
| $\dot v$ | accelerations | `qacc` |
| $M(q)$ | joint-space inertia, symmetric positive definite | `M` (sparse), dense via `mj_fullM` |
| $c(q,v)$ | bias: gravity + Coriolis + centrifugal | `qfrc_bias` |
| $\tau$ | applied generalized force = passive + actuator + user | `qfrc_passive + qfrc_actuator + qfrc_applied` |
| $J$ | constraint Jacobian (contacts, limits, equalities) | `efc_J` |
| $f$ | constraint forces | `efc_force`; $J^\top f$ = `qfrc_constraint` |

The notebook checks the identity to $10^{-16}$ on a Panda with a box in contact with the floor.

## 2. Where each term comes from
- **$M(q)$** is computed by the composite rigid-body algorithm. Its eigenvalues span orders of magnitude (≈0.1 to 3.7 for the Panda): proximal joints move most of the robot's mass, distal joints move little. That spread is why joint-space control needs per-joint gains.
- **$c(q,v)$** comes from recursive Newton–Euler with $\dot v = 0$. Split it as $c = g(q) + C(q,v)v$:
  - gravity $g(q)$ dominates at low speed (it's what a robot must hold just to stay still),
  - Coriolis/centrifugal terms grow with $v^2$.
- **Kinetic energy:** $T = \tfrac12 v^\top M v$. Without damping or contact, $T + U$ is conserved, so energy drift is a direct check of integration error (notebook 22).

## 3. Forward dynamics (simulation)
$$\dot v = M^{-1}\big(\tau + J^\top f - c\big)$$
with $f$ found by the constraint solver (lecture 21). This is what `mj_step` does.

## 4. Inverse dynamics (control and learning)
Given a desired acceleration, what force produces it?
$$\tau = M(q)\,\dot v + c(q,v) - J^\top f$$
`mj_inverse` computes this into `qfrc_inverse`. Uses:
- **Feed-forward / computed-torque control** (S01 lecture 04, S04).
- **Inverse-dynamics learning** (S02 lab 15 learns $c(q,v)$ from samples).
- **System identification:** fit masses from logged torques.

MuJoCo's soft contact model makes inverse dynamics *well defined even with contacts*, which rigid (LCP) simulators can't guarantee.

## 5. Why VLA builders care
A VLA usually outputs end-effector deltas, and a low-level controller (IK + PD) turns them into torques. Understanding $M$ and $c$ explains:
- why a policy trained on one robot's dynamics struggles on another (cross-embodiment, S13),
- why low-level gains and gravity compensation matter for imitation data quality,
- why safety layers monitor torques and forces (S15).

## Key takeaways
- $M\dot v + c = \tau + J^\top f$, and every term is readable from `MjData`.
- $M$ is symmetric positive definite and configuration-dependent; $c$ = gravity + velocity-product terms.
- Forward dynamics simulates; inverse dynamics (`mj_inverse`) gives the torques for a desired motion.

## References
- MuJoCo docs: *Computation → Physics, Forward dynamics, Inverse dynamics*
- Featherstone, *Rigid Body Dynamics Algorithms* (2008)
- Lynch & Park, *Modern Robotics*, Ch. 8
