# 42c · 42d · Balance with LQR, and a Quadruped Trot Without Learning

**Section:** S04 · **Type:** Projects (theory + measured results) · **Duration:** ~40 min
**Notebooks:** `42c_project_lqr_balance` (cart-pole + Unitree G1), `42d_project_go2_trot` (Unitree Go2)
**Videos:** `outputs/42c_lqr_balance.mp4`, `outputs/42d_go2_trot.mp4`

---

## 42c · The Linear Quadratic Regulator

### The recipe
1. Pick an equilibrium (upright pole; a standing humanoid) and the control that holds it.
2. Linearize: $x_{t+1} \approx A x_t + B u_t$. MuJoCo computes $A$, $B$ for any model with `mjd_transitionFD` (finite differences of one step), so no hand-derived equations of motion.
3. Choose costs: $Q$ for state error, $R$ for control effort.
4. Solve the discrete algebraic Riccati equation for $P$; the optimal feedback is $u = -Kx$, $K = (R + B^\top P B)^{-1} B^\top P A$.

### Cart-pole (measured)
| R (price of force) | Recovers from up to | Peak force | Settles from 17° in |
|---|---|---|---|
| 0.001 | 52° | 30 N (saturated) | 0.16 s |
| 0.01 | 49° | 23 N | 0.22 s |
| 0.1 | 43° | 13 N | 0.33 s |
| 1.0 | 29° | 10 N | 0.44 s |

Cheaper force = faster and a larger set of recoverable starts, until the actuator limit bites.
**Swing-up:** LQR only works near upright. Energy pumping (Åström & Furuta) pushes the cart in step with the swing until the pendulum has the energy of standing upright; within 20° of upright, LQR takes over (caught after ~0.7 s). A hybrid controller, as real systems do it.

### A 35-DoF humanoid with the same recipe (Unitree G1)
Following the MuJoCo LQR tutorial:
- **Set-point:** the pelvis height where the feet carry exactly the weight; inverse dynamics gives the holding torques; with position servos, $\text{ctrl}_0 = q_0 + \tau_0 / k_p$.
- **Linearize:** $A$ 70×70, $B$ 70×29. **Cost:** large weight on the centre of mass over the feet (CoM Jacobian minus feet Jacobians), moderate on legs and waist, small on arms. $u = \text{ctrl}_0 - K\Delta x$ ($\Delta x$ via `mj_differentiatePos`).
- Three closed-loop eigenvalues stay at 1.0: where the robot stands and which way it faces cost nothing, so LQR leaves them alone.

**Push tests (0.1 s shove on the torso, 16 forces 20–320 N per direction):**
| Direction | Joint PD (each joint holds its angle) | Whole-body LQR |
|---|---|---|
| forward | survives up to 40 N (2/16 forces) | up to 240 N (11/16) |
| backward | falls even at 20 N (0/16) | up to 40 N (2/16) |
| sideways | up to 140 N (7/16) | up to 140 N (7/16) |

In the sagittal plane, LQR moves hips and ankles *together* to put the CoM back over the feet (the ankle and hip strategies of human balance): several times the push tolerance. Sideways, the wide stance does the work for both. **Limit:** one linear controller around one pose can lean but not *step*. That needs MPC or a learned policy (S05 53b).

---

## 42d · A quadruped trot, built by hand (Unitree Go2)

### The stack (the Go2's motors take raw torque)
```
command (vx, vy, yaw rate) → gait clock (trot: diagonal pairs, 0.4 s) → swing/stance foot trajectories
  → Raibert foot placement → analytic leg IK → joint PD (500 Hz) + gravity feed-forward → 12 torques
```
- **Leg IK in closed form:** hip roll puts the foot in the leg's plane; then a 2-link planar IK (law of cosines); knee bends backwards. Checked against the standing keyframe (0, 0.9, −1.8 rad).
- **Stance:** feet slide backwards under the body at the commanded velocity, which moves the body forward.
- **Swing:** lift on a sine profile, land at $p_\text{hip} + v T_\text{st}/2 + k(v - v_\text{cmd})$ (Raibert 1986): step where you are drifting.
- **Gravity feed-forward:** stance feet push down with their share of the weight, $\tau = J^\top (0, 0, -mg/n)$.

### Measured
- **Feed-forward:** joint PD alone sags 2.5 cm, and with backward-bending knees the sag makes it drift backwards at ~0.1 m/s even when told to stand. With feed-forward + stiffer PD + velocity feedback: commanded 0 / 0.3 / 0.6 m/s → achieved −0.03 / 0.24 / 0.51 m/s, at the commanded height.
- **Raibert foot placement vs sideways shoves (trotting at 0.3 m/s, 0.1 s push):** no foot placement survives up to 80 N; k = 0.08 up to 100 N; **k = 0.2 up to 160 N**.
- **Figure-8 with a pure-pursuit path follower** (look-ahead 35 cm, yaw rate ∝ heading error): mean tracking error 3 cm, max 5 cm. The video shows it from a follow camera and from above, with the planned path (white) and the trunk's trail (green).

### Takeaways
- Legged locomotion decomposes into clean layers: timing, foot placement, kinematics, joint control. Each layer is debuggable on its own.
- Feed-forward (gravity, reference velocity) does most of the work; feedback corrects.
- **Foot placement is the balance mechanism of walking.** It's the step that LQR (42c) can't take.
- Compare with RL (S05 53, 53b): the learned policies handle rough terrain, pushes and gait transitions without hand-tuning, but give no guarantees and need a simulator and many millions of steps.

## References
- Kwakernaak & Sivan, *Linear Optimal Control Systems*; MuJoCo LQR tutorial (`mjd_transitionFD`)
- Åström & Furuta 2000, *Swinging up a pendulum by energy control*
- Raibert 1986, *Legged Robots That Balance*; Di Carlo et al. 2018 (MIT Cheetah 3 convex MPC)
