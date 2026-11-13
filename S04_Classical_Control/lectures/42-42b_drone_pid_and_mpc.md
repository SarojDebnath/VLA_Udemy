# 42 · 42b · Two Control Showcases: a PID Drone Racer and MPC Through a Crowd

**Section:** S04 · **Type:** Projects (theory + measured results) · **Duration:** ~35 min
**Notebooks:** `42_project_drone_pid_racer` (Skydio X2), `42b_project_mpc_crowd_navigation` (Stanford TidyBot)
**Videos:** `outputs/42_drone_pid_racer.mp4`, `outputs/42b_mpc_crowd_navigation.mp4`

---

## 42 · The PID drone racer

### Why a quadrotor is the classic cascade
Four motors, six degrees of freedom: a quadrotor can't move sideways without tilting first. So control is layered:

```
position error ──PID──▶ desired acceleration ──▶ desired tilt + total thrust ──PD──▶ body torques ──mixer──▶ 4 motor thrusts
     (outer loop, slow)                                         (inner loop, fast)
```
This is the structure inside PX4 and Betaflight. Rule: **the inner loop must be much faster than the outer loop.**

### The mixer (allocation matrix)
Motor $i$ at $(x_i, y_i)$ pushes with $f_i$ and twists the body with $k_i f_i$:
$$\begin{bmatrix}T\\ \tau_x\\ \tau_y\\ \tau_z\end{bmatrix} = A\,f,\qquad A = \begin{bmatrix}1&1&1&1\\ y_1&\cdots&&y_4\\ -x_1&\cdots&&-x_4\\ k_1&\cdots&&k_4\end{bmatrix},\qquad f = A^{-1}(T, \tau)$$
For the X2 (1.325 kg): hover = 3.25 N per motor; a roll command speeds up one side; a yaw command speeds up one diagonal pair.

### From acceleration to tilt
Desired acceleration $a$ (plus gravity) → thrust $T = m\,a_z / (\cos\phi\cos\theta)$, pitch $\theta_d = \operatorname{atan2}(a_x, a_z)$, roll $\phi_d = -\operatorname{atan2}(a_y, a_z)$ (rotated into the yaw frame), clipped to a maximum tilt (0.5 rad).

### Measured (notebook 42)
| Experiment | Result |
|---|---|
| 1 m step, PID | 3.8% overshoot, within 5 cm after 1.58 s. P alone oscillates (the drone is a double integrator); D supplies the damping |
| Unknown 0.3 kg payload (+23% mass) | PD sags 28 cm below the target; **PID settles at 1 cm**. The integral term finds the missing thrust |
| 2 N sideways gust for 4 s | drift 38 cm (PD) vs 31 cm (PID): the integral term helps, but slowly. Fast disturbances need bandwidth or a disturbance observer |
| 5-gate race at 2 m/s | tracking error **118 cm without feed-forward → 14 cm with** (velocity + acceleration of the reference) |
| 3 m/s | 3/5 gates without feed-forward, 5/5 with |
| 4 m/s | fails either way: the 0.5 rad tilt limit saturates in the turns |

**Takeaways.** P makes it move, D makes it stop, I makes it exact under unknown loads. Feed-forward is the biggest single win: let the reference *tell* the controller the motion, and use feedback only to correct errors. Every controller has a speed where its limits bind: find it.

### Anti-windup, two kinds
The integrator is clamped (`i_limit`) **and** only integrates while the error is small (`i_zone`, conditional integration). Without the second, a 1 m step winds the integrator up and the drone overshoots and drifts back slowly.

---

## 42b · Model predictive control by predictive sampling

### The idea
PID reacts to the error **now**. MPC asks every control step: *of all the things I could do over the next $T$ seconds, which works out best?* It simulates candidates with a model, picks the cheapest, executes **only the first action**, and re-plans 0.1 s later (receding horizon).

### Predictive sampling (the method in MuJoCo MPC, Howell et al. 2022)
No gradients, no solver:
1. **Candidates:** the previous plan shifted by one step (warm start), "stop", "drive straight at the goal", and 125 Gaussian perturbations of the plan (5 knots, linearly interpolated to 20 steps = 2 s).
2. **Rollout:** simulate every candidate with the robot's limits (1 m/s, 1.5 m/s²) and **predict the obstacles** (default: constant velocity).
3. **Cost:** $\int \|p - p_\text{goal}\|\,dt$ + terminal distance + a large penalty inside a 0.25 m safety margin + a small effort term.
4. **Execute** the first command of the cheapest candidate.

Vectorized in numpy: **2.3 ms per step** for 128 candidates × 20 steps (budget at 10 Hz: 100 ms).

### The task
The TidyBot crosses a 6 m room while five people walk back and forth across its path. The base needs 0.67 s and 0.33 m to stop from full speed, so reacting late means contact.

### Measured (20 scenarios, success = goal reached with zero contact)
| Controller | Success | Closest approach |
|---|---|---|
| Reactive potential field (P to goal + repulsion from where people are now) | **5/20** | −14 cm (contact) |
| MPC, constant-velocity prediction | **16/20** | +10 cm |
| MPC, bigger safety margin (0.4 m) | 16/20 | +21 cm, 1 s slower |
| MPC, longer horizon (3 s) | 14/20 | +20 cm |
| MPC, perfect prediction (oracle) | 18/20 | +27 cm |

**Takeaways.**
- **Prediction beats reaction.** The potential field only feels a person when they are already close.
- **MPC is only as good as its model of the future.** People here turn around; "they keep walking straight" is wrong exactly then. Better prediction (a learned pedestrian model) is where real crowd navigation improves.
- **A longer horizon isn't free:** 3 s of a wrong prediction is worse than 2 s.
- **Margins buy clearance,** not necessarily success; the confidence intervals of the MPC rows overlap at n = 20.

### Where these connect to the rest of the course
- Chunked policies (S08) execute a *plan* and re-plan: the same receding-horizon idea, with a learned model of what to do.
- World models (S14) are learned versions of the rollout model used here: MPC + a learned model = model-based RL.
- Capstone C7 drives the same TidyBot; extension 1 replaces its simple navigation with this MPC.

## References
- Mahony, Kumar & Corke 2012, *Multirotor aerial vehicles* (IEEE RAM)
- Howell et al. 2022, *Predictive Sampling: Real-time Behaviour Synthesis with MuJoCo* (MuJoCo MPC)
- Williams et al. 2017 (MPPI); Rawlings, Mayne & Diehl, *Model Predictive Control* (textbook)
