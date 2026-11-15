# 33–35 · Dynamics-Based Control: PD + Gravity, Computed Torque, Operational Space, Impedance

**Section:** S04 · **Type:** Math · **Duration:** ~45 min
**Companion notebooks:** `33_pd_gravity_compensation`, `34_computed_torque`, `35_operational_space_impedance`

## Learning objectives
- Explain why PD control sags under gravity and how feed-forward fixes it.
- Derive computed-torque control and its robustness to model error.
- Derive operational-space control and use impedance for safe contact.

---

## 1. Setup: a torque-controlled Panda
`vlacourse.torque_panda()` replaces the Menagerie position servos with 7 torque motors (87/12 N·m limits). Now we write $\tau$ ourselves, using the manipulator equation (lecture 20):
$$M(q)\ddot q + C(q,\dot q)\dot q + g(q) = \tau$$

## 2. PD control and gravity sag (notebook 33)
$$\tau = K_p(q_d - q) - K_d\dot q$$
At rest, $\dot q = \ddot q = 0$, so $K_p(q_d - q) = g(q)$:
$$q_{ss} - q_d = -K_p^{-1} g(q)$$
The heavy shoulder/elbow joints sag (4° on joint 2 with $K_p = 600$). Raising $K_p$ shrinks the sag but makes the robot stiff and dangerous.

**PD + gravity compensation:** $\tau = K_p e - K_d\dot q + g(q)$ gives zero steady-state error. Every commercial arm does this internally. Our UR5e TableTop needed it too (MuJoCo's `gravcomp` body attribute).

## 3. Computed-torque control (notebook 34)
Choose a desired acceleration with PD feedback, then cancel the dynamics:
$$\tau = M(q)\big(\ddot q_d + K_p e + K_d\dot e\big) + C(q,\dot q)\dot q + g(q)$$
If the model is exact, the closed loop becomes **linear and decoupled**:
$$\ddot e + K_d\dot e + K_p e = 0$$
so every joint behaves like a critically damped spring you choose. In MuJoCo it's one call: set `qacc`, call `mj_inverse`, read `qfrc_inverse`.

**Model error:** with +30% link mass in the controller's model, the error grows, but the outer PD loop absorbs it: RMS error 10.8° → 2.9° → 0.74° as gains go from $K_p = 25$ to $400$. **Feedforward does the bulk of the work; feedback cleans up.** This split recurs everywhere, including VLAs that predict actions while a low-level controller tracks them.

## 4. Operational-space control (notebook 35)
Control the end effector directly. Task-space dynamics:
$$\Lambda(q)\,\ddot x + \mu + p = F,\qquad \Lambda = (J M^{-1} J^\top)^{-1}$$
$\Lambda$ is the **effective mass** felt at the end effector. Choose
$$F = \Lambda\big(\ddot x_d + K_p e_x + K_d\dot e_x\big),\qquad \tau = J^\top F + N^\top\tau_{\text{null}} + c(q,\dot q)$$
with the dynamically consistent null-space $N = I - J^\top\bar J^\top$, $\bar J = M^{-1}J^\top\Lambda$, holding a rest posture. The notebook tracks a 10 cm circle at 0.5 Hz to millimeter accuracy (Khatib 1987).

## 5. Impedance control
Make the end effector behave like a spring-damper around the target:
$$F_{\text{ext}} \approx K\,\Delta x \quad\Rightarrow\quad \Delta x = K^{-1}F_{\text{ext}}$$
In our formulation the stiffness felt is $\Lambda K_p$. With a 20 N push, $K_p = 100$ gives 68 mm deflection, 400 gives 17 mm, 1600 gives 4 mm, and the arm returns smoothly afterwards.

Why it matters for learned policies:
- **Contact-rich tasks** (insertion, wiping) need compliance, not stiffness.
- **Safety:** a compliant arm doesn't crush an unexpected obstacle (S15).
- Many VLA deployments output end-effector targets to an impedance controller running at 1 kHz.

## 6. Control-rate hierarchy
| Layer | Rate | Example |
|---|---|---|
| VLA policy | 3–50 Hz | OpenVLA ~6 Hz, π0 50 Hz chunks |
| IK / OSC | 100–1000 Hz | the TableTop IK runs at the action rate |
| Joint torque loop | 1–10 kHz | Franka FCI 1 kHz |

## Key takeaways
- PD sags by $K_p^{-1}g(q)$; gravity compensation removes it.
- Computed torque linearizes the dynamics; feedback absorbs model error.
- OSC controls the end effector through $\Lambda$; impedance sets compliance, which is key for contact and safety.

## References
- Khatib 1987, *A unified approach for motion and force control of robot manipulators: the operational space formulation*
- Hogan 1985, *Impedance Control*
