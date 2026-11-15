# 39–41 · Projects: The Three-Layer Cake, the Data Engine, Cross-Embodiment, and Where Classical Methods Still Carry Weight

**Section:** S04 · **Type:** Project briefs + theory · **Duration:** ~30 min
**Notebooks:** `39_project_pddl_to_torque`, `40_project_data_engine`, `41_cross_embodiment_expert`

---

## Project 2: PDDL → IK → joint control (notebook 39)
**Goal:** turn the 10-step Blocksworld plan into physical motion on the Panda.

The three-layer cake:
| Layer | Our implementation | Output |
|---|---|---|
| Symbolic | pyperplan A* + h_FF on `blocksworld.pddl` | `(unstack a b)`, `(put-down a)`, … |
| Geometric | per-action waypoints (above, at, grasp, lift) from **current** block poses, DLS IK | joint targets |
| Dynamic | quintic interpolation → position servos; then a replay with computed torque | torques |

**Result:** the tower a-b-c-d-e is built with < 2 cm xy spread. The computed-torque replay tracks the recorded trajectory with 0.04° RMS error. Joint-4 torque is non-zero at rest (gravity) and peaks during lift and transport (inertia).

**Lessons:**
- Reading block poses *at execution time* (closed loop) absorbs small drift; a fixed pre-computed trajectory would not.
- `put-down` needs a free table slot, which is geometric reasoning the symbolic layer doesn't do.
- The learned layer is the missing piece: a policy that maps pixels + language to actions without hand-written domains or poses. The rest of the course builds it.

## Project 3: the data engine (notebook 40)
**Goal:** produce the demonstrations every later model trains on.
- `vlacourse.collect()` runs the privileged scripted expert in TableTop and records, per step: front + wrist images (128 × 128), state (8-D), joint angles, **expert action (the label)** and executed action.
- Diversity: random layouts, cube yaw, task mix (pick/place/stack × 3–4 colors = 12 instructions), and randomized expert speed and hover height (timing multimodality).
- Format: one compressed `.npz` per episode + `meta.json` with action/state statistics (mean, std, 1st/99th percentiles for normalization and tokenization in S10).
- Dataset `data/tabletop_panda_v1`: **200 episodes, 8,420 frames, 166 MB**, 100% expert success.
- **DART option:** `dart_noise > 0` executes noisy actions but records the clean expert label, so the data contains recoveries from perturbed states. Used in S06 against compounding error.

Why a scripted expert rather than teleoperation? It's free, perfectly reproducible, and infinitely scalable, and the failure modes of learned policies become cleanly attributable to the learning, not the data. Teleop (S06) adds the human multimodality and noise that real datasets have.

## Cross-embodiment (notebook 41)
The same TableTop task interface (end-effector deltas + gripper) and the same scripted expert run unchanged on:
| Robot | Arm DoF | Gripper | Expert success | Notes |
|---|---|---|---|---|
| Franka Panda | 7 | parallel jaw (tendon) | 100% | high-gain servos |
| UR5e + Robotiq 2F-85 | 6 | linkage gripper, ctrl 0 = open | 100% | needed gravity compensation and a sensible ready posture |
| SO-101 | 5 | rotating jaw | reach only | 5-DoF: position + soft "point down" only; ~0.4 m reach; jaw axis is the site's x, not z |

The end-effector trajectories look almost identical while the joint trajectories are completely different. This is why **end-effector action spaces enable cross-embodiment datasets** (Open X-Embodiment, S10) and why joint-space policies don't transfer.

Lessons that recur in S13 (fine-tuning for a new robot):
- gripper conventions differ (open = 255 vs open = 0),
- workspace scale differs (SO-101 needs a rescaled table),
- tool-frame conventions differ (z vs x approach axis),
- low-level controller quality (gravity compensation, gains) changes what the same action does.

## Where classical methods still carry the load
Even in 2026 VLA systems:
1. **Below the policy:** IK, operational-space/impedance control, joint servos at 1 kHz.
2. **Around the policy:** collision checking, workspace limits, force thresholds (safety shields, S15).
3. **Before the policy:** planners and scripted experts generate training data (MimicGen, our data engine).
4. **Above the policy:** task decomposition (PDDL / LLM planners, System 2).

The dividing line moves up over time, but the classical stack never disappears.

## References
- Laskey et al. 2017, *DART*; Mandlekar et al. 2023, *MimicGen*
- Open X-Embodiment Collaboration 2023 (arXiv:2310.08864)
