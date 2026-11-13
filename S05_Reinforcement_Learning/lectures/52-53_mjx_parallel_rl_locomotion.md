# 52–53 · Massively Parallel RL with MJX and MuJoCo Playground: Quadruped Locomotion

**Section:** S05 · **Type:** Theory + Colab labs · **Duration:** ~30 min
**Notebooks:** `52_mjx_parallel_rl`, `53_project_go1_walks` (Colab GPU; T4 or better)

> Environment pins (training verified on a Colab T4): `playground==0.2.0`, `jax[cuda12]==0.7.2`, `flax==0.11.2`, `orbax-checkpoint==0.11.22` (brax 0.14.2, mujoco 4.0). Newer JAX releases removed `jax.device_put_replicated`, which brax 0.14 still calls. The "PJRT_Api already exists" / CUDA-13 plugin warnings printed on Colab are harmless.

---

## 1. Why parallel simulation changed robot RL
PPO is sample-hungry (lecture 46–48). When every sample is a GPU-parallel simulation step, the hunger stops mattering: 10⁸ steps take minutes, not weeks. That's how humanoid and quadruped controllers are trained in 2025–2026, including the low-level "System 0/1" controllers that sit under humanoid VLAs (S11, S16).

## 2. MJX
`mujoco.mjx` reimplements the MuJoCo pipeline in JAX (and in NVIDIA Warp for the MJX-Warp backend):
- `mjx.put_model(mj_model)` → device model; `mjx.make_data` → device state.
- `jax.vmap(mjx.step)` steps thousands of worlds in one kernel; `jax.jit` compiles the whole loop.
- One environment is *slower* than CPU MuJoCo (the MJX docs say ~10×); the win is throughput at 1k–10k parallel worlds. Notebook 52 benchmarks 1 → 4096 Go1s.
- The JAX backend is differentiable (mostly); Warp is faster on NVIDIA GPUs and covers nearly all MuJoCo features.

## 3. MuJoCo Playground
Ready-made GPU environments (locomotion, manipulation, dexterous hands) built on MJX and Menagerie robots, with tuned PPO configs for Brax:
```python
env = registry.load("Go1JoystickFlatTerrain")
cfg = locomotion_params.brax_ppo_config("Go1JoystickFlatTerrain")   # 8192 envs, unroll 20, 32 minibatches...
make_inference_fn, params, _ = ppo.train(environment=env, wrap_env_fn=wrapper.wrap_for_brax_training,
                                         randomization_fn=registry.get_domain_randomizer("Go1JoystickFlatTerrain"), ...)
policy = make_inference_fn(params, deterministic=True)               # params = (obs normalizer, policy, value)
```
Locomotion envs include Go1 (joystick, getup, handstand), Spot, Barkour, Berkeley Humanoid, Booster T1, Unitree G1/H1, Apptronik Apollo and OP3.

## 4. The Go1 joystick task
- **Observation:** proprioception history (gyro, gravity vector, joint positions and velocities, last action) + velocity command. The critic gets privileged state (asymmetric actor-critic).
- **Action:** joint position offsets at 50 Hz around a default stance, tracked by PD.
- **Reward:** track the commanded linear/angular velocity, plus penalties for energy, action rate, foot slip and base orientation, plus a feet-air-time term for a clean gait.
- **Domain randomization:** friction, link masses, motor strength and joint offsets; push perturbations (`pert_config`).

## 5. The project (notebook 53)
1. **Train once, save forever:** train with domain randomization (6×10⁷ steps requested; with 10 evaluations the run actually covered 7.4×10⁷ steps in ~26 min on a T4) and save the parameters to Google Drive. Every later run loads them (`get_policy`). Losing a 25-minute training run because the weights only lived in a Colab session is a lesson you learn once.
2. **Watch it walk:** a scripted joystick session (forward, arc left, side-step, backward, stand) rendered with the robot-tracking `track` and `side` cameras.
3. **Command tracking sweep:** commanded vs achieved forward velocity.
4. Train a second policy without randomization (also saved).
5. **Stress test:** lateral pushes of increasing size, 8 seeds in parallel per push, fraction still standing for each policy.

**Results from our T4 run (domain-randomized policy):**
| | |
|---|---|
| Eval reward | 0 → 9.5 (8M steps) → 17.4 (16M) → 19.5 (25M) → **23.6 (74M)** |
| Command 1.0 m/s | achieved **0.95 m/s**, no fall in 400 steps |
| Tracking sweep | −1.0 → −0.95, −0.5 → −0.42, 0 → 0.0, 0.5 → 0.52, 1.0 → 0.95, **1.5 → 0.0** |

The last row is the teaching moment. The policy tracks well inside the command range it was trained on and **stops** when asked for 1.5 m/s, a command it never saw. A learned controller doesn't extrapolate; it's only as good as its training distribution. That's the same lesson as covariate shift in imitation learning (S06) and out-of-range action tokens (S10), now in an RL controller.

Expected stress-test picture (the lecture 49–51 thesis at scale): the randomized policy gives up a little nominal tracking and survives larger pushes. Fill in your own push-survival plot after running the last cell.

## 6. Engineering for speed: writing JAX code that doesn't waste the GPU
A first version of this notebook needed about 2 minutes per 400-step video and over an hour for the push test. Three mistakes, three fixes:
| Mistake | Symptom | Fix |
|---|---|---|
| Re-creating the env and `jax.jit`-ing `reset`/`step` inside every rollout | recompilation on every call | build and compile **once**, reuse |
| A Python `for` loop calling the jitted step 400 times | 400 host↔GPU round trips | the whole episode in **one** `jax.lax.scan`; commands and pushes become `(T, 3)` arrays |
| `env.render(trajectory)` copying every state off the GPU separately | ~0.5 s per frame | stack `qpos` on the GPU, copy **once**, render with a plain `mujoco.Renderer` |

Then `jax.vmap` over random keys runs all seeds of the stress test as **one** batched call. The same principles (compile once, keep loops on the device, batch everything, minimize transfers) are what make MJX training itself fast.

**Cameras:** the default free camera stays at the world origin, so a walking robot leaves the frame. The Go1 model defines `trackcom` cameras (`track`, `side`, `back`, `top`) that follow the center of mass. Use them for every video.

## 7. From here to VLAs
Locomotion policies trained this way are the "body" under a humanoid VLA: the VLA (System 2) chooses *where* to go or *what* to grasp, and the RL-trained controller turns that into stable whole-body motion (Helix's System 0, GR00T, Gemini Robotics 2; S11, S16).

## Key takeaways
- MJX trades single-env speed for massive parallelism; PPO's sample hunger becomes affordable.
- Playground gives tuned locomotion tasks with domain randomization and push perturbations.
- A learned locomotion policy tracks commands well *inside* its training range and fails outside it (1.5 m/s → stops).
- Fast JAX: compile once, run whole episodes with `lax.scan`, batch seeds with `vmap`, copy results off the GPU once; save trained weights immediately.
- RL-trained low-level controllers and imitation-trained VLAs are complementary layers.

## References
- MuJoCo docs: *MJX*; Zakka et al. 2025, *MuJoCo Playground*
- Rudin et al. 2022, *Learning to walk in minutes using massively parallel deep RL*
- Kandapagari, *Action Models for Robot Learning*, §7.3, §7.5

---

## 53b · Project: a humanoid that keeps its balance (Unitree G1, `53b_project_g1_rl_balance`)
**Setup:** MuJoCo Playground `G1JoystickFlatTerrain`: 29 joint targets at 50 Hz, a 103-D observation for the actor (joint angles/velocities, gravity direction, angular velocity, command, last action) and a 216-D privileged observation for the critic. PPO with 8,192 parallel environments, domain randomization and random pushes during training. **200M environment steps in 20 minutes** on one RTX PRO 6000; evaluation reward −6 → +14.

**Push test (standing, 16 seeds per kick, velocity kick on the pelvis):**
| Kick (m/s) | 0 | 0.25 | 0.5 | 0.75 | 1.0 | 1.25 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|---|---|
| learned, forward | 100% | 100% | 100% | 100% | 75% | 75% | 75% | 62% |
| learned, backward | 100% | 100% | 100% | 94% | 75% | 69% | 75% | 50% |
| learned, sideways | 100% | 100% | 100% | 88% | 88% | 75% | 69% | 56% |
| hold the pose (joint PD) | 0% | 0% | 0% | 0% | 0% | 0% | 0% | 0% |

The pose-holding baseline doesn't even stand with zero push: Playground's G1 uses soft joint gains made for a policy that keeps moving, and with them a fixed pose collapses. Balance is a closed loop through the whole body, not a property of the joints. The learned policy recovers from most shoves **by stepping**, which LQR (S04 42c) cannot do. A joystick session (walk, curve, side-step, walk back, two shoves) completes without a fall; the video shows it.

**Trade-offs to discuss:** no model needed at run time, but 200M simulated steps of practice; no stability guarantee (S15 wraps learned controllers in safety filters).
