# 111–115 · The π-Lineage, Dual-System VLAs, Latency Budgets, and Asynchronous Inference

**Section:** S11 · **Type:** Theory + labs · **Duration:** ~60 min
**Companion notebooks:** `113_latency_dual_system`, `114_async_inference`, `115_project_dual_system`

---

## 111 · π0 → π0.5 → π0.6 → π0.7
- **π0-FAST:** the same backbone with FAST action tokens (autoregressive); trains faster, but inference is slower than flow.
- **π0.5 (2025): open-world generalization.** Co-trained on heterogeneous data (multiple robots, web data, high-level subtask labels, verbal instructions). It predicts a **high-level subtask in text** and then low-level actions with the flow expert, so it can clean an *unseen* kitchen.
- **π0.6 / π*0.6:** RL from experience (RECAP): learn from the policy's own deployments with value-based advantage conditioning and corrections; large gains in throughput and robustness on long tasks (espresso, laundry, boxes).
- **π0.7:** steerable behavior with richer conditioning (book §13.4).
The trend: the same architecture, with better data mixtures, subtask reasoning and post-training on experience.

## 112 · Dual-system architectures
"System 2" (slow, deliberate) + "System 1" (fast, reactive):
| System | Role | Rate |
|---|---|---|
| **Helix (Figure, 2025):** S2 = 7B VLM, S1 = 80M visuomotor transformer | S2 writes a latent "intent" vector; S1 turns intent + fresh pixels into full upper-body humanoid control | 7–9 Hz / **200 Hz** |
| **Helix-02:** adds **System 0**, a learned whole-body balance and locomotion controller (RL-trained) | 1 kHz-class |
| **GR00T N1 (NVIDIA):** S2 = Eagle VLM, S1 = diffusion-transformer action head (flow matching) with cross-attention | 10 Hz / 120 Hz-class |
| **Gemini Robotics / -ER / 2:** an embodied-reasoning VLM plans; an on-robot action decoder executes; whole-body loco-manipulation on Apptronik Apollo | — |

Why split: a single network that is both smart (billions of parameters) and fast (hundreds of Hz) doesn't exist on robot hardware. The interface between the systems is either **language subtasks** (π0.5, SayCan-style) or **latent vectors** (Helix).

## 113 · Latency budgets (notebook 113)
Simulated with a big and a small network:
| Loop | Median | p99 | Missed budget |
|---|---|---|---|
| Single system (big net every cycle) | 14.5 ms | 36 ms | 100% |
| Dual, same thread (big net every 25 cycles) | 0.025 ms | 14 ms | 4% |
| Dual, **decoupled clocks** (separate thread/hardware, shared buffer) | 0.026 ms | 0.14 ms | 0% |

**Tail latency, not the mean, is what breaks real-time control.** Real dual systems run S2 and S1 on separate compute and communicate through a buffer that S1 reads without ever blocking.

## 114 · Asynchronous inference
With an inference latency of L control steps:
- **Synchronous:** the robot stops and waits L steps for every chunk, which lengthens episodes and produces jerky stop-and-go motion.
- **Asynchronous:** keep executing the current chunk; start the next inference early; when it arrives (computed from an L-step-old observation), skip its first L actions (already in the past) and splice it in.
- **Real-time chunking** (Physical Intelligence 2025) goes further: it *inpaints* the new chunk so it matches the actions already committed, which removes discontinuities at the seam.
LeRobot ships an async inference server/client for SmolVLA built on the same idea.

**Measured (notebook 114, flow policy, 30 episodes per cell, mean steps of successful episodes in brackets):**
| Latency | Synchronous | Asynchronous |
|---|---|---|
| 0 ms | 97% (41 steps) | 97% (41) |
| 200 ms | 83% (51) | **100% (42)** |
| 400 ms | 87% (61) | 80% (51) |
| 600 ms | 80% (**70**) | **93% (45)** |

The cleanest signal is **time**. Synchronous episodes get 70% longer as latency grows (41 → 70 steps), because the robot stands still while it thinks. Asynchronous episodes stay almost as fast (41 → 45). Success rates differ by a few episodes either way at n = 30 (overlapping CIs; see S12), so don't over-read the 400 ms row. The video shows both at 600 ms: stop-and-go vs continuous motion.

## 115 · Project 8: dual-system TableTop
- **System 2:** SmolLM2-1.7B-Instruct, prompted few-shot, decomposes a long instruction ("put the red cube in the blue tray and then put the green cube in the blue tray", "first move the green cube into the blue tray, after that the red cube") into subtasks. It runs once per task (0.3 s).
- **System 1:** the S09 FiLM language-conditioned chunk policy, executing each subtask at 10 Hz. Between subtasks the arm retracts to its home pose, so each subtask starts where its demonstrations did.
- **Baselines:** an oracle plan (perfect decomposition) and a single system (the full long instruction fed straight to System 1, which was never trained on compositions).
- **Metrics:** decomposition accuracy, subtasks completed, and full-task success with Wilson CIs.
**Measured (20 two-step episodes, "sort" scene):**
| | Plan accuracy | Subtasks completed | Both steps done |
|---|---|---|---|
| **Dual: LM plan → System 1** | **100%** | 93% | **17/20** (CI 64–95%) |
| Oracle plan → System 1 | 100% | 93% | 17/20 |
| Single system: full instruction → System 1 | — | 25% | **1/20** |

Lessons:
- **Decomposition turns an out-of-distribution instruction into in-distribution ones.** System 1 was only ever trained on single-step instructions. Fed the long sentence it completes 1/20; fed the planner's steps, 17/20, exactly as well as a perfect (oracle) plan.
- **Planner capacity matters.** Our first System 2 was SmolVLM-256M. It copied the few-shot example verbatim for every instruction (0% plan accuracy), so we switched to a 1.7B text model. Use few-shot examples whose colours never appear in your scene, so copying is caught immediately.
- **Errors compound across steps:** 93% per subtask becomes 85% for both; with ten steps it would be under 50%.
- **Check your success metric.** An earlier version of this notebook reported 11/20: the second cube often lands *on top of* the first one in the tray, and the success check required it to lie flat on the tray floor. The robot was right and the metric was wrong. Watch videos of your "failures" before you believe a number. Every extra step multiplies the risk.
- **Start states matter:** between subtasks the arm retracts to its home pose, so each subtask begins from the kind of state its demonstrations began from. It's cheap insurance against starting a skill from a pose the policy has never seen (an exercise: remove the retract and measure the difference).

## Key takeaways
- The π lineage improves through data, subtask reasoning and RL from experience more than through architecture.
- Dual systems pair a slow reasoner with a fast controller over language or latent interfaces.
- Design for tail latency: decouple clocks and use asynchronous chunk execution.

## References
- Physical Intelligence 2025 (π0.5, π*0.6, real-time chunking); Figure 2025 (Helix); NVIDIA 2025 (GR00T N1); Google DeepMind 2025 (Gemini Robotics)
- Shukor et al. 2025 (SmolVLA, async inference)
- Kandapagari, *Action Models for Robot Learning*, Ch. 13.4, Ch. 14, Exercises 14.x
