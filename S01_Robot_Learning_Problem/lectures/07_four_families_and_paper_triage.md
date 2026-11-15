# 07 · The Four Families of Action Models, and How to Triage a Paper

**Section:** S01 · **Type:** Theory + exercise · **Duration:** ~15 min (+90 min exercise)

## Learning objectives
- Compare the four families on data needs, generality, precision and failure modes.
- Use the 7-cell paper triage table to read any robotics paper in 20 minutes.

---

## 1. The four families side by side
| | Classical / analytical | Reinforcement learning | Imitation learning | Foundation / VLA |
|---|---|---|---|---|
| Training signal | derivation from a model | reward | demonstrations | web pretraining + demos |
| Data needed | none (a model) | millions of interactions | 10s–1000s of demos | 100k–1M episodes (pretraining) + 10s–100s (fine-tune) |
| Strength | precision, guarantees | can exceed humans | simple, sample-efficient | generalizes to new objects and instructions |
| Typical failure | model mismatch | reward hacking, sim-to-real | compounding error, multimodality | silent failures, latency, cost |

## 2. Worked example: the dishwasher
*Task: empty a dishwasher.*
- **Classical:** CAD models of plates and racks, grasp planner, motion planner. Fails on an unknown mug.
- **RL:** reward = items placed in the cupboard. Needs a simulator and careful reward shaping; may learn to throw plates.
- **Imitation:** 200 teleoperated demos with a leader-follower rig. Fails when a plate is in an unseen slot.
- **VLA:** fine-tune a pretrained checkpoint on 50 demos; prompt "put the blue mug in the cupboard". May fail silently with a confident wrong grasp; needs a safety layer.

## 3. The 7-cell triage table
For each paper, read the abstract and figures only (20 minutes) and fill in:
1. **Family** (one of four)
2. **Era** (from lecture 06)
3. **Slot 1, input**: resolution, cameras, proprio, language, history
4. **Slot 2, output**: what is emitted, and at what rate
5. **Slot 3, training signal**: on what data
6. **Compounding-error response**: what the paper does about drift
7. **Why it mattered**: one sentence, no hedging

**Worked example: ALVINN (1988)**
| Cell | Value |
|---|---|
| Family | Imitation |
| Era | 3 (first learned approaches) |
| Input | 30×32 grayscale, single front camera |
| Output | steering angle (scalar), ~10 Hz |
| Training signal | supervised regression on human driving |
| Compounding error | none explicit; drift was the dominant failure |
| Why it mattered | first neural network to output a control signal directly from pixels on real hardware |

## 4. Exercises
1. Triage four papers, one per family: LaValle *Planning Algorithms* Ch. 5 (RRT); Mnih et al. 2015 (DQN); Zhao et al. 2023 (ACT); Kim et al. 2024 (OpenVLA).
2. Write the dishwasher problem for a household task you know, one paragraph per family (under 150 words each).
3. Define without looking: compounding error, three-slot anatomy, action tokenization, the classical/learned dividing line, cross-embodiment generalization.

## Key takeaways
- The four families trade data for generality and precision for flexibility.
- The 7-cell table turns any paper into a comparable summary.

## References
- Sapkota et al. 2025, *Vision-Language-Action Models: Concepts, Progress, Applications and Challenges* (survey)
