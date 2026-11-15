# 124–132 · Fine-Tuning a VLA for Your Robot

**Section:** S13 Fine-Tuning · **Type:** Math + theory + labs · **Duration:** ~70 min
**Notebooks:** `124_lora_from_scratch`, `126_coverage_beats_volume`, `127_smolvla_finetune_colab`, `128_openvla_qlora_colab`, `129_lora_ablations`, `130_overspecialization`, `131_visual_domain_randomization`, `132_project_new_embodiment`

---

## 124 · LoRA: low-rank adaptation
Freeze the pretrained weight $W_0 \in \mathbb R^{d\times k}$ and learn a low-rank update:
$$W = W_0 + \frac{\alpha}{r}BA,\qquad B\in\mathbb R^{d\times r},\ A\in\mathbb R^{r\times k},\ r \ll \min(d,k)$$
- **Init:** $A$ random, $B = 0$, so training starts *exactly* at the pretrained model.
- **α/r scaling** keeps the effective learning rate roughly constant when you change r.
- **Merging:** $W_0 + \frac{\alpha}{r}BA$ is a single matrix, so inference costs nothing extra (verified in the notebook).
- **Why it works:** fine-tuning updates have low intrinsic rank. In the notebook, a rank-4 update (singular values 27.9, 25.9, 25.4, 23.9, then exactly 0) is fit exactly by r ≥ 4 and poorly by r < 4: final MSE 3.36 (r=1), 2.26 (r=2), **1.5·10⁻⁸ (r=4, 1.6% of the parameters)**. Larger ranks (8, 32) fit slightly *worse* in the same number of steps (2·10⁻², 5·10⁻⁴), because extra capacity costs optimization time.
- **Memory (notebook estimates, weights + grads + Adam):** OpenVLA-7B full fine-tune ≈ 120 GB, LoRA ≈ 16 GB, QLoRA ≈ 5 GB; π0 ≈ 53 / 7 / 2 GB. That's why QLoRA is what makes a 7B VLA fine-tunable on one consumer GPU.
- **Memory:** a full fine-tune of a 7B model needs ~112 GB (weights + gradients + Adam). LoRA trains ~1% of the parameters; **QLoRA** (Dettmers 2023) also stores the frozen base in 4-bit NF4, so OpenVLA fits on a single consumer or Colab GPU.

## 125 · Picking a base model and a fine-tuning mode
| Base | Size | Strength | When |
|---|---|---|---|
| OpenVLA | 7.5B | tokens, OXE pretraining, LoRA-friendly | single third-person camera, slow control is OK |
| Octo | 27–93M | flexible observations, diffusion head | cheap fine-tuning, new sensors |
| π0 / π0.5 | 3.3B | flow chunks, 50 Hz, multi-camera | dexterous, high-frequency |
| SmolVLA | 450M | flow expert, LeRobot-native, runs on consumer GPUs | SO-100/101, community data |
| Distilled or custom (our MiniVLA) | <1B | full control, cheap | research, education |

Modes: **head-only** (fast; weak if the visual domain changes), **LoRA** (the default, ≈ full quality), **full** (best with lots of data and compute; risk of forgetting). Keep the vision encoder trainable when the scene looks different from pretraining (OpenVLA's finding).

## 126 · Demonstrations that don't waste your time
Notebook result, success on the FULL workspace:

| Demonstrations | 40 | 160 |
|---|---|---|
| Narrow region (10 × 16 cm) | 3% | **0%** |
| Full workspace | 22% | **62%** |

4× more narrow demonstrations doesn't help at all, while the same budget spread over the workspace triples success. Coverage of object positions, orientations, distractors, lighting and approach strategies matters more than repetitions.

Checklist: one consistent action convention and gripper sign; reset and re-randomize between episodes; record at the deployment control rate; include recoveries (DART/interventions); check normalization statistics; replay a few demos open-loop on the robot before training.

## 127–128 · Fine-tuning real VLAs
- **SmolVLA via LeRobot** (Colab): `lerobot-train --policy.path=lerobot/smolvla_base --dataset.root=<v3 dataset>`, with a `--rename_map` from our camera names to the policy's `camera1/camera2`. The base config expects a 6-D state and 3 cameras; when you fine-tune, the dataset's features define the new input and output heads.
- **OpenVLA QLoRA** (Colab): 4-bit base + LoRA r = 32 on all linear layers, actions mapped to the last 256 vocabulary ids with your own q01/q99, **left padding** so action tokens follow the prompt.

## 129–130 · Ablations and over-specialization
On MiniVLA v2 (SmolVLM-256M + LoRA + flow action expert, S11), "single" scene, 3,500 fine-tuning steps per run, 40 evaluation episodes each:

| Demonstrations (LoRA rank 64) | 50 | 150 | 450 |
|---|---|---|---|
| Success | 28% (16–43%) | **60%** (45–74%) | 62% (47–76%) |

| LoRA rank (450 demos) | 4 | 16 | 64 |
|---|---|---|---|
| Success | 35% (22–50%) | 30% (18–45%) | **62%** (47–76%) |

- **Data:** success doubles from 50 to 150 demonstrations, then flattens. Once the demos cover the workspace, a fixed training budget gains little from more of them (notebook 126 makes the same point about *coverage*).
- **Rank:** on this task the LoRA rank matters. Ranks 4 and 16 are indistinguishable from each other (overlapping intervals), and rank 64 is clearly better. A 256M backbone adapting to a new *action* output, not just a new style of text, needs capacity. Measure it for your task before copying a default.

**Three-set diagnosis** (25 demos, rank 64, long training):
1. initial conditions from training → high success;
2. the same scenes with objects moved ~3 cm → collapse;
3. open-loop replay on a training episode → predictions match the demo.
High (1) + clean (3) + low (2) = **over-specialization**: it memorized layouts. If (3) fails instead, suspect a convention or normalization bug, not the amount of data.

**Measured (notebook 130):** MiniVLA v2 fine-tuned on only 25 demonstrations, LoRA rank 64, 1,500 steps:
| Test set | Success (25 episodes) |
|---|---|
| 1) the exact training layouts | **60%** |
| 2) the same layouts, cubes nudged ~3 cm | **32%** |
| fresh layouts | 8% |
| 3) open-loop replay of a training episode | next-target error 5 mm (x), 6 mm (y), 24 mm (z) |

Nudging the cubes by 3 cm halves success: the model learned *these 25 scenes*, not *how to pick*. The video shows the same layout succeeding, and then failing once the cubes have moved. The z replay error (24 mm, four times the x/y error) is worth a look too: the model is least certain about *how far down* to go. Exercise: plot it over the episode and check whether the failures happen during the descent.

## 131 · Sim-to-real fine-tuning loops, in miniature
The S05 pendulum showed that dynamics randomization doesn't automatically help a feedback policy. For **pixel policies**, visual variation is the gap that actually bites. The notebook collects TableTop data with per-episode random camera pose (±5 cm, ±4° FOV), lighting (×0.4–1.6) and table color, then evaluates both flow policies on the held-out S12 perturbation suite. Measured (30 episodes per perturbation, "single" scene, flow-matching policy, 500 demos each):

| Training data | Nominal | Mean over visual perturbations | Worst perturbation |
|---|---|---|---|
| Nominal | **100%** | 91% | 77% |
| Visually randomized | 93% | **96%** | **90%** |

A 7-point nominal cost buys a 13-point better worst case. That's the classic DR trade, and the right one when you don't know the real camera pose to the centimeter.

The real-world loop: sim pretraining with randomization → small real dataset → fine-tune → evaluate with intervals → collect interventions on failures → repeat.

## 132 · Project 9: a new embodiment
Panda-trained flow policy → UR5e + Robotiq (different arm, gripper, wrist-camera view; the same end-effector action space):
| Policy | UR5e success (40 episodes) |
|---|---|
| Zero-shot Panda policy | **0%** |
| UR5e from scratch, 50 demos | 50% (35–65%) |
| **Panda policy fine-tuned on the same 50 UR5e demos** | **88%** (74–95%) |

Zero-shot fails completely: the arm, gripper and wrist-camera image all differ. But the Panda pretraining isn't wasted. With the same 50 demonstrations, starting from the Panda policy beats starting from scratch by 38 points. That's the cross-embodiment argument behind Open X-Embodiment in one table. The video shows the Panda policy driving the UR5e before and after fine-tuning.

**Recipe card for a new embodiment:** match the action space (EE deltas, frame, gripper sign) → record 50–200 diverse demos at the deployment rate → compute normalization stats → LoRA fine-tune (vision trainable) → replay check → evaluate with CIs and a perturbation suite → iterate on the failures.

## Key takeaways
- LoRA/QLoRA make billion-parameter fine-tuning feasible on one GPU; start from B = 0 and merge for deployment.
- Coverage beats volume; diagnose failures with train/shifted/replay sets.
- Visual randomization and pretraining both buy robustness; measure it with a suite, not one number.

## References
- Hu et al. 2022 (LoRA); Dettmers et al. 2023 (QLoRA)
- Kim et al. 2024 (OpenVLA fine-tuning study); Shukor et al. 2025 (SmolVLA)
