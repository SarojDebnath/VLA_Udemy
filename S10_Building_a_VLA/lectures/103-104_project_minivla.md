# 103–104 · Project 7: MiniVLA from Scratch, and Scaling Laws in Miniature

**Section:** S10 · **Type:** Project brief · **Notebooks:** `103_project_minivla`, `104_data_scaling` · **Library:** `vlacourse/minivla.py`

## Goal
Reproduce the OpenVLA recipe at laptop scale and see every moving part: a pretrained VLM, repurposed vocabulary tokens as action bins, LoRA fine-tuning on robot demonstrations, constrained decoding, and closed-loop evaluation in TableTop.

## Recipe
| Component | OpenVLA | MiniVLA |
|---|---|---|
| VLM | Prismatic-7B (DINOv2+SigLIP → Llama-2 7B) | SmolVLM-256M (SigLIP-style encoder → pixel-shuffle projector → SmolLM2) |
| Image | 224², one third-person camera | 256² front camera (16 image tokens), upsampled on GPU from 128² |
| Prompt | "In: What action should the robot take to {x}?\nOut:" | the same question through SmolVLM's chat template |
| Action tokens | 256 least-used Llama ids | the last 256 vocabulary ids (49024–49279) |
| Bins | 1st–99th percentile per dimension | same (`ActionBins`) |
| Fine-tuning | full or LoRA r = 32 | LoRA r = 16 on attention of **both** vision and language towers + trainable LM head (31M of 288M parameters) |
| Loss | next-token CE on the 7 action tokens | same (prompt tokens masked with −100) |
| Decoding | greedy, 7 tokens | greedy, logits restricted to the 256 action ids |

Memory: about 1.1 GB with gradient checkpointing at batch 16, which trains on the 4 GB laptop GPU (≈1.4 s/step) or a Colab T4.

## What to measure
1. **Action-token accuracy** on held-out episodes. It's a proxy and correlates only loosely with success.
2. **Closed-loop success** with Wilson CIs (30 episodes), compared with the Project 5/6 baselines that use two cameras and continuous heads.
3. **Paraphrase robustness:** unseen wordings ("grab the red block"). The language model brings this for free.
4. **Latency per action** (7 sequential decoding passes). This is the motivation for S11's continuous heads.

## What actually happens (and why it's the best lesson in S10)
Measured on the "single" scene (one cube + a tray, 450 demos, front camera), 30 new layouts:
- **Action-token accuracy: 85–88%** on held-out frames. By the usual ML metric, the model is fine.
- **Closed-loop success: 0%** (CI 0–11%). The video cell shows what the robot does instead; watch it before reading on.

**Debugging it, step by step (the notebook's second half).**

*Step 1: compare what the model outputs with the labels.* 39% of the training labels have no translation at all: the scripted expert pauses on a *timer* while the gripper closes. The model predicts "no translation" **69%** of the time. Argmax decoding picks the most likely bin, a single frame can't show how long the pause has lasted, so "don't move" wins and the arm freezes. (Same mechanism as L1 regression in S08 lecture 73.)

*Step 2: apply the standard fixes, one at a time, and measure each (30 test layouts):*

| Variant | Success |
|---|---|
| raw per-step deltas, argmax (the OpenVLA recipe) | 0% |
| relative-target labels (never "zero") | 0% |
| + decode the bin *mean* instead of the most likely bin | 0% |
| + predict 8 actions per call (56 tokens), execute 4 or 8 | 0% |
| + 512 px images (64 instead of 16 image tokens) | 0% |
| + gripper position written into the prompt | 0% |

*Step 3: look again.* The arm no longer freezes: it reaches decisively, descends and closes the gripper, **5–11 cm away from the cube, and always short of it** (top-down trace in the notebook, 4 test layouts). It heads in the right direction and stops too early: the predictions are pulled toward the average target of the training data. The remaining failure is **precision**: a 256M VLM emitting discrete tokens, fine-tuned on 450 demos, doesn't localize a 4 cm cube to the centimetre. Every bin is a hard decision, and one wrong bin early in the sequence changes the context for all later tokens.

**Lessons:**
- **Token accuracy (85–88%) is a proxy, and here it fails completely.** Always evaluate in closed loop.
- Fix one diagnosed failure at a time and *re-diagnose*: fixing the freeze revealed the precision problem underneath.
- This is the field's history in one notebook. OpenVLA made the token recipe work with 7B parameters and ~970k episodes; FAST compresses actions into better tokens; π0, SmolVLA and GR00T stopped asking the language model for numbers and added a continuous action expert. S11 notebook 108 does exactly that, with the **same SmolVLM backbone and the same data: 0% → 53%**.

## 104 · Scaling in miniature
Hold compute fixed (600 steps) and vary the number of demonstrations (10 → 180). Held-out cross-entropy keeps falling with more *diverse* data, which is the small-scale version of the data-scaling argument behind Open X-Embodiment: below a diversity floor the model memorizes; above it, it generalizes.

**Measured (same 600 steps of compute each):**
| Demos | 10 | 30 | 90 | 180 |
|---|---|---|---|---|
| Held-out cross-entropy | 3.05 | 2.12 | 1.57 | 1.36 |
| Held-out token accuracy | 71.7% | 74.4% | 75.4% | 75.9% |

The loss keeps falling smoothly with more data while token accuracy saturates around 75%. Accuracy is a coarse proxy, and the cross-entropy shows the model is still getting more *confident* in the right bins.

## References
- Kim et al. 2024 (OpenVLA); Marafioti et al. 2025 (SmolVLM); Hu et al. 2022 (LoRA)
