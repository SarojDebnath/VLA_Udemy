# 71–74 · Multimodality, the Conditional VAE, Action Chunking, and ACT

**Section:** S08 Generative Action Heads · **Type:** Theory + math · **Duration:** ~50 min
**Companion notebooks:** `71_multimodality`, `72_cvae_policy`, `73_action_chunking`, `74_act`

---

## 1. The problem: two good answers (notebook 71)
Demonstrators reach the green goal by going **left or right** around a red obstacle. An MSE-trained policy predicts the conditional mean,
$$\pi_{MSE}(o) = \mathbb E[a\mid o] = \tfrac12 a_{left} + \tfrac12 a_{right} \approx \text{straight ahead},$$
and drives into the obstacle. The learned velocity field points into the disc exactly where the demonstrations split. Real tasks are full of this: grasp a mug by the handle or the rim, push a T from either side (PushT), fold a shirt left sleeve first or right sleeve first.

## 2. Latent-variable policies and the CVAE (notebook 72)
Introduce a latent $z$ that selects the mode:
$$\pi(a\mid o) = \int p_\theta(a\mid o, z)\,p(z)\,dz$$
Train with the ELBO (lecture 13):
$$\log p(a\mid o) \ge \mathbb E_{q_\phi(z\mid o,a)}\big[\log p_\theta(a\mid o,z)\big] - D_{KL}\big(q_\phi(z\mid o,a)\,\|\,\mathcal N(0,I)\big)$$
- **Encoder** $q_\phi(z\mid o,a)$ sees the expert action (the answer) and compresses "which way" into $z$.
- **Decoder** $p_\theta(a\mid o,z)$ reconstructs the action (L1 loss = Laplace likelihood).
- **Reparameterization:** $z = \mu + \sigma\odot\epsilon$ makes sampling differentiable.
- The notebook's 2-D latent cleanly separates left (red) from right (blue) demos.
- **β** trades reconstruction against latent regularity. Measured success with z sampled from the prior: β = 0 gives **0.00** (z is an arbitrary code, KL ≈ 16, and prior samples are junk); β = 0.01 gives **0.85**; β = 1 gives **0.20** (KL → 0, z is ignored, and the decoder is back to averaging).

## 3. Action chunking (notebook 73)
Predict the next **H actions** at once: $\pi(a_{t:t+H}\mid o_t)$.
- **Fewer decisions** per episode: the effective horizon in the $\epsilon T^2$ bound shrinks by about H.
- **Temporal consistency:** a chunk is one coherent motion; a single-step policy can switch modes every step (left, right, left…), which is fatal in multimodal data.
- **Handles pauses and non-Markovian demos:** a human hesitating is easier to model as part of a sequence.

Executing a chunk:
| Strategy | Behavior |
|---|---|
| execute all H, then replan | smooth but blind to changes mid-chunk; jumps at boundaries |
| execute k < H (receding horizon) | trade-off; Diffusion Policy uses k = 8 of 16 |
| **temporal ensembling** (ACT) | query every step and average all chunks that predict the current step with weights $w_i = e^{-k i}$ → smooth and reactive, at H× the compute |

**Measured (notebook 73).** "Single" scene (one cube + a tray; pick or place), 450 demos, the same encoder and data for every row, 50 new layouts:

| Policy | Success | Pick | Place |
|---|---|---|---|
| Single step (H = 1) | **14%** | 28% | 0% |
| H = 16, execute all 16 (open loop) | 98% | 96% | 100% |
| H = 16, execute 8 | **100%** | 100% | 100% |
| H = 16, execute 1 (re-plan every step, no averaging) | 28% | 56% | 0% |
| H = 16, temporal ensembling | **100%** | 100% | 100% |

Read it in three steps:
1. **Chunking turns a failing policy into a working one here:** 14% → 100% with nothing else changed. But check *why* before you generalize (box below).
2. **Predicting a chunk isn't enough; you must also commit to it.** The same H = 16 network drops to 28% when only its first action is executed each step. Every step it re-decides from a slightly different image, and those decisions don't agree with each other (the mode switching described above). Pick half-survives; the longer place task never does.
3. **Temporal ensembling gets reactivity without that penalty** by averaging the overlapping chunks, at the cost of one network call per step.

The video cell puts the single-step and the chunked policy side by side on the same episode.

**Box: is it the chunk, or the loss?** Our regression heads train with an **L1** loss (as ACT does). L1 predicts the *median* of the labels, and in our demonstrations 38% of all actions are exactly zero: the expert stands still while the gripper closes, for a fixed number of steps. A single frame can't tell the policy how long it has been waiting, so the median answer in those states is "don't move", and the robot freezes. We re-ran the comparison with the loss as the only change (50 episodes each, "single" scene):

| Policy | L1 loss | MSE loss |
|---|---|---|
| Single step, relative-target labels | 12% | **94%** |
| Single step, raw delta labels | 18% | 62% |
| H = 16 chunk, relative-target labels | **100%** | **100%** |

MSE predicts the *mean*: a small "go up" that eventually gets the arm out of the wait. A chunk sidesteps the problem completely, because the chunk that starts during the wait already contains the lift that follows it, whatever the loss. Lessons:
- When a policy freezes, compare what it predicts with what the labels look like. Hidden state in the demonstrator (timers, phase counters) is a classic cause.
- Chunks are robust to it with *any* loss; single-step policies only survive it with a loss that averages.
- This is also why OpenVLA-style token policies (argmax = the *mode*) freeze in the same way (S10 notebook 103).

**What each action means matters as much as the network.** Our chunks are *commanded gripper targets, expressed relative to the current gripper pose* (as in Diffusion Policy/UMI), not the expert's raw per-step deltas. With raw deltas the same pipeline scored 0%: a small error in every predicted delta accumulates into centimetres. See the S06 lecture on pixel policies for the full debugging story.

## 4. ACT: Action Chunking with Transformers (Zhao et al. 2023; notebook 74)
- **CVAE encoder:** a transformer over [CLS, joints/state, $a_{t:t+H}$] → $\mu,\sigma$ of a 32-D $z$ (training only).
- **Policy:** ResNet-18 feature maps from each camera (here 2 × 4×4 = 32 image tokens) + state + task + $z$ tokens → transformer encoder → transformer decoder with H learned **queries** → the action chunk in one forward pass.
- **Loss:** L1 reconstruction + β·KL with β = 10.
- **Inference:** $z = 0$ (the prior mean). Mode choice then comes from the *observation*: ACT deliberately trades sampling diversity for consistent, precise motion, and gets its multimodality handling mostly from chunking.
- It made fine bimanual skills (ALOHA: battery insertion, cup opening) learnable from about 50 demonstrations.

Our compact ACT: d = 256, 4 + 4 layers, ≈22M parameters including ResNet-18, trained on 450 demonstrations of the "single" scene (17 minutes on one GPU).

**Measured (notebook 74):** ACT with temporal ensembling **96%** (95% CI 87–99%; pick 92%, place 100%); executing 8 of 16 actions gives the same 96%. The notebook's video shows four rollouts on new layouts.

## Key takeaways
- MSE regression averages modes; latent variables let a policy commit to one.
- Chunking reduces compounding error and enforces temporal consistency.
- ACT = CVAE + transformer + chunking + temporal ensembling.

## References
- Kingma & Welling 2014 (VAE); Sohn et al. 2015 (CVAE)
- Zhao et al. 2023, *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware* (ACT/ALOHA)
- Kandapagari, *Action Models for Robot Learning*, §10.2
