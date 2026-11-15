# 57–58 + 62 · Behavior Cloning from State and from Pixels, and Project 5: What Makes Pixel BC Work

**Section:** S06 · **Type:** Labs + project · **Duration:** ~40 min
**Companion notebooks:** `57_bc_from_state`, `58_bc_from_pixels`, `62_project_pixel_bc`

## Learning objectives
- Train BC from privileged state and from camera images on TableTop, and evaluate both in closed loop with confidence intervals.
- Explain why a low validation loss doesn't predict closed-loop success.
- Identify the ingredients of a working pixel policy by ablation, one ingredient at a time.

---

## 1. BC from state (notebook 57)
An MLP maps a 20-D privileged state (gripper pose, cube and goal relative to the gripper, task one-hot) to the expert's 7-D action. 200 expert episodes, 8,442 pairs, trained in seconds.

| Metric | Value |
|---|---|
| Validation MSE (normalized) | 0.07 |
| Closed-loop success (100 episodes) | **50%** (CI 40–60%) |
| Pick / stack / place | 97% / 34% / 15% |

A good loss, a mediocre robot. The failures are not random: they are the long tasks, and every failed episode runs into the step limit. This gap is the subject of lecture 59.

## 2. BC from pixels (notebook 58)
Two cameras (front + wrist), a ResNet-18 encoder per camera, the robot state, one action per step (the expert's raw end-effector deltas), MSE loss. "Single" scene (one cube + one tray, pick or place), 450 demos.

**Measured:** **76%** (CI 63–86%): pick 84%, place 68%. Pixels work, but not reliably. The video shows four new layouts (3 of 4 succeed).

## 3. Project 5: which ingredients matter (notebook 62)
Start from a policy that works and remove one ingredient at a time. Same data, same encoder, same 50 evaluation layouts per row:

| Variant | Success (95% CI) |
|---|---|
| **Full recipe:** 16-step chunk, relative targets, 2 cameras, augmentation | **100%** (93–100%) |
| Classic BC: 1 step, raw delta actions | 16% (8–29%) |
| 1 step, relative targets | 12% (6–24%) |
| 16-step chunk, raw delta actions | 96% (87–99%) |
| Full recipe, front camera only | 100% |
| Full recipe, wrist camera only | 98% |
| Full recipe, 50 demos instead of 450 | 66% (52–78%) |
| Full recipe, no augmentation | 100% |

How to read it:
- **The chunk is what matters here.** Removing it drops success from ~100% to 12–16%, whatever the action representation. Swapping relative targets for raw deltas while keeping the chunk barely matters (96%).
- **The cameras are redundant on this task:** either one alone is enough. Data quantity matters (450 → 50 demos: 100% → 66%). Augmentation doesn't help in simulation, where the images never change; it earns its keep on a real camera.
- **Why does the 1-step row fail so badly here (12–16%) when notebook 58's 1-step policy gets 76%?** The loss. Notebook 62 uses the L1 loss of the S08 heads (ACT uses L1 too); notebook 58 uses MSE. In our demonstrations 38% of all actions are exactly zero, because the expert pauses for a fixed number of steps while the gripper closes. A single frame can't tell how long the pause has lasted, so L1, which predicts the *median*, answers "don't move", and the arm freezes. MSE predicts the *mean*, a small upward move that eventually escapes. We checked by changing only the loss: 1-step with relative targets scores **12% with L1 vs 94% with MSE**; with a 16-step chunk both reach 100%. (Full table: S08 lecture 73.)
- So the honest summary is: **single-step policies are fragile to hidden state in the demonstrator, chunks are not.** A chunk that starts during the pause already contains the lift that follows it.

## 4. Debugging habit for every BC project
1. Evaluate in closed loop, with a CI. Never stop at the validation loss.
2. Watch the failures. "Frozen arm" and "time-outs" have different causes than "wrong place".
3. Compare the *distribution* of predicted actions with the labels (for example, the fraction of zero moves). A big mismatch points at the loss or the decoding, not at the data.
4. Ablate one ingredient at a time from a working recipe, not from a broken one.

## Key takeaways
- Low offline loss ≠ a working robot; measure success in closed loop.
- On TableTop, action chunking turns pixel BC from ~15% into ~100%; cameras, augmentation and the action representation matter much less.
- The training loss interacts with the demonstrator: L1 (median) and argmax decoding (mode) freeze on timed pauses; MSE (mean) and chunks don't.

## References
- Pomerleau 1989 (ALVINN); Zhao et al. 2023 (ACT, L1 + chunks); Chi et al. 2023 (Diffusion Policy, relative targets)
- Mandlekar et al. 2021 (robomimic: what matters in learning from offline human demonstrations)
