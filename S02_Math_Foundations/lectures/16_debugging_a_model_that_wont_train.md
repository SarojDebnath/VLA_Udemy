# 15–16 · The Training Loop and Debugging a Model That Won't Train

**Section:** S02 · **Type:** Theory (companion to labs 15 and 16) · **Duration:** ~20 min
**Companion notebooks:** `15_training_loop.ipynb`, `16_fault_drills.ipynb`

## Learning objectives
- Identify the six lines of a training loop that do real work, versus the plumbing around them.
- Recognize the loss-curve signature of five common faults.
- Apply a fixed checklist before changing any hyperparameter.

---

## 1. Anatomy of the loop
```python
for epoch in range(E):
    for x, y in loader:                 # 1 sample a batch
        pred = model(x)                 # 2 forward
        loss = loss_fn(pred, y)         # 3 compare with the target
        opt.zero_grad()                 # 4 clear old gradients
        loss.backward()                 # 5 backprop (chain rule, lecture 11)
        opt.step()                      # 6 update (lecture 14)
```
Everything else is plumbing: normalization statistics, train/val split, logging, checkpointing, device placement. **Most robot-learning bugs live in the plumbing, not in the six lines.**

Our task: learn the Panda's bias torques $c(q,\dot q)$ (gravity + Coriolis) from 20,000 MuJoCo samples. It's a real, smooth, 14→7-D function with a known ground truth.

## 2. The baseline curve
Train and validation loss both decrease monotonically and flatten, with validation slightly above training. Every drill below is read against this picture.

## 3. Five faults and their signatures
| Fault | What you see | Why |
|---|---|---|
| **LR far too large** (1.0) | loss explodes ($10^6$+), spikes, maybe NaN | step $\eta > 2/\lambda_{\max}$ |
| **LR far too small** (1e-8) | flat line, under 1% change | updates are ~zero |
| **Unnormalized inputs** (std ~50) | huge initial loss, slow anomalous decay | activations saturate, the effective curvature is huge |
| **Outlier labels** (5% of batches contain a 100× value) | periodic spikes; clipping removes them | squared error amplifies outliers |
| **Bimodal labels** (two demonstrators disagree) | plateau well above baseline that never goes down | MSE cannot fit two answers; the floor is the irreducible variance |

The last one connects straight to lecture 12: **a loss that plateaus high on clean code often means the data is multimodal, not that the model is too small.**

## 4. The checklist (run it in order)
1. **Overfit one batch.** The loss must go to ~0. If not, the bug is in the model or the loss.
2. **Check normalization statistics.** Are inputs and targets roughly zero-mean, unit-std? Were the stats computed on *train* only, and are they saved with the checkpoint? (OpenVLA's `unnorm_key` bug is this item.)
3. **Inspect the data.** Plot a few targets and look for outliers, unit mix-ups (degrees vs radians, mm vs m), and swapped dimensions.
4. **Gradient norms.** Log $\|g\|$. Spikes point to outliers or a learning rate that's too high.
5. **Learning-rate sweep.** Try 3–4 values across decades before anything else.
6. **Train vs validation gap.** A large gap means overfitting (more data, augmentation, smaller model); no gap and a high loss means underfitting or multimodal targets.
7. **Closed-loop test.** A low offline loss doesn't guarantee task success (S06). Always roll out.

## Key takeaways
- Six lines do the learning; the bugs hide in the plumbing.
- Each fault has a recognizable loss-curve signature.
- A high plateau on clean code is often multimodal data, not model capacity.

## References
- Karpathy, *A Recipe for Training Neural Networks* (2019)
- Kandapagari, *Action Models for Robot Learning*, §3.3, §3.5, Exercises 3.x.1–3.x.2
