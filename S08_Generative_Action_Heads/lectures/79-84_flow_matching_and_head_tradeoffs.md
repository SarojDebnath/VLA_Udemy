# 79–84 · Flow Matching, Rectified Flow, and Choosing an Action Head

**Section:** S08 · **Type:** Math + project · **Duration:** ~50 min
**Companion notebooks:** `80_flow_matching_2d`, `81_diffusion_to_flow`, `82_regression_vs_diffusion_vs_flow`, `83_receding_horizon`, `84_project_heads_comparison`

---

## 1. Flow matching (Lipman et al. 2023; Liu et al. 2023)
Define a straight path from noise $x_0 \sim \mathcal N(0,I)$ to data $x_1$:
$$x_\tau = (1-\tau)\,x_0 + \tau\,x_1,\qquad \frac{dx_\tau}{d\tau} = x_1 - x_0$$
Train a velocity field to match it:
$$\mathcal L_{FM} = \mathbb E_{x_1,x_0,\tau}\big\|v_\theta(x_\tau,\tau) - (x_1-x_0)\big\|^2$$
Sample by integrating the ODE $\dot x = v_\theta(x,\tau)$ from τ = 0 to 1 with Euler steps:
```python
x = torch.randn(n, d)
for k in range(n_steps):  x = x + v(x, k / n_steps) / n_steps
```
- Like diffusion, it's regression with no noise schedule or variance bookkeeping, and the target is a velocity instead of ε.
- It's equivalent to diffusion with a particular schedule; its paths are straighter, so **fewer steps** are needed.

## 2. The seam (notebook 80)
With two modes at x = ±2, the loss can't reach zero: on the vertical line x = 0, conditional velocities point both ways, and the learned *marginal* field is short and undecided there. Noise samples on either side flow to their own mode, so **the field splits the noise instead of averaging it**. After 10 Euler steps almost no samples end up between the modes; with a single step, the curved marginal paths are cut short and some samples land in between.

**Reflow / rectified flow:** regenerate (noise, sample) pairs with the trained model and retrain on them. The paths straighten, approaching one-step generation, at the cost of offline compute.

## 3. Conditional flow for actions
$$\mathcal L = \mathbb E\big\|v_\theta(A_\tau, \tau, o) - (A^1 - A^0)\big\|^2$$
with $A^1$ an action chunk. π0 uses exactly this with H = 50 and about 10 Euler steps at 50 Hz (S11). Our `FlowPolicy` keeps Diffusion Policy's U-Net and observation encoder and changes only the loss and the sampler (notebook 81).

**Measured (notebook 81): same network, same data, only the generative recipe differs** (40 episodes per cell):

| Network calls per chunk | 1 | 2 | 4 | 10 |
|---|---|---|---|---|
| Diffusion (DDIM) | 0% | 0% | 72% | 100% |
| **Flow matching (Euler)** | **95%** | 97% | 100% | 97% |
| Latency, diffusion | 19.5 ms | | | 74.9 ms |
| Latency, flow | 14.8 ms | | | 30.8 ms |

Flow matching's straight paths mean one Euler step already lands near the data: 95% with a single network call, where diffusion needs ten. Same quality at a fraction of the compute is why π0, SmolVLA and GR00T use flow-matching action heads. The video compares diffusion at 10 steps with flow at 1 step on the same episode.

## 4. Closed-loop comparison on the multimodal toy (notebook 82)
Same data, same MLP size, chunk H = 8, execute 4, success = reach the goal **without touching the obstacle**:
- **Regression:** averages the two routes; success **0.00**, collisions **1.00**.
- **Diffusion (DDIM, 10 steps):** success **0.78**. **Flow (Euler, 10 steps):** success **0.80**. Both commit to a side; the remaining ~20% are near-misses that clip the obstacle edge.

Also from notebook 80: with **1** Euler step every sample lands *between* the modes (fraction in the middle 1.00), with 2 steps 0.52, with 10 steps 0.02. Few-step sampling only works once paths are straight.
- The NFE sweep shows the generators are interchangeable in quality once given enough steps; flow tends to degrade more gracefully at very low NFE.

## 5. Receding-horizon window (notebook 83)
Execute k of H = 16 predicted actions, with and without a mid-episode perturbation (the cube is bumped 3 cm):
- k = 16: fewest network calls and smoothest motion, but blind to the bump.
- k = 1: maximally reactive, but the most calls and chunk-to-chunk mode switching.

**Measured (flow policy, 40 episodes per cell). It's not what most people predict:**
| Actions executed per chunk (k) | 1 | 2 | 4 | 8 | 16 |
|---|---|---|---|---|---|
| Nominal | 95% | 100% | 100% | 97% | 97% |
| **Cube bumped 3 cm at step 12** | **25%** | 15% | 12% | 55% | **62%** |
| Network calls per episode (approx.) | 2000 | 1000 | 500 | 250 | 125 |

Committing to long chunks survived the bump **better** than re-planning every step. The likely reasons (a good discussion question before revealing them):
- **The bump is small relative to the gripper** (a 3 cm shift against an 8 cm opening): finishing the planned grasp often still catches the cube.
- **Re-planning from a scene that changed mid-approach is out of distribution.** No demonstration ever had the cube jump while the gripper was descending, so the re-planned chunks are unreliable, and frequent re-planning keeps switching between them.
- Reactivity only pays off when the policy has *seen* recoveries. That's what DART/DAgger data (S06) and RL fine-tuning (S16) add.

So "re-plan more often" is not automatically "more robust". Execute 8 of 16 (Diffusion Policy's default) costs 16× fewer network calls than k = 1 and is the better trade here. The video shows the bumped episode with k = 1 and k = 16 side by side.

## 6. Choosing an action head (book §10.4–10.5)
| Head | Multimodal | Precision | Inference cost | Used in |
|---|---|---|---|---|
| Regression (single step) | ✗ | high when unimodal | 1 pass | classic BC |
| Chunked regression + ensembling | partial (via chunking) | high | 1 pass per step (ensembling) | — |
| CVAE (ACT) | via z (z = 0 at test) | high | 1 pass | ALOHA, ACT |
| Discrete tokens | ✓ | bin-limited | 7·H autoregressive passes | RT-1/2, OpenVLA |
| Diffusion | ✓ | high | ~10–100 passes | Diffusion Policy, Octo |
| Flow matching | ✓ | high | ~1–10 passes | π0, GR00T, SmolVLA |

## 7. Project 6 (notebook 84)
Single-step regression, chunked regression, ACT, Diffusion Policy and flow matching, all with the same ResNet-18 encoder, the same 450 training episodes ("single" scene) and the same 100 evaluation episodes, plus a robustness test with the front camera moved 5 cm (50 episodes). Results are saved to `results/project6_heads.json`.

**Measured:**
| Action head | Success (100 episodes, 95% CI) | Front camera moved 5 cm |
|---|---|---|
| Single-step regression | **7%** (3–14%) | 20% |
| Chunked regression (H = 16) | **100%** (96–100%) | 98% |
| ACT (CVAE transformer) | 96% (90–98%) | 100% |
| Diffusion Policy (DDIM 10) | 93% (86–97%) | 94% |
| Flow matching (Euler 4) | 99% (95–100%) | 100% |

How to read it honestly:
- **One gap is real: single-step vs everything else.** Chunking is the ingredient that matters on this task. The single-step row uses an L1 loss, which freezes on the demonstrator's timed waits; with MSE a single-step policy reaches 94% (lecture 73, "is it the chunk, or the loss?"). Chunked heads don't care which loss you use.
- **The four chunked heads are statistically indistinguishable** on nominal success: their intervals overlap (S12 teaches the test). Don't rank ACT, diffusion and flow from this table.
- **Plain chunked regression is at the top.** This task has one sensible way to do it, so there's no multimodality for a generative head to win on. Notebooks 71 and 82 show the other side: with two valid routes around an obstacle, regression collides 100% of the time and the generative heads don't. Choose the head for the *data* you have: multimodal human teleoperation needs a generative head; a scripted, unimodal expert doesn't.
- **What differs is cost:** flow reaches its accuracy in 1–4 network calls, diffusion needs about 10 (section 3).
- A 5 cm camera shift barely hurts these two-camera policies; the wrist camera moves with the gripper and is unaffected. S12 notebook 123 pushes harder.

The final video runs all five heads on the same episode, side by side.

## Key takeaways
- Flow matching = regress a straight-line velocity; sample with a few Euler steps.
- Generative heads represent multimodality; regression cannot.
- The head choice trades expressiveness, precision and inference cost; chunk execution trades reactivity against smoothness.

## References
- Lipman et al. 2023 (Flow Matching); Liu, Gong & Liu 2023 (Rectified Flow)
- Black et al. 2024 (π0)
- Kandapagari, *Action Models for Robot Learning*, §10.3–10.5, §13.3, Exercises 10.x and 13.x
