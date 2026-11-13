# 75–78 · Diffusion Models and Diffusion Policy

**Section:** S08 · **Type:** Math · **Duration:** ~60 min
**Companion notebooks:** `75_ddpm_math`, `77_diffusion_2d_toy`, `78_diffusion_policy`

## Learning objectives
- Derive the DDPM forward process, its closed form, and the ε-prediction loss.
- Understand ancestral (DDPM) and deterministic (DDIM) sampling, score matching, and classifier-free guidance.
- Map the recipe onto robot action chunks: Diffusion Policy.

---

## 1. Forward process (notebook 75)
Gradually add Gaussian noise with schedule $\beta_t$:
$$q(x_t\mid x_{t-1}) = \mathcal N\big(\sqrt{1-\beta_t}\,x_{t-1},\ \beta_t I\big)$$
With $\alpha_t = 1-\beta_t$ and $\bar\alpha_t = \prod_{s\le t}\alpha_s$, any step has a **closed form**:
$$q(x_t\mid x_0) = \mathcal N\big(\sqrt{\bar\alpha_t}\,x_0,\ (1-\bar\alpha_t)I\big),\qquad x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon$$
The notebook checks that 500 iterated steps and the closed form give the same mean and variance. As $\bar\alpha_T \to 0$, $x_T \sim \mathcal N(0,I)$.

Schedules: linear (DDPM) or **cosine** (Nichol & Dhariwal 2021; used by Diffusion Policy), which keeps more signal at mid t.

## 2. Reverse process and training
The true posterior is Gaussian:
$$q(x_{t-1}\mid x_t,x_0) = \mathcal N(\tilde\mu_t, \tilde\beta_t I),\quad \tilde\mu_t = \frac{\sqrt{\bar\alpha_{t-1}}\beta_t}{1-\bar\alpha_t}x_0 + \frac{\sqrt{\alpha_t}(1-\bar\alpha_{t-1})}{1-\bar\alpha_t}x_t$$
Substituting $x_0 = (x_t - \sqrt{1-\bar\alpha_t}\,\epsilon)/\sqrt{\bar\alpha_t}$:
$$\tilde\mu_t = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1-\bar\alpha_t}}\epsilon\Big)$$
(verified numerically in the notebook). So the network only needs to predict the noise:
$$\mathcal L = \mathbb E_{x_0,t,\epsilon}\big\|\epsilon - \epsilon_\theta(x_t, t)\big\|^2$$
This is a simplified (reweighted) ELBO. Training is plain regression, stable, and needs no adversary.

## 3. Score-matching view
$\epsilon_\theta(x_t,t) \approx -\sqrt{1-\bar\alpha_t}\,\nabla_{x_t}\log q(x_t)$. The network learns the **score** (the direction toward higher data density) at every noise level, and sampling follows it uphill from noise. Multimodality comes for free: near each mode, the score points to that mode.

## 4. Sampling
- **DDPM (ancestral):** $x_{t-1} = \tilde\mu_\theta(x_t,t) + \sqrt{\beta_t}\,z$, T steps (100 in the toy, 1000 for images).
- **DDIM (Song et al. 2021):** deterministic, skips steps:
$$\hat x_0 = \frac{x_t - \sqrt{1-\bar\alpha_t}\,\epsilon_\theta}{\sqrt{\bar\alpha_t}},\qquad x_{t'} = \sqrt{\bar\alpha_{t'}}\,\hat x_0 + \sqrt{1-\bar\alpha_{t'}}\,\epsilon_\theta$$
The same trained network works with 10 or even 3 steps (notebook 77 compares 100 / 10 / 3).

## 5. Classifier-free guidance
Train one network both conditionally and unconditionally (drop the condition 20% of the time), then sample with
$$\tilde\epsilon = (1+w)\,\epsilon_\theta(x_t,t,c) - w\,\epsilon_\theta(x_t,t,\varnothing)$$
Larger w → samples more faithful to c and less diverse (the notebook shows the spread shrinking with w). VLAs use the same trick to strengthen language conditioning.

## 6. Diffusion Policy (Chi et al. 2023; notebook 78)
Apply the recipe to **action chunks** conditioned on observations:
$$\mathcal L = \mathbb E\big\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}A^0 + \sqrt{1-\bar\alpha_t}\epsilon,\ t,\ O)\big\|^2,\qquad A^0 = a_{t:t+H}$$
Design choices (all in `vlacourse/chunk_policies.py`):
- **1-D temporal U-Net** over the chunk (channels = action dims, length H = 16), conv kernels of size 5, GroupNorm, Mish.
- **FiLM conditioning:** the observation embedding + diffusion-step embedding produce per-channel scale and shift in every residual block.
- **Observation:** ResNet-18 per camera (front + wrist), state, task (language arrives in S09).
- **Training:** 100-step cosine schedule, ε-prediction, AdamW, random-shift augmentation.
- **Inference:** DDIM with 10 steps; execute 8 of 16 (receding horizon).

Why it works so well on real robots: it represents multimodal action distributions, generates temporally consistent chunks, and is stable to train. The cost is several network evaluations per chunk, which flow matching (next lecture) reduces.

**Measured (notebook 78):** 6.6M parameters (U-Net 5.9M); "single" scene, 450 demos, 50 new layouts: **100%** success (95% CI 93–100%) with DDIM 10 steps, executing 8 of 16. The video shows four rollouts.

**How many denoising steps do you need? (notebook 81, 40 episodes each):**
| DDIM steps (network calls per chunk) | 1 | 2 | 4 | 10 |
|---|---|---|---|---|
| Success | 0% | 0% | 72% | **100%** |

Below about 4 steps the sample is still mostly noise, and the robot fails completely. That step count is the price of diffusion, and it sets the control rate you can afford.

## Key takeaways
- Forward: $x_t = \sqrt{\bar\alpha_t}x_0 + \sqrt{1-\bar\alpha_t}\epsilon$; train by predicting ε.
- DDIM gives deterministic, few-step sampling with the same network; CFG trades diversity for adherence.
- Diffusion Policy = conditional U-Net denoiser over action chunks + receding-horizon execution.

## References
- Ho et al. 2020 (DDPM); Song et al. 2021 (DDIM, score SDE); Ho & Salimans 2022 (CFG)
- Chi et al. 2023, *Diffusion Policy*
- Kandapagari, *Action Models for Robot Learning*, §10.1–10.2
