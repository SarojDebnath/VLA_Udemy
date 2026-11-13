# 17 · Three Loss Families: Regression, Classification, Denoising

**Section:** S02 · **Type:** Theory + math · **Duration:** ~25 min
**Companion notebook:** `17_loss_families.ipynb`

## Learning objectives
- Write the three losses used by every action head in the course.
- Predict how each one behaves on multimodal demonstrations.
- Relate them to concrete models: BC-MLP, RT-1/OpenVLA, Diffusion Policy/π0.

---

## 1. Regression (MSE / L1)
$$\mathcal L_{\text{reg}} = \mathbb{E}\,\|a - f_\theta(o)\|^2$$
- Gaussian maximum likelihood with fixed variance (lecture 12). Predicts the **conditional mean**.
- One forward pass, deterministic, smooth.
- Fails on multimodal data: the notebook's 10 samples per observation all land at the mean ≈ 0, between the two demonstrated bands.
- Used by: plain behavior cloning, ACT's decoder (L1, made multimodal by the CVAE latent).

## 2. Classification (cross-entropy over bins)
Discretize each action dimension into $K$ bins, then:
$$\mathcal L_{\text{CE}} = -\mathbb{E}\,\log p_\theta(k^*(a)\mid o)$$
- Represents **any** 1-D distribution, including multiple modes.
- Resolution limited to the bin width; dimensions are usually predicted independently or autoregressively.
- Used by: RT-1, RT-2, OpenVLA (256 bins per dimension), FAST tokens in π0-FAST.

## 3. Denoising (diffusion / flow)
Corrupt the action with noise at level $t$ and train a network to undo it:
$$\mathcal L_{\text{diff}} = \mathbb{E}_{a,t,\epsilon}\,\big\|\epsilon - \epsilon_\theta(\sqrt{\bar\alpha_t}\,a + \sqrt{1-\bar\alpha_t}\,\epsilon,\; o,\; t)\big\|^2$$
- Still an MSE, but on *noise*, not on the action. The model learns the **score** (gradient of log-density), so sampling recovers the full distribution.
- Continuous, multimodal, high-dimensional (whole action chunks).
- Costs several network evaluations per sample (50 in the notebook; DDIM and flow matching cut that to ~10).
- Used by: Diffusion Policy, Octo, π0 (flow-matching variant), GR00T.

## 4. Side-by-side (notebook results)
| Loss | Samples at an ambiguous observation | Cost per action |
|---|---|---|
| MSE | all at the mean (wrong) | 1 pass |
| Cross-entropy, 64 bins | both bands, quantized | 1 pass (+ sampling) |
| Denoising, 50 steps | both bands, continuous; a few stragglers | 50 passes |

## 5. Choosing in practice
- **Unimodal, precise, fast** → regression (+ chunking).
- **Language-model backbone, discrete tokens** → cross-entropy (it reuses the LLM's vocabulary head).
- **Dexterous, high-frequency, multimodal** → diffusion or flow.

Sections S08, S10 and S11 build each of these heads on TableTop and compare them under one evaluation harness.

## Key takeaways
- Regression predicts the mean; classification predicts a histogram; denoising learns to sample the full distribution.
- Every VLA action head is one of these three, or a hybrid of them.
- The trade-off is expressiveness vs inference cost vs precision.

## References
- Ho et al. 2020, *DDPM*; Chi et al. 2023, *Diffusion Policy*
- Brohan et al. 2022, *RT-1*; Kim et al. 2024, *OpenVLA*
- Kandapagari, *Action Models for Robot Learning*, §3.4, Exercise 3.x.3
