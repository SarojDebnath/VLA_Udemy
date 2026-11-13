# 13 · Information Theory: Entropy, Cross-Entropy, KL Divergence

**Section:** S02 · **Type:** Math · **Duration:** ~25 min
**Companion notebook:** `13_information_theory.ipynb`

## Learning objectives
- Compute entropy, cross-entropy and KL divergence and relate them.
- Derive the closed-form KL between two Gaussians.
- Contrast forward KL (mode-covering) with reverse KL (mode-seeking) and connect each to BC, VAEs and RL.

---

## 1. Entropy: uncertainty of a distribution
$$H(p) = -\sum_x p(x)\log p(x) \qquad (\text{nats if } \log = \ln)$$
- 0 for a certain outcome; $\log K$ for a uniform distribution over $K$ outcomes.
- **In RL:** SAC maximizes reward + $\alpha H(\pi)$ to keep exploring (S05).
- **In VLAs:** low action-token entropy means a confident policy, and spikes can flag out-of-distribution states (S15).

## 2. Cross-entropy and KL
$$H(p,q) = -\sum_x p(x)\log q(x),\qquad
D_{KL}(p\|q) = \sum_x p(x)\log\frac{p(x)}{q(x)} \ge 0$$
$$H(p,q) = H(p) + D_{KL}(p\|q)$$
Since $H(p)$ (the data) is fixed, **minimizing cross-entropy = minimizing KL from the data to the model = maximum likelihood**.
KL isn't symmetric: $D_{KL}(p\|q) \ne D_{KL}(q\|p)$.

## 3. KL between Gaussians (closed form)
$$D_{KL}\big(\mathcal N(\mu_1,\sigma_1^2)\,\|\,\mathcal N(\mu_2,\sigma_2^2)\big) = \log\frac{\sigma_2}{\sigma_1} + \frac{\sigma_1^2 + (\mu_1-\mu_2)^2}{2\sigma_2^2} - \frac12$$
This term appears in the VAE/CVAE loss (ACT, S08), in PPO's trust region (S05), and in the RSSM world model (S14). The notebook checks it against a Monte Carlo estimate.

## 4. Forward vs reverse KL
Fit a single Gaussian $q$ to a two-mode target $p$:
- **Forward** $D_{KL}(p\|q)$, i.e. maximum likelihood: $q$ must put mass wherever $p$ does, so it **covers both modes** and puts its peak *between* them. This is behavior cloning with a unimodal head.
- **Reverse** $D_{KL}(q\|p)$: $q$ is penalized for putting mass where $p$ has none, so it **locks onto one mode**. This is variational inference, and RL-style policy improvement.

| Notebook fit | μ | σ |
|---|---|---|
| Forward KL | ≈ 0 | ≈ 2.1 (covers both) |
| Reverse KL | ≈ 2 | ≈ 0.5 (one mode) |

Neither is "right" for robots. We want the *full* multimodal distribution, which is why expressive heads (diffusion, flow, tokens) beat unimodal Gaussians.

## 5. ELBO preview
For latent-variable models $p(a) = \int p(a\mid z)p(z)\,dz$:
$$\log p(a) \ge \mathbb{E}_{q(z\mid a)}[\log p(a\mid z)] - D_{KL}\big(q(z\mid a)\,\|\,p(z)\big)$$
Reconstruction minus KL. This is the training objective of ACT's CVAE (S08) and the basis of diffusion models.

## Key takeaways
- Cross-entropy = entropy + KL, so training with cross-entropy is maximum likelihood.
- Gaussian KL has a closed form used in VAEs, PPO and world models.
- Forward KL covers modes; reverse KL seeks one mode.

## References
- Cover & Thomas, *Elements of Information Theory*, Ch. 2
- Kandapagari, *Action Models for Robot Learning*, §3.2, Appendix B
