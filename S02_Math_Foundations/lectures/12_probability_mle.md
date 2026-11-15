# 12 · Probability for Policies: Random Variables, Gaussians, Maximum Likelihood

**Section:** S02 · **Type:** Math · **Duration:** ~25 min
**Companion notebook:** `12_probability_mle.ipynb`

## Learning objectives
- Treat a policy as a conditional distribution $\pi(a\mid o)$.
- Derive MSE as Gaussian maximum likelihood and cross-entropy as categorical maximum likelihood.
- Explain why MSE regression averages the modes of multimodal demonstrations.

---

## 1. Random variables and expectations
- A random variable $X$ with density $p(x)$; expectation $\mathbb{E}[f(X)] = \int f(x)p(x)\,dx \approx \frac1N\sum_i f(x_i)$ (Monte Carlo).
- Mean, variance: $\mu = \mathbb{E}[X]$, $\sigma^2 = \mathbb{E}[(X-\mu)^2]$.
- Conditional distributions: $p(a\mid o)$. **A policy is a conditional distribution over actions.**

## 2. The Gaussian
$$\mathcal{N}(x;\mu,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big)$$
Negative log-likelihood:
$$-\log p(x) = \tfrac12\log(2\pi\sigma^2) + \frac{(x-\mu)^2}{2\sigma^2}$$

## 3. Maximum likelihood estimation
Given demonstrations $\{(o_i,a_i)\}$, choose parameters that make the data most probable:
$$\theta^* = \arg\max_\theta \sum_i \log \pi_\theta(a_i\mid o_i)$$
- **Gaussian policy with fixed $\sigma$:** $-\log\pi = \frac{\|a - \mu_\theta(o)\|^2}{2\sigma^2} + \text{const}$, so **MLE = minimize MSE**.
- **Categorical policy over bins:** $-\log\pi = -\log p_\theta(k^*\mid o)$, so **MLE = minimize cross-entropy**. This is exactly how RT-1, RT-2 and OpenVLA train their action tokens.

## 4. The averaging trap
If half the demonstrators steer left ($a=-1$) and half steer right ($a=+1$) around an obstacle, the MSE-optimal prediction is
$$\mu^*(o) = \mathbb{E}[a\mid o] = 0$$
which goes **straight into the obstacle**. A unimodal Gaussian has no way to say "either left or right".

The notebook's categorical policy learns $p(a \mid o)$ with two peaks, and *sampling* from it picks one mode per rollout.

Fixes used in VLAs:
| Approach | Model | Course section |
|---|---|---|
| Discretize + cross-entropy | RT-1, RT-2, OpenVLA | S10 |
| Mixture of Gaussians | BC-RNN (robomimic) | S06 |
| Latent variable (CVAE) | ACT | S08 |
| Diffusion / flow | Diffusion Policy, Octo, π0 | S08, S11 |

## 5. Sampling vs the mean
With a multimodal $\pi$, you **sample** an action (or take the argmax mode), never the mean. Temperature and argmax decoding choices directly affect robot behavior (S10).

## Key takeaways
- A policy is $\pi(a\mid o)$; training is maximum likelihood.
- MSE = Gaussian MLE, and cross-entropy = categorical MLE.
- Unimodal regression averages modes, which drives most of the design of modern action heads.

## References
- Bishop, *Pattern Recognition and Machine Learning*, Ch. 1–2
