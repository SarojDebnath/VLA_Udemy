# 46–48 · Policy Gradients, GAE, and PPO

**Section:** S05 · **Type:** Math · **Duration:** ~45 min
**Companion notebooks:** `46_reinforce_policy_gradient`, `48_ppo_from_scratch`

## Learning objectives
- Derive the policy gradient theorem and the REINFORCE estimator.
- Explain why baselines and advantages reduce variance without adding bias.
- Derive GAE and PPO's clipped surrogate objective, and read PPO's diagnostics.

---

## 1. Optimizing the policy directly
Maximize $J(\theta) = \mathbb E_{\tau\sim\pi_\theta}[G(\tau)]$. Using the log-derivative trick, $\nabla p = p\,\nabla\log p$:
$$\nabla_\theta J = \mathbb E_{\tau}\Big[\sum_t \nabla_\theta\log\pi_\theta(a_t\mid s_t)\;G_t\Big]$$
The dynamics $P$ drop out, so **no model is needed**: we only need log-probabilities of our own actions. That's why policy gradients work with simulators, real robots, and (in S16) with VLAs whose action heads give log-probabilities.

## 2. REINFORCE and variance
The Monte Carlo estimate with returns-to-go $G_t$ is unbiased but very noisy, because every action is credited with all future luck.

**Baseline:** subtracting any state-dependent $b(s_t)$ keeps the estimate unbiased:
$$\mathbb E_{a\sim\pi}[\nabla\log\pi(a\mid s)\,b(s)] = b(s)\,\nabla\sum_a\pi(a\mid s) = 0$$
With $b = V_\phi(s)$, the weight becomes the **advantage** $A_t = G_t - V(s_t)$: "how much better than expected was this action?" The notebook measures per-sample gradient variance with and without a learned baseline on the MuJoCo cart-pole.

## 3. Generalized Advantage Estimation
TD error: $\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$ (low variance, biased by $V$).
GAE blends n-step estimates:
$$\hat A_t^{GAE(\gamma,\lambda)} = \sum_{l\ge0}(\gamma\lambda)^l\,\delta_{t+l}$$
λ = 0 → one-step TD (low variance, high bias); λ = 1 → Monte Carlo (high variance, unbiased). PPO default: λ = 0.95. Computed backwards in one pass; reset at episode ends; **bootstrap at truncation** (time limit ≠ failure).

## 4. Trust regions and PPO
Large policy steps can destroy performance, because the data was collected by the *old* policy. TRPO constrains $\mathrm{KL}(\pi_{old}\|\pi_\theta) \le \delta$. PPO approximates that with a clipped importance ratio:
$$r_t(\theta) = \frac{\pi_\theta(a_t\mid s_t)}{\pi_{\theta_{old}}(a_t\mid s_t)},\qquad
L^{CLIP} = \mathbb E_t\big[\min\big(r_t\hat A_t,\ \mathrm{clip}(r_t, 1-\epsilon, 1+\epsilon)\,\hat A_t\big)\big]$$
Once the ratio leaves $[1-\epsilon, 1+\epsilon]$ in the helpful direction, the gradient is zero, so there's no incentive to move further.

Full loss: $-L^{CLIP} + c_v (V_\phi - \hat G)^2 - c_e\,\mathcal H[\pi]$.

## 5. PPO in practice (notebook 48)
Recipe: collect 2048 steps with the current policy → GAE → 10 epochs of minibatch (64) updates → discard the data (on-policy). Normalize advantages, clip gradients (0.5), Adam 3e-4, Gaussian policy with a learned state-independent log-std for continuous actions.

What to watch:
- **Return / success curve:** the goal.
- **Approximate KL** $\mathbb E[(r-1) - \log r]$: if it spikes (> 0.05), the policy jumped. Return collapses usually follow a KL spike.
- **Clip fraction:** the share of samples hitting the clip; persistently high means the learning rate is too big.

Results (notebook 48, CPU): cart-pole reaches a 475+ average return after ~35k environment steps (~3 min); the 7-DoF Panda learns to reach random targets from state in 3×10⁵ steps (~8 min), with 20/20 targets reached by the deterministic policy. REINFORCE (notebook 46) also solves cart-pole, and the learned value baseline cuts per-sample gradient variance by roughly an order of magnitude, giving faster and more stable learning.

## 6. Why PPO matters for VLAs
- Locomotion controllers under humanoid VLAs (System 0/1) are trained with PPO in massively parallel sim (MJX, Isaac).
- RL fine-tuning of VLAs (π-RL, RIPT, VLA-RL; S16) uses PPO/GRPO-style clipped objectives on action-token or flow log-likelihoods.
- PPO's sample hunger is fine when samples are cheap (simulation) and painful on real robots, which motivates off-policy methods (lecture 49).

## Key takeaways
- $\nabla J = \mathbb E[\nabla\log\pi\cdot A]$: model-free, but high variance.
- Baselines and GAE cut variance; bootstrap through truncations.
- PPO = clipped importance ratio ≈ trust region; monitor KL and clip fraction.

## References
- Williams 1992, *REINFORCE*; Sutton et al. 2000, *Policy Gradient Methods*
- Schulman et al. 2016, *GAE*; Schulman et al. 2017, *PPO*
- Kandapagari, *Action Models for Robot Learning*, §7.2–7.3, Exercise 7.x.1
