# 61 · A Glance at IRL and Adversarial Imitation, and Choosing Between BC, IRL and RL

**Section:** S06 · **Type:** Theory · **Duration:** ~20 min

## Learning objectives
- Explain inverse reinforcement learning and the GAIL objective.
- Know when IRL or adversarial imitation is worth its cost.
- Choose a learning signal for a new robot task.

---

## 1. Inverse reinforcement learning
BC copies *actions*. IRL infers the **reward** that makes the expert look optimal, then runs RL on it:
$$\text{find } r \text{ such that } \pi^* = \arg\max_\pi \mathbb E_\pi\Big[\sum_t r(s_t,a_t)\Big]$$
- The problem is ill-posed (r = 0 explains everything), so it needs regularization: max-margin (Abbeel & Ng 2004) or maximum entropy (Ziebart 2008):
$$p(\tau) \propto \exp\Big(\sum_t r_\psi(s_t,a_t)\Big)$$
- Benefit: a reward transfers to new dynamics and lets RL improve beyond the demonstrator; it also handles compounding error, because RL trains on its own states.
- Cost: RL in the inner loop.

## 2. GAIL: Generative Adversarial Imitation Learning (Ho & Ermon 2016)
Skip the explicit reward. A discriminator $D_\psi(s,a)$ tells expert from policy pairs; the policy is trained with RL to fool it:
$$\min_\pi\max_D\ \mathbb E_{\pi^*}[\log D(s,a)] + \mathbb E_{\pi}[\log(1-D(s,a))] - \lambda\mathcal H(\pi)$$
The policy's reward is $-\log(1-D(s,a))$. This matches **occupancy measures** (state-action distributions), not individual actions, so it fixes covariate shift by construction. Variants: AIRL (recovers transferable rewards), DAC, and AMP for character animation.

## 3. Why VLAs don't use them (yet)
- Adversarial training is unstable, and RL in the loop is expensive for image-based, high-dimensional policies.
- Large, diverse demonstration sets plus expressive generative policies (diffusion, flow, autoregressive tokens) get much of the benefit with plain supervised learning.
- Learned rewards reappear in **RL fine-tuning of VLAs** (S16): success detectors and VLM-based reward models are IRL-flavored ideas.

## 4. Choosing a method
| Situation | Choose |
|---|---|
| Many good demonstrations, short horizon | **BC** (with chunking, diffusion or flow heads) |
| An expert you can query on any state (simulator, scripted) | **DAgger** |
| Human demonstrators, cheap perturbations | **DART / interventions** |
| A reliable reward and a fast simulator | **RL** (PPO in parallel sim, SAC on hardware) |
| Few demonstrations, a simulator, and a need to exceed the demonstrator | **IRL / GAIL / demo-augmented RL** |
| Broad generalization across objects and instructions | **VLA pretraining + BC fine-tuning**, then optional RL fine-tuning |

## Key takeaways
- IRL learns a reward; GAIL matches state-action distributions adversarially.
- Both fix compounding error but cost RL in the loop.
- Modern VLAs = large-scale BC; RL and reward learning come back for fine-tuning.

## References
- Abbeel & Ng 2004; Ziebart et al. 2008; Ho & Ermon 2016 (GAIL); Fu et al. 2018 (AIRL)
