# 05 · Compounding Error: The Central Difficulty of Learning to Act

**Section:** S01 · **Type:** Theory + math · **Duration:** ~20 min
**Companion notebook:** `05_compounding_error.ipynb`

## Learning objectives
- State the covariate-shift problem in imitation learning.
- Derive the intuition behind the $O(\epsilon T^2)$ behavior-cloning error bound, versus $O(\epsilon T)$ for i.i.d. prediction.
- See why open-loop replay of demonstrations fails, and why closed-loop policies and data coverage matter.

---

## 1. Training and test distributions differ
Behavior cloning trains a policy $\pi_\theta$ on states visited by the **expert**:

$$\theta^* = \arg\min_\theta \; \mathbb{E}_{s \sim d_{\pi^*}}\big[\ell(\pi_\theta(s), \pi^*(s))\big]$$

But at test time the policy visits states from **its own** distribution $d_{\pi_\theta}$. As soon as it makes a mistake, it reaches states the expert never visited, where its training gave it no guidance. This mismatch $d_{\pi_\theta} \neq d_{\pi^*}$ is called **covariate shift**.

## 2. The $\epsilon T^2$ argument (Ross & Bagnell, 2010)
Let the learner err with probability $\epsilon$ on states from the expert distribution, and let the episode length be $T$.
- Once it errs, it may leave the expert distribution and stay off it for the rest of the episode. Each mistake can then cost up to $T$ more steps.
- The first mistake happens around step $1/\epsilon$. Summing the expected cost over the possible times of the first error:

$$J(\pi_\theta) \le J(\pi^*) + \epsilon\,T^2 \quad (\text{up to constants})$$

- Compare supervised learning with independent samples: total errors $\approx \epsilon T$.

**Interpretation:** doubling the episode length quadruples the worst-case cost. Long-horizon tasks (folding, cooking) are exactly where naive cloning breaks.

## 3. The toy experiment
In the notebook, a "lane keeper" has a 1% chance per step of a small mistake. Inside the demonstrated region $|x| \le 0.05$ it corrects like the expert. Outside, it has no data and acts randomly. Measured steps spent off-distribution grow super-linearly with $T$, between the $\epsilon T$ and $\epsilon T^2/2$ lines on a log-log plot.

## 4. Open-loop replay is the extreme case
The TableTop experiment records the expert's actions and then **replays them blindly** after moving the cube a few centimeters:

| Cube displacement | Closed-loop expert | Open-loop replay |
|---|---|---|
| 0 cm | 100% | 100% |
| 2 cm | 100% | ~75% |
| 4 cm | 100% | ~45% |

The recorded actions are perfect for the world they were recorded in, and wrong for a slightly different one. A policy must **condition on the current observation**. This is also why a learned policy that ignores its camera input (a common bug!) behaves like replay.

## 5. The toolbox against compounding error
| Remedy | Idea | Where in the course |
|---|---|---|
| Closed-loop execution | re-observe every step | everywhere |
| **DAgger** | query the expert on the learner's own states, aggregate | S06 |
| **Data coverage / noise injection** | demonstrate recoveries; perturb the expert (DART) | S06, S13 |
| **Action chunking** | predict $H$ actions at once, so there are fewer independent decisions | S08 (ACT) |
| **Temporal ensembling / receding horizon** | smooth and re-plan chunks | S08 |
| **RL fine-tuning** | learn from the policy's own mistakes | S05, S16 |

## Key takeaways
- The policy's own errors move it off the training distribution (covariate shift).
- Behavior cloning's worst-case cost grows as $\epsilon T^2$, against $\epsilon T$ for i.i.d. prediction.
- Replaying actions open-loop fails under tiny scene changes; policies must use the current observation.
- DAgger, data coverage, chunking and RL are the standard counter-measures.

## References
- Ross & Bagnell 2010, *Efficient Reductions for Imitation Learning*
- Ross, Gordon & Bagnell 2011, *DAgger* (AISTATS)
- Laskey et al. 2017, *DART: Noise Injection for Robust Imitation Learning*
- Kandapagari, *Action Models for Robot Learning*, §1.1, §6.3
