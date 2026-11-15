# 49–51 · Off-Policy Actor-Critic (DDPG → TD3 → SAC) and Domain Randomization

**Section:** S05 · **Type:** Math + labs · **Duration:** ~45 min
**Companion notebooks:** `50_sac_from_scratch`, `51_domain_randomization`

---

## 1. Why off-policy
On-policy methods discard data after each update. Off-policy methods store every transition in a **replay buffer** and reuse it many times, which is essential when samples are expensive (real robots).

## 2. From Q-learning to continuous actions
$\max_a Q(s,a)$ is intractable for continuous $a$, so learn an actor $\mu_\theta(s) \approx \arg\max_a Q(s,a)$.
- **DDPG (Lillicrap 2016):** deterministic actor, critic $Q_\phi$, target networks, exploration noise. Actor gradient: $\nabla_\theta Q_\phi(s,\mu_\theta(s))$.
- **TD3 (Fujimoto 2018):** fixes Q overestimation with (i) **twin critics, use the min**, (ii) delayed actor updates, (iii) target policy smoothing.
- **SAC (Haarnoja 2018):** stochastic actor + **maximum-entropy** objective.

## 3. Soft Actor-Critic
Objective:
$$J(\pi) = \sum_t\mathbb E\big[r_t + \alpha\,\mathcal H(\pi(\cdot\mid s_t))\big]$$
Soft Bellman target with twin critics:
$$y = r + \gamma(1-d)\Big(\min_{i=1,2}Q_{\bar\phi_i}(s',a') - \alpha\log\pi(a'\mid s')\Big),\quad a'\sim\pi(\cdot\mid s')$$
Actor loss (reparameterized): $\mathbb E\big[\alpha\log\pi(a\mid s) - \min_i Q_{\phi_i}(s,a)\big]$, $a = \tanh(\mu + \sigma\epsilon)$.

**Tanh-squashing correction:** $\log\pi(a) = \log\mathcal N(u) - \sum\log(1-\tanh^2 u)$, computed stably as $2(\log 2 - u - \mathrm{softplus}(-2u))$. Forgetting this is the classic SAC bug.

**Automatic temperature:** minimize $-\alpha(\log\pi + \bar{\mathcal H})$ with target entropy $\bar{\mathcal H} = -\dim(\mathcal A)$. α starts high (explore) and decays as the policy becomes confident; the notebook plots it.

Polyak target update: $\bar\phi \leftarrow (1-\tau)\bar\phi + \tau\phi$, τ = 0.005.

## 4. The sample-efficiency gap (notebook 50)
On the MuJoCo pendulum swing-up (notebook 50), SAC first exceeds a return of −300 after **5,000** environment steps; PPO needs **112,640**, a **22.5× gap**. Both plateau around −215 (the swing-up ceiling for this torque limit). That ratio is why off-policy RL is the choice for real-robot learning (e.g. SERL, RL fine-tuning on hardware), and why PPO dominates when a GPU simulator makes samples free.

## 5. Sim-to-real and domain randomization (notebook 51)
**Reality gap:** the simulator's masses, friction, latencies, sensor noise and visuals differ from reality, and a policy tuned to one simulator overfits it.

**Domain randomization (Tobin 2017; Peng 2018; OpenAI Dactyl 2019):** train on a *distribution* of simulators
$$\max_\pi\ \mathbb E_{\xi\sim p(\xi)}\big[J(\pi;\xi)\big]$$
so reality looks like one more sample. Here ξ = pole mass and length, ±30% resampled each episode (rebuilding the MJCF is ~1 ms).

**The experiment:** both SAC policies are evaluated on a 6 × 5 grid of "real" pendulums, including parameters outside the training band.
- The fixed-sim policy is best (or equal) at the nominal point and degrades away from it.
- The randomized policy gives up a little at the nominal point and is better across the band and in the worst case.
- Beyond the band, both degrade: DR buys robustness only where it randomized.

What to randomize for manipulation (S13): object mass and friction, actuator gains, latency, camera pose, lighting, textures (visual DR, S03 notebook 25), and action noise.

## Key takeaways
- Off-policy actor-critic reuses data; twin-min critics fight overestimation.
- SAC = soft Bellman backup + reparameterized squashed Gaussian + automatic entropy temperature.
- Domain randomization trades nominal performance for robustness across the randomized range.

## References
- Lillicrap et al. 2016 (DDPG); Fujimoto et al. 2018 (TD3); Haarnoja et al. 2018 (SAC)
- Tobin et al. 2017; Peng et al. 2018 (dynamics randomization)
