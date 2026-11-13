# 133–140 · World Models and Model-Based Control

**Section:** S14 World Models · **Type:** Theory + math + labs · **Duration:** ~60 min
**Notebooks:** `134_mppi_with_mujoco`, `136_mini_dreamer`, `137_planning_horizon_vs_model_error`, `138_stochastic_latent`

---

## 133 · What a world model is, really
A learned (or given) model of how the world evolves under actions:
$$\hat s_{t+1} \sim p_\theta(s_{t+1}\mid s_t, a_t)\quad\text{(or in a latent space, or in pixels)}$$
Uses:
1. **Planning:** search over action sequences inside the model (MPC).
2. **Imagination training:** train a policy on imagined rollouts (Dreamer).
3. **Evaluation and data:** a learned simulator to test policies or to generate experience.
4. **Representation:** predicting the future forces features to encode physics.

## 134 · The perfect world model: MuJoCo itself
**Model-predictive path integral control (MPPI):** sample K perturbed control sequences around the current plan, roll each out in the model, weight by $e^{-C_k/\lambda}$, average, execute the first action, shift, repeat.
$$U \leftarrow \sum_k w_k U_k,\qquad w_k = \frac{e^{-C_k/\lambda}}{\sum_j e^{-C_j/\lambda}}$$
With `mujoco.rollout` (multithreaded C) the planner evaluates 256 rollouts × 30 steps per control step and swings the pendulum up **with no training at all**. It also tolerates moderate model error because it replans every step: closed-loop feedback again (S01). The costs are compute at every step and needing an accurate model, which is exactly what we lack for contact-rich, deformable, real-world scenes.

**Measured (notebook 134):**
| | |
|---|---|
| MPPI, exact model, 256 × 30 rollouts | return **−366**, 200 control steps in 2 s (6.7 ms per step), **zero training** |
| SAC after 20k env steps (S05 50) | about −215: learning eventually beats planning with a short horizon |
| Plan with the nominal model while the true pendulum mass is 0.5 → 1.6 kg | wrong-model returns within 1–25 points of exact-model returns at every mass: replanning absorbs the error |
| Samples × horizon sweep | 16×10: −939 · 64×30: −368 · 256×30: −366. **Horizon matters far more than samples**; beyond about 64 samples you only pay compute |

## 135 · Latent dynamics: RSSM and Dreamer
The Recurrent State-Space Model (Hafner et al.) splits the state into
- a **deterministic** recurrent part $h_t = f(h_{t-1}, z_{t-1}, a_{t-1})$ (GRU), which carries memory, and
- a **stochastic** part $z_t$ with a **prior** $p(z_t\mid h_t)$ (for imagining) and a **posterior** $q(z_t\mid h_t, o_t)$ (when an observation is available).

Training objective (an ELBO over sequences):
$$\mathcal L = \sum_t \Big[\underbrace{-\log p(o_t\mid h_t,z_t)}_{\text{reconstruct}} - \log p(r_t\mid h_t,z_t) + \underbrace{\beta\,\mathrm{KL}\big(q(z_t\mid h_t,o_t)\,\|\,p(z_t\mid h_t)\big)}_{\text{make the prior predict what the posterior sees}}\Big]$$
(with "free bits": KL clamped at ≥ 1 nat). **Dreamer V1–V3** then train an actor-critic purely on imagined latent rollouts; DreamerV3 solves Atari, DMC and Minecraft diamonds with one hyperparameter set.

**Notebook 136:** a scaled-down RSSM learns 64×64 pendulum video (15k frames, loss 3755 → 6 in 94 s). After 5 warm-up frames the prior alone imagines the future: sharp at first, then it drifts. The prediction error doubles at **step 12**: that is this model's **honest planning horizon**.

## 136–137 · How far ahead should you plan?
Notebook 137 trains an MLP dynamics model on pendulum transitions and plans with MPPI at horizons 2 to 60:
- open-loop prediction error **compounds** with horizon (the S01 lesson again, now inside the model);
- achieved return rises with horizon and then **falls**: past a point the planner optimizes against its own hallucinations.
**Measured (notebook 137):** one-step model error is tiny (MSE 7·10⁻⁶), yet

| Planning horizon | 2 | 5 | 10 | 20 | **40** | 60 |
|---|---|---|---|---|---|---|
| Return | −1651 | −1684 | −1120 | −237 | **−218** | −338 |

Too short and the planner can't see the swing-up pay off; too long and it exploits compounding model errors. A near-perfect one-step model still has a best horizon.

Practical answers: plan short and replan often (MPC); bootstrap beyond the horizon with a learned value (TD-MPC, Dreamer); use model ensembles to detect where the model is unsure.

## 138 · Why the stochastic state matters
A pole with a ball on top, released almost upright, falls left or right depending on a nudge far below the sensor's resolution (angle measured to 0.05 rad, the nudge is ~0.003 rad), so every release *looks* identical. In reality 49% fall right and 51% left, ending at x ≈ ±0.5 m. A **deterministic** model trained with MSE predicts the average future: final x = **−0.03**, a pole that never falls, which never happens. A **stochastic latent** (here a CVAE over the future) samples left *or* right futures (samples end at +0.54, −0.50, +0.52, −0.46, …). This is the world-model version of the multimodality problem from S08, and the reason RSSM keeps $z_t$ random.

## 139 · Video world models
- **Genie / Genie 2–3 (DeepMind):** action-controllable interactive worlds learned from video, with latent actions inferred without labels.
- **V-JEPA / V-JEPA 2 (Meta):** predict *representations*, not pixels, from masked video. **V-JEPA 2-AC** adds action conditioning trained on under 62 hours of unlabeled robot video and plans zero-shot pick-and-place by matching goal-image embeddings with MPC.
- **Cosmos (NVIDIA):** world foundation models for physical AI, used to generate synthetic training video and evaluate policies.
- **World-action models (DreamZero, Cosmos-Policy-style):** one network jointly predicts future video *and* actions, using the video-prediction objective as a physics prior for the policy.

## 140 · World models vs VLAs: the architecture debate
| | VLA (model-free imitation) | World model + planning |
|---|---|---|
| Learns | $o, \ell \mapsto a$ directly | how the world responds to actions |
| Data | action-labeled demos | any video (actions optional), plus some interaction |
| Inference | one forward pass (fast) | search over futures (slow, but adapts to new goals) |
| Failure | out-of-distribution states | compounding model error, hallucinated plans |
| Generalization | across instructions seen in data | to new goals and objectives without retraining |

The field is converging: VLAs add predictive auxiliary objectives (future-frame or latent prediction), and world models add policy heads. The book's position: they are complementary layers of the same system.

## Key takeaways
- With an accurate model, sampling-based MPC (MPPI) solves control with zero training.
- Learned models drift; plan within the honest horizon and replan.
- Stochastic latents represent uncertain futures that deterministic models average away.

## References
- Williams et al. 2017 (MPPI); Hafner et al. 2019–2023 (PlaNet, Dreamer V1–V3); Hansen et al. 2022 (TD-MPC)
- Bruce et al. 2024 (Genie); Assran et al. 2025 (V-JEPA 2); NVIDIA 2025 (Cosmos)
- Kandapagari, *Action Models for Robot Learning*, Ch. 9, Exercises 9.x.1–9.x.4
