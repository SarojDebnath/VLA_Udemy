# 148–153 · Frontiers: RL Post-Training, Reasoning VLAs, Whole-Body Control, and What Comes Next

**Section:** S16 Frontiers · **Type:** Theory + labs · **Duration:** ~60 min
**Notebooks:** `149_rl_finetune_flow_policy`, `151_humanoid_whole_body`

---

## 148 · RL post-training of pretrained VLAs (book §16.8, pending in the book)
Imitation caps a policy at the quality and coverage of its demonstrations. RL on the policy's own rollouts fixes its *own* mistakes (the DAgger idea, with a reward instead of an expert).
- **Discrete-token VLAs** (OpenVLA): PPO/GRPO directly on action-token log-probabilities (VLA-RL, RIPT-VLA, RLinf-VLA). Sparse success rewards, often in simulation, with large gains on LIBERO-style suites.
- **Flow/diffusion VLAs** (π0): exact log-likelihoods are intractable. Options:
  - **noise-injected sampling** (ReinFlow): make the sampling chain stochastic, so each step is Gaussian and the log-probability is a sum of Gaussian terms;
  - **DPPO:** treat the denoising chain as a multi-step MDP;
  - **π-RL / π*0.6 (RECAP):** advantage-conditioned training on real deployments, plus corrections.
- **Failure modes:** reward hacking (a success detector fooled), catastrophic forgetting of language-following and skills outside the RL task, and instability. Mitigations: KL or BC regularization toward the pretrained policy, mixed replay of demonstrations, conservative learning rates.

## 149 · ReinFlow-lite (notebook)
A flow policy behavior-cloned from only 30 demonstrations of the go-around-the-obstacle task, then fine-tuned with:
- stochastic execution: chunk = Euler-sampled mean + σ·ε (σ = 0.08), so $\log\pi$ is Gaussian and differentiable through the flow chain,
- REINFORCE with a moving-average baseline on the binary success reward,
- a **BC anchor** (the flow-matching loss on the demos), which keeps the policy close to the data.
Read the curve as: does success climb above the BC starting point, and does it stay stable?

**Measured:** BC from 30 demos gives **70%**. After 40 RL iterations, evaluated on 100 fresh episodes, success is **76%**. The per-iteration training curve is noisy (50–81%), peaking at 82% (deterministic eval) around iteration 10 before settling.
Two honest lessons: RL post-training *does* lift a small-data BC policy, but with a binary reward and small batches the gain is modest and the curve is noisy. In real systems the BC anchor and many more rollouts (or simulation at MJX scale) are what make it reliable. Checkpoint selection on a held-out eval matters, because the best iterate isn't the last.

## 150 · Reasoning + action
- **Embodied chain-of-thought (ECoT):** the VLA writes plan, subtask, object boxes and gripper position in text before its action tokens. More interpretable, and better on novel scenes, at the cost of latency.
- **Latent reasoning / CoT-dropout:** train with reasoning, run without it (or with short latent tokens).
- **Gemini Robotics-ER / "Embodied Thinking":** a reasoning VLM produces pointing, trajectories and success checks for a downstream controller.
- **Does it actually reason?** Stress tests (BeTTER) show reasoning traces that don't causally drive the actions; faithfulness critics check whether the stated plan matches the behavior (book §18.7). Treat text reasoning as a useful interface, not proof of understanding.

## 151 · Cross-embodiment and whole-body control (notebook)
The Unitree G1: 29 actuated joints plus a 6-DoF floating base. Notebook findings:
- the model has 36 position coordinates (7 for the floating base), 35 velocities and 29 actuators;
- without control it collapses; joint PD on the standing pose holds still;
- **push test** (0.2 s at the pelvis): 20 N, still standing (pelvis 0.79 m); **60 N, falls** (pelvis 0.06 m); 120 N, falls. Joint PD has no notion of balance;
- raising one arm moves the center of mass, and nothing re-balances; the PD posture only *survives* because the shift is small.
A VLA emitting arm targets at a few Hz cannot keep a humanoid upright. That's why humanoid stacks add a learned whole-body **System 0** controller at high rate (Helix-02), trained with massively parallel RL (S05 notebooks 52–53) and accepting targets from the VLA. Gemini Robotics 2 and GR00T N1.x push toward unified loco-manipulation.

## 152 · More frontiers
- **Discrete-diffusion VLAs:** decode action tokens in parallel by iterative unmasking (adaptive order, re-masking) instead of left to right; this combines token heads with parallel decoding.
- **In-context robot learning:** adapt from a few demonstrations given in the prompt, without gradient steps.
- **Video-pretrained action models:** V-JEPA 2-AC plans from under 62 h of robot video; crowdsourced egocentric human video as scalable "demonstrations".
- **Tactile sensing:** vision can't see contact forces; tactile world models and policies for insertion and in-hand manipulation.

## 153 · How to keep up and contribute
1. Pick a problem that matches your resources (book Exercise 18.x.1): sim-only → evaluation, tokenizers, RL fine-tuning; one robot → data and fine-tuning recipes; a fleet → scaling.
2. Read papers with the three-slot table (lecture 07) and look for the evaluation details (seeds, n, CIs).
3. Reproduce before you extend: our TableTop + `vlacourse` harness is a template.
4. Share data in LeRobot format, report Wilson intervals, release evaluation code.

**Reading roadmap:** OpenVLA → Octo → Diffusion Policy → ACT → π0 → FAST → π0.5 → SmolVLA → GR00T N1 → Gemini Robotics → RL post-training (VLA-RL, RIPT, π*0.6) → world models (Dreamer V3, V-JEPA 2) → safety (Code-as-Monitor, backdoor papers).

## Key takeaways
- RL post-training improves VLAs beyond their demos; regularize toward the pretrained policy.
- Reasoning traces help and are not guaranteed to be faithful.
- Humanoids need fast whole-body controllers under the VLA.

## References
- Kandapagari, *Action Models for Robot Learning*, Ch. 18, §16.7–16.8
- Zhang et al. 2025 (ReinFlow); Ren et al. 2024 (DPPO); Zawalski et al. 2024 (ECoT); Physical Intelligence 2025 (π*0.6)
