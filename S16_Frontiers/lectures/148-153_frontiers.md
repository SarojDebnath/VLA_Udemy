# 148–153 · Frontiers: RL Post-Training, Reasoning VLAs, Whole-Body Control, and What Comes Next

**Section:** S16 Frontiers · **Type:** Theory + labs · **Duration:** ~60 min
**Notebooks:** `149_rl_finetune_flow_policy`, `151_humanoid_whole_body`

---

## 148 · RL post-training of pretrained VLAs
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
- **Does it actually reason?** Stress tests (BeTTER) show reasoning traces that don't causally drive the actions; faithfulness critics check whether the stated plan matches the behavior. Treat text reasoning as a useful interface, not proof of understanding.

### 150 notebook · A reasoning VLA that drives: NVIDIA Alpamayo 2 Edge
`150_alpamayo_reasoning_driver.ipynb` · video `outputs/150_alpamayo_in_mujoco.mp4` (+ a 1.3× LinkedIn cut) · kernel **Python (alpamayo)**, Linux GPU ≥ 16 GB.

**The model (measured on load):** 2.65B parameters, bf16: a 2.45B vision-language backbone (Cosmos 3 Edge: SigLIP2 + Nemotron LM) and a **0.2B flow-matching action expert**.
- **In:** 6 cameras (cross-left/right 120°, front-wide 120°, rear-left/right 70°, front-tele 30°), 22 frames over the last 2 s, 20 past poses, a navigation instruction ("Turn right in 30m").
- **Out:** per candidate, a chain-of-causation sentence (or `<NOMINAL>` when nothing needs explaining: *adaptive reasoning*) and a 6.4 s trajectory (64 poses, 10 Hz) from acceleration and curvature, as in our C8 model.
- **NVIDIA's numbers:** 15.15 km per at-fault incident (AlpaSim closed loop), minADE₆ 0.52 m at 5 s (real data). Weights under OpenMDW-1.1, code Apache-2.0.

**Zero-shot in our MuJoCo city:** we mount Alpamayo's camera rig on the C8 ego car (pinhole cameras), keep a 2 s buffer, and build exactly the input dict NVIDIA's real-data loader builds. Nothing retrained.

| Situation | What it said (candidate 1) |
|---|---|
| green light, "Turn left in 20m" | "Proceed forward and turn left at the intersection to follow our route, as the traffic signal is green and our path is clear." (all 6 turn left) |
| light just turned yellow at 32 km/h | "Proceed straight through the intersection… as the traffic light has turned yellow and we cannot stop safely before the stop line." |
| jaywalker at a red light | "Continue decelerating to stop before the crosswalk for the red traffic light and the pedestrian crossing our path." |
| free question: "How should the ego vehicle drive?" | "Stop and wait behind the pedestrian in the crosswalk, as the traffic light ahead is red and the pedestrian is crossing our path." |

**Closed loop (it drives): 0/5.** 2 off-road, 3 collisions. The top-down plot shows why: the car tracks Alpamayo's trajectory faithfully, but its **left turns swing 4–6 m wide**, sized for real streets seen through real (fisheye) lenses, not our 10 m road and pinhole renders. It reads the scene; its metric geometry doesn't transfer. Sim-to-real in reverse.

**Shadow mode (the expert drives, Alpamayo plans alongside): 509 plans scored against what the car really did over 3 s.**

| | minADE₆ @ 3 s | candidate-1 ADE | stop/go agreement |
|---|---|---|---|
| all scenarios | **0.35 m** | 0.60 m | **89%** (93% while moving, **81% while stopped**) |

Good open-loop numbers, failed closed loop: the same lesson as C8. And a 19% "go" while the expert waits is exactly what causes crashes when it drives.

**Faithfulness check (lecture 150):** of 554 plans, 520 had reasoning text and 434 mentioned a light. When it names a colour, it matches our approach's light (within 45 m): "green" **63%** of 209, "red" **76%** of 249, "yellow" **32%** of 40. Part of the gap is timing ("has turned green" just before it does) and other signals in view, but the takeaway stands: the text is a useful interface, not ground truth.

**Latency:** median **1.48 s** per call (6 candidates with reasoning, 10 flow steps), p90 2.03 s, RTX PRO 6000, PyTorch SDPA without CUDA graphs. Real cars replan at ~10 Hz: hence NVIDIA's accelerated path and DRIVE Thor deployment.

**Real NVIDIA clips** (part 2 of the notebook) need the gated `nvidia/PhysicalAI-Autonomous-Vehicles` dataset: accept its terms on Hugging Face and `hf auth login`; otherwise the notebook skips that part.

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
1. Pick a problem that matches your resources: sim-only → evaluation, tokenizers, RL fine-tuning; one robot → data and fine-tuning recipes; a fleet → scaling.
2. Read papers with the three-slot table (lecture 07) and look for the evaluation details (seeds, n, CIs).
3. Reproduce before you extend: our TableTop + `vlacourse` harness is a template.
4. Share data in LeRobot format, report Wilson intervals, release evaluation code.

**Reading roadmap:** OpenVLA → Octo → Diffusion Policy → ACT → π0 → FAST → π0.5 → SmolVLA → GR00T N1 → Gemini Robotics → RL post-training (VLA-RL, RIPT, π*0.6) → world models (Dreamer V3, V-JEPA 2) → safety (Code-as-Monitor, backdoor papers).

## Key takeaways
- RL post-training improves VLAs beyond their demos; regularize toward the pretrained policy.
- Reasoning traces help and are not guaranteed to be faithful.
- Humanoids need fast whole-body controllers under the VLA.

## References
- Zhang et al. 2025 (ReinFlow); Ren et al. 2024 (DPPO); Zawalski et al. 2024 (ECoT); Physical Intelligence 2025 (π*0.6)
