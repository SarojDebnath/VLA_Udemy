# 06 · A Short History of Action Models: From STRIPS to π0

**Section:** S01 · **Type:** Theory · **Duration:** ~20 min

## Learning objectives
- Place the major action-model ideas on a timeline and explain what each one unlocked.
- Recognize the recurring pattern: a new representation plus more data leads to a jump in generality.

---

## Era 1: Symbolic planning (1971–1990s)
- **STRIPS (Fikes & Nilsson, 1971)**: the world as logical facts; actions with preconditions, add-lists and delete-lists. Shakey the robot.
- **PDDL (1998)**: a standard language for planning problems. It's still used for task planning today (S04).
- *Unlocked:* long-horizon reasoning. *Missing:* perception and continuous motion.

## Era 2: Geometric and dynamic control (1980s–2000s)
- Inverse kinematics, operational-space control (Khatib 1987), computed-torque control.
- Sampling-based motion planning: PRM (1996), RRT (LaValle 1998).
- *Unlocked:* precise, safe motion. *Missing:* the ability to handle unmodeled objects.

## Era 3: First learned policies (1988–2010s)
- **ALVINN (Pomerleau, 1988)**: a neural network steering a truck from 30×32 camera pixels. Behavior cloning, and the first time compounding error was observed.
- **DAgger (Ross et al., 2011)**: the theoretical fix for covariate shift.
- Learning from demonstration / kinesthetic teaching, DMPs.

## Era 4: Deep RL (2013–2019)
- **DQN (Mnih et al., 2015)**: human-level Atari from pixels.
- TRPO / PPO, DDPG / TD3 / SAC: continuous control in MuJoCo becomes the benchmark.
- **Domain randomization (Tobin 2017; OpenAI Dactyl 2018)**: sim-to-real for dexterous hands.
- *Unlocked:* skills beyond human demonstration. *Missing:* sample efficiency and generality.

## Era 5: Transformers and large-scale imitation (2020–2023)
- **CLIP (2021)**: aligned vision-language embeddings from 400M image-text pairs.
- **Decision Transformer (2021)**: control as sequence modeling.
- **BC-Z (2021), RT-1 (2022)**: language-conditioned imitation on 130k real demos, discrete action tokens.
- **ACT (2023), Diffusion Policy (2023)**: action chunks and generative action heads; robust real-world bimanual and contact-rich skills.

## Era 6: Vision-Language-Action foundation models (2023–present)
- **RT-2 (2023)**: a VLM fine-tuned to output actions as text tokens; web knowledge transfers to robots.
- **Open X-Embodiment (2023)**: 1M+ episodes, 22 embodiments, 21 institutions.
- **Octo (2024)**: an open generalist policy with a diffusion head.
- **OpenVLA (2024)**: open 7B VLA on Llama-2 + SigLIP/DINOv2.
- **π0 (2024)**: PaliGemma + a flow-matching action expert, 50 Hz action chunks. Then π0-FAST, π0.5 (open-world), π0.6+ (RL from experience).
- **Dual systems: Helix, GR00T N1 (2025)**: a slow VLM "System 2" plus a fast "System 1" controller for humanoids.
- **SmolVLA (2025)**: a small, community-trained VLA on LeRobot data.
- Frontier: world-action models, reasoning VLAs, RL post-training of VLAs.

## The recurring pattern
| Transition | New representation | New data source |
|---|---|---|
| Symbolic → geometric | continuous configuration space | robot models |
| Geometric → learned | neural networks | demonstrations / reward |
| Learned → foundation | tokens + transformers | web text/images + pooled robot data |

Each era didn't replace the previous one; it was layered on top. A 2026 humanoid still runs IK and PD loops under its VLA.

## Key takeaways
- Every era added a capability (reasoning → precision → learning → generality).
- VLAs = imitation learning + web-pretrained VLMs + cross-embodiment data.
- Classical components remain the bottom layer of modern stacks.

## References
- Fikes & Nilsson 1971; LaValle 1998; Pomerleau 1988; Mnih et al. 2015
- Brohan et al. 2022 (RT-1), 2023 (RT-2); Kim et al. 2024 (OpenVLA); Black et al. 2024 (π0)
- Kandapagari, *Action Models for Robot Learning*, §1.3
