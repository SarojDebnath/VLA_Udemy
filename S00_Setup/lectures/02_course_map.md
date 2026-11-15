# 02 · Course Map: What a VLA Is and How We Will Build One

**Section:** S00 Welcome · **Type:** Theory · **Duration:** ~15 min
**Companion notebooks:** `00_env_check`, `01_openvla_libero_colab`, `03_menagerie_tour`

## Learning objectives
- Define a Vision-Language-Action (VLA) model with the three-slot anatomy: input, output, training signal.
- Name the four families of action models and place VLAs among them.
- Understand the course's "one world, many models" design and the TableTop thread.
- Know which compute tier each part of the course needs.

---

## 1. What is a VLA?
A **Vision-Language-Action model** is a policy

$$\pi_\theta(a_{t:t+H} \mid o_t, \ell)$$

that maps an **observation** $o_t$ (one or more camera images, often the robot's proprioceptive state too) and a **language instruction** $\ell$ ("put the red cube in the yellow tray") to **actions**. That is either a single action $a_t$ or a *chunk* of the next $H$ actions.

What makes it a VLA and not just "a policy" is where the weights come from. The vision and language parts start from models pretrained on internet-scale image-text data (CLIP/SigLIP, DINOv2, Llama, Gemma, PaliGemma). Only the step from *aligned perception* to *motor commands* is learned from robot data.

Our first demo (`01_openvla_libero_colab`) shows the whole thing working:
- Input: a 224×224 RGB image plus a sentence.
- Model: OpenVLA-7B, a Llama-2 backbone with a fused SigLIP + DINOv2 vision encoder.
- Output: 7 discrete tokens → 7 numbers $(\Delta x, \Delta y, \Delta z, \Delta\text{roll}, \Delta\text{pitch}, \Delta\text{yaw}, \text{grip})$.
- Loop: observe → predict → `env.step(action)` → repeat at ~5–15 Hz.

## 2. The three-slot anatomy
Any action model can be described by filling three slots:

| Slot | Question | Examples |
|---|---|---|
| **1. Input** | What does the model see? | image resolution, number of cameras, wrist cam, proprio, history, language |
| **2. Output** | What does it emit, at what rate? | symbolic plan, joint targets, end-effector deltas, discrete tokens, continuous chunks |
| **3. Training signal** | Where does learning come from? | derivation (physics), reward (RL), demonstrations (imitation), pretraining + fine-tune |

We will use this table for every model in the course: RT-1, RT-2, OpenVLA, Octo, π0, SmolVLA, and our own MiniVLA.

## 3. The four families of action models
1. **Classical / analytical.** The action comes from a model of the world: kinematics, dynamics, planners, PDDL. It is precise and needs no data, but it is brittle when the world model is wrong. *(Section S04.)*
2. **Reinforcement learning.** The action is learned from a reward by trial and error. It can be superhuman, but it needs a lot of samples and reward design is hard. *(S05.)*
3. **Imitation learning.** The action is learned from demonstrations. Simple and data-efficient, but it suffers from compounding error and multimodality. *(S06–S08.)*
4. **Foundation / VLA.** Imitation at scale on top of web-pretrained vision-language models, trained across many robots. It generalizes across objects, scenes and instructions. *(S09–S13.)*

Modern robot stacks combine all four. A VLA proposes actions. Classical IK and controllers execute them. A safety shield (S15) vetoes bad ones. RL post-training (S16) polishes them.

## 4. Why "action" is the hard part
Perception mistakes cost one wrong label. Action mistakes **compound**: each action changes the next observation. A policy that is 99% accurate per step can be far off after 100 steps, because it drifts into states it never saw during training. This one fact explains most of the course's design choices: DAgger, action chunking, diffusion heads, closed-loop evaluation and safety layers.

## 5. The course design: one world, many models
- **S03** builds **TableTop**: a Franka Panda on a table with colored cubes and trays, front/wrist/top cameras, randomized layouts and language tasks (`pick`, `place`, `stack`).
- **S04** writes a **scripted expert** that solves TableTop using privileged state. That expert generates every dataset in the course.
- The same data and the same evaluation harness then serve each model in turn:

$$\text{BC} \rightarrow \text{ACT} \rightarrow \text{Diffusion Policy} \rightarrow \text{Flow matching} \rightarrow \text{MiniVLA (tokens)} \rightarrow \text{MiniVLA (flow)} \rightarrow \text{SmolVLA / OpenVLA fine-tune}$$

Because everything is measured the same way, every claim in the theory lectures becomes a number you can reproduce.

## 6. The robot cast (MuJoCo Menagerie)
| Robot | Where |
|---|---|
| Franka Panda (main arm) | TableTop: S03–S13, capstones C1, C5, C6 |
| UR5e + Robotiq, SO-101 | cross-embodiment (S04 41, S13 132), sim-to-real starter (C3) |
| ALOHA (two ViperX arms) | bimanual handover with ACT (C2) |
| Skydio X2 quadrotor | PID drone racer (S04 42) |
| Stanford TidyBot (mobile base + Kinova arm) | MPC crowd crossing (S04 42b), mobile fetch VLA (C7) |
| Unitree Go1 / Go2 | RL walking in MJX (S05 53), classical trot (S04 42d) |
| Unitree G1 humanoid | LQR balance (S04 42c), RL balance (S05 53b), whole-body (S16 151), humanoid VLA with Dex3 hands (C4) |
| Shadow / LEAP hands | Menagerie tour (S00 03) |
| A self-driving car (built in code, in a procedural MuJoCo city) | mini driving VLA (C8); NVIDIA Alpamayo 2 Edge zero-shot and in shadow mode (S16 150) |

## 7. Compute tiers
- 🟢 laptop / CPU: math, MuJoCo, classical control, RL toys, BC, ACT/diffusion on small tasks.
- 🟡 Colab T4: OpenVLA 4-bit inference, SmolVLA, MiniVLA training, MJX locomotion.
- 🔴 A100: full OpenVLA LoRA, π0. Always optional, always with a 🟡 fallback.

## Key takeaways
- A VLA is $\pi(a \mid o, \ell)$ built on web-pretrained vision-language weights and fine-tuned on robot data.
- Describe every model with three slots: input, output, training signal.
- There are four families of action models, and real systems mix them.
- Compounding error is the central difficulty of acting.
- The course follows one world (TableTop) and one data engine (the scripted expert) through every model.

## References
- Kim et al. 2024, *OpenVLA* (arXiv:2406.09246)
- Brohan et al. 2022/2023, *RT-1* (arXiv:2212.06817), *RT-2* (arXiv:2307.15818)
- Black et al. 2024, *π0* (arXiv:2410.24164)
