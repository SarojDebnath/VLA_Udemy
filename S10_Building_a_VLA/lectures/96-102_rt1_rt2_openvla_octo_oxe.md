# 96–102 · The VLA Lineage: RT-1, RT-2, OpenVLA, Octo, and Open X-Embodiment

**Section:** S10 · **Type:** Theory · **Duration:** ~60 min
**Companion notebooks:** `99_openvla_inference_inside` (Colab), `100_openvla_failure_drills` (Colab), `S00/01_openvla_libero_colab`

---

## 96 · RT-1: Robotics Transformer (Brohan et al. 2022)
**Three slots:** six 300×300 history frames + an instruction (Universal Sentence Encoder) → 11 discrete action dims (7 arm + 3 base + mode), 256 bins each, at 3 Hz. Trained on 130k real demonstrations covering 700+ tasks (13 robots, 17 months).

Architecture:
1. **FiLM-EfficientNet-B3:** the language embedding modulates every MBConv block (early fusion; S09 notebook 90).
2. **TokenLearner:** compresses 81 visual tokens per frame to 8 learned tokens (48 total), which is key for real-time inference.
3. **Decoder-only transformer** (8 layers, 19M parameters) predicts all action tokens from the 48 image tokens.

What changed: one model, hundreds of tasks; generalization to new instructions combining known skills and objects. What didn't: no web knowledge, new objects remain hard, and a single embodiment.

## 97 · RT-2: Vision-Language-Action (Brohan et al. 2023)
Take a VLM (PaLI-X 55B or PaLM-E 12B) and **write actions as text**: "1 128 91 241 5 101 127", 8 integers, each a 256-bin token. **Co-fine-tune** on web VQA data and robot trajectories so the model keeps its web knowledge.

Emergent capabilities from web pretraining: "move the banana to the sum of two plus one", "pick up the extinct animal", "move the coke can to Taylor Swift". Semantic reasoning about *which* object transfers; physical skills don't (it only performs motions present in the robot data).

Cost: 55B parameters, 1–3 Hz, closed weights. RT-2-X (trained on Open X-Embodiment) improves cross-embodiment transfer.

## 98 · OpenVLA (Kim et al. 2024)
Open, 7B, reproducible.
- **Backbone:** Prismatic-7B = **DINOv2 + SigLIP** (≈600M, features concatenated) → 2-layer MLP projector → **Llama-2 7B**.
- **Input:** one 224×224 third-person image + the prompt "In: What action should the robot take to {instruction}?\nOut:". No wrist camera, no proprioception, no history.
- **Output:** 7 action tokens, 256 bins over the 1st–99th percentile, on the 256 least-used Llama tokens.
- **Data:** 970k episodes from Open X-Embodiment.
- **Findings:** fine-tuning the vision encoder is crucial; it beats RT-2-X (55B) by 16.5 points on 29 tasks with 7× fewer parameters. LoRA fine-tuning on ~50 demos reaches full-fine-tune quality (S13). 4-bit quantization has no measurable success drop and fits a 15 GB T4 (our S00 notebook).
- **Speed:** ~6 Hz on an RTX 4090 in bf16.

**The three silent failures (notebook 100, book §2.4):**
| Bug | Symptom | Detectable by |
|---|---|---|
| Image not flipped (LIBERO renders upside down) | healthy-looking actions, never succeeds | elimination: the world the model sees doesn't exist |
| Wrong prompt template | arm twitches near zero | mean ‖Δxyz‖ ≪ baseline |
| Wrong `unnorm_key` | translations 3–5× too large | ‖Δxyz‖ > 0.1 m on many steps |

These predicates become the kernel of the runtime monitor in S15.

## 101 · Octo (Octo Model Team 2024)
A generalist policy *designed* for fine-tuning.
- **Transformer over tokens:** language (T5), image patches (any number of cameras), proprio, and **readout tokens** that attend to everything (like CLS tokens per timestep).
- **Diffusion action head** on the readout embeddings → continuous action chunks.
- **Flexible observation/task specs:** a missing camera is just a missing token group (block-wise masking), and goal images or language both work as tasks.
- 27M (Small) and 93M (Base) parameters, trained on 800k OXE episodes; fine-tunes to new sensors and action spaces with ~100 demos.

## 102 · Open X-Embodiment (2023)
- 1M+ trajectories, **22 embodiments**, 21 institutions, 60 datasets, 527 skills, stored as RLDS.
- **Action-space normalization:** most datasets use end-effector deltas, mapped to a common 7-D convention; normalization statistics are kept per dataset (hence `unnorm_key`).
- **Mixture weights** are hand-tuned to balance huge datasets (e.g. Kuka grasping) against small, diverse ones.
- **Key result:** RT-1-X trained on the mixture beats the original per-lab models on their own robots by ~50% on average; positive transfer across embodiments.
- Successors: DROID (76k teleop episodes, 564 scenes), the AgiBot World and Unitree fleets, and LeRobot community datasets (S12).

## The lineage at a glance
| Model | Year | Params | Vision | Language | Action head | Data |
|---|---|---|---|---|---|---|
| RT-1 | 2022 | 35M | FiLM-EfficientNet | USE (frozen) | 256-bin tokens | 130k demos, 1 robot |
| RT-2 | 2023 | 12–55B | ViT (PaLI-X/PaLM-E) | LLM | text tokens | web + robot |
| Octo | 2024 | 27–93M | patch tokens | T5 | diffusion | 800k OXE |
| OpenVLA | 2024 | 7.5B | DINOv2+SigLIP | Llama-2 | 256-bin tokens | 970k OXE |
| π0 | 2024 | 3.3B | SigLIP (PaliGemma) | Gemma | flow-matching expert | 10k h, 7 robots |
| **MiniVLA (ours)** | — | 0.29B | SigLIP-style (SmolVLM) | SmolLM2 | 256-bin tokens | 180 TableTop demos |

## References
- Brohan et al. 2022 (arXiv:2212.06817); Brohan et al. 2023 (arXiv:2307.15818)
- Kim et al. 2024 (arXiv:2406.09246); Octo Model Team 2024 (arXiv:2405.12213)
- Open X-Embodiment Collaboration 2023 (arXiv:2310.08864); Khazatsky et al. 2024 (DROID)
- Kandapagari, *Action Models for Robot Learning*, Ch. 11–12
