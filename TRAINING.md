# Training the course models yourself (or downloading them)

Every notebook that uses a trained model follows the same rule: **if the weights are in its `checkpoints/` folder it
loads them; if not, it trains them and saves them there.** So you have two options for each section:

| | Download the pretrained weights | Train them yourself |
|---|---|---|
| Command | `python tools/weights.py get` (all, ~750 MB) or `python tools/weights.py get S08 S17` | just run the notebook |
| Time | a few minutes | see the table below |
| Good for | recording, following along on a laptop, comparing your results with ours | learning: you see the training curves, can change things, and reproduce the numbers |

To retrain something you downloaded, delete its file in `SXX_*/checkpoints/` and re-run the notebook.

## Datasets
Demonstrations are never downloaded: the scripted experts record them the first time a notebook needs them, using all
CPU cores (`vlacourse.scene_dataset`, `tower_dataset`, `collect_g1_parallel`, ...), and save them in `data/`. A
500-episode TableTop dataset takes about 1–2 minutes on a 16-core machine with GPU rendering, longer on a laptop.

## What trains, and how long it took us
Times are the full notebook run (recording data + training + evaluation + videos) on one NVIDIA RTX PRO 6000. They
are the only times we measured. A Colab T4 or a gaming GPU is several times slower; a 4 GB laptop GPU much slower,
so for the heavy rows use the downloads or Colab.

| Notebook | Trains | Our time | Where to run it |
|---|---|---|---|
| S05 48 / 50 | PPO, SAC from scratch (small control tasks) | 2–3 min | 🟢 laptop |
| S05 53 | Go1 walking, PPO in MJX (60M steps) | ~25 min on a T4 | 🟡 Colab |
| S05 53b | G1 humanoid walking and balance, PPO in MJX (200M steps) | 20 min | 🟡 Colab A100/L4 or a Linux GPU |
| S06 57 / 60 | behaviour cloning from state, DAgger | 2–5 min | 🟢 laptop |
| S06 58 / 62 | pixel behaviour cloning, 8-variant ablation | 16 / 33 min | 🟡 GPU |
| S07 66 / 67 / 69 | Decision Transformers | 5–17 min | 🟢/🟡 |
| S08 73 / 74 / 78 / 83 / 84 | chunked regression, ACT, Diffusion Policy, flow matching (five heads) | 19–31 min each | 🟡 GPU |
| S09 90 / 91 | language-conditioned policies (concat vs FiLM) | 36 / 10 min | 🟡 GPU |
| S10 103 / 104 | MiniVLA (SmolVLM-256M + LoRA) | 24 / 8 min | 🟡 GPU (runs on 4 GB with gradient checkpointing, slowly) |
| S11 108 / 114 / 115 | MiniVLA with a flow expert; async inference; dual system | 12 / 17 / 6 min | 🟡 GPU |
| S12 123 | robustness suite (evaluation of S08 policies) | 9 min | 🟡 GPU |
| S13 129 / 130 / 131 / 132 | LoRA ablations, over-specialization, visual DR, new embodiment | 35 / 8 / 27 / 5 min | 🟡 GPU |
| S13 127 / 128 | SmolVLA fine-tune, OpenVLA-7B QLoRA | not timed here (our estimate: 30–45 min on an A100) | 🟡 Colab (A100/L4) |
| S15 144 / 146 | runtime monitor, adversarial patch | 3 / 9 min | 🟡 GPU |
| S17 C1 / C2 | language sorter (uses S09), ALOHA ACT | 4 / 4 min (+25 min for C2's ACT) | 🟡 GPU |
| S17 C4 | G1 humanoid VLA (ACT and FiLM, 1,000 demos) | 38 min | 🟡 GPU |
| S17 C5 / C6 / C7 | tower builder, data engine, mobile fetch | 5 / 13 / 3 min | 🟡 GPU |

Everything not listed (S01–S04, S14, most of S09, S12, S15, S16) trains nothing large and runs in minutes on a laptop.

## Reproducibility
Seeds are fixed, but GPU training is not bit-for-bit deterministic. Expect your success rates to land within the
confidence intervals reported in the lectures, not on the exact same number. That is itself a lesson of S12.
