# 116–123 · Datasets, Benchmarks, and Evaluation You Can Trust

**Section:** S12 Data, Benchmarks, Evaluation · **Type:** Theory + math + labs · **Duration:** ~70 min
**Notebooks:** `117_lerobot_dataset` (kernel: lerobot), `119_success_rate_statistics`, `120_comparing_policies`, `121_evaluation_harness`, `122_libero_seeds_openvla` (Colab), `123_robustness_suite` · S06 `56_lerobot_dataset_format`

---

## 116 · Anatomy of a robot dataset
- **Episode** = one attempt at one task; a list of **timesteps** at a fixed control rate.
- **Timestep** = observations (1–3 camera images, proprioception), the action taken, the language instruction, and flags (first/last/terminal), sometimes reward.
- **Metadata:** fps, robot type, action-space convention (end-effector delta vs joint, frame, gripper sign), normalization statistics.

**RLDS (TensorFlow Datasets):** the format of Open X-Embodiment, with sequences of steps in TFRecords. TF-first and heavy for PyTorch users.
**LeRobotDataset (Hugging Face):**
- tabular data (state, action, timestamps, indices) in **Parquet**,
- camera streams as **MP4 video** (consecutive frames are nearly identical; video codecs exploit that),
- metadata: `info.json` (schema, fps, paths), tasks, episodes, **stats** (first-class normalization statistics, so the "wrong `unnorm_key`" bug becomes loud).
- **v3.0 (LeRobot ≥ 0.4):** many episodes per Parquet/MP4 file, with episode metadata in Parquet. That scales to millions of episodes and streams from the Hub.
- **`delta_timestamps`:** ask for `action` at offsets [0, 0.1, …, 1.5] s and get a 16-step chunk plus an `action_is_pad` mask; ask for `observation.images.front` at [−0.1, 0] and get frame history. Chunking policies are configured through this one dictionary.

Notebook 117 exports all 200 TableTop episodes with `LeRobotDataset.create → add_frame → save_episode → finalize` and loads them back with chunk queries.

## 118 · Benchmarks
| Benchmark | Simulator | What it tests |
|---|---|---|
| **LIBERO** | robosuite/MuJoCo | 4 suites × 10 tasks: Spatial, Object, Goal, Long (lifelong learning) |
| **CALVIN** | PyBullet | long-horizon language chains (5 subtasks in a row) |
| **RoboCasa** | robosuite/MuJoCo | large-scale kitchen scenes, 100 tasks, procedural assets |
| **SimplerEnv** | SAPIEN | sim scenes *matched* to real Google-robot and WidowX/Bridge setups, so sim success correlates with real |
| **RoboArena** | real robots, crowdsourced | pairwise A/B comparisons of policies across many labs |

Caveats: benchmark saturation (LIBERO is near 95%+ for modern VLAs), sim-to-real correlation only where it has been validated, and sensitivity to evaluation details (image flip, prompt, normalization key).

## 119 · A success rate is a random variable
k successes in n episodes ~ Binomial(n, p). With n = 25 and true p = 0.75, five seeds of the *same* policy gave 68%, 72%, 80%, 88% and 92%.
**Wilson interval** (use it, not the textbook Wald interval):
$$\frac{\hat p + \frac{z^2}{2n} \pm z\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$
Wald $\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$ undercovers badly near 0 and 1; for 25/25 it gives the absurd zero-width [1, 1]. Width shrinks as $1/\sqrt n$: 70% measured with n = 25 is [48%, 83%]; with n = 100 it's [60%, 78%].
**Pooling** independent seeds of the same setup tightens the interval (5 × 25 → [72%, 86%]).

## 120 · Comparing two policies honestly
- A 12-point gap (84% vs 72%) is *not* significant at n = 25 or 50 and is borderline at n = 100 (Fisher exact p = 0.06).
- **Sample size** to detect a difference with α = 0.05 and 80% power: 70→80% needs ~294 episodes *per policy*; 85→90% needs ~686.
- **Pair** the comparison: evaluate both policies on the same initial conditions (same seeds). Use paired bootstrap or McNemar on the discordant pairs; pairing removes scene-difficulty noise.
- Report: n, seeds, the interval, the per-task breakdown, and the exact evaluation conditions.

## 121–123 · Building your own evaluation
`vlacourse.evaluate` + `vlacourse.eval_suite`:
- fixed seed lists → reproducible initial conditions,
- per-task success, episode lengths, optional videos,
- **perturbation suite** applied after reset with identical seeds: front-camera shift (3 and 6 cm), dim and bright light, table color, slippery cubes (μ = 0.5),
- paired lost/gained episode counts per perturbation.

Notebook 121 validates the harness with a controllable "policy quality" knob (expert + noise). Notebook 123 produces a **robustness matrix** for ACT, Diffusion Policy and flow matching (30 paired episodes per cell, "single" scene):

| Condition | ACT | Diffusion | Flow |
|---|---|---|---|
| nominal | 97% | 97% | 97% |
| camera +3 cm | 93% | 93% | 100% |
| camera +6 cm and lower | 90% | 90% | 100% |
| dim light (×0.5) | 80% | 93% | 83% |
| bright light (×1.6) | 77% | 73% | 87% |
| **grey table** | **57%** | **70%** | **73%** |
| slippery cubes (μ = 0.5) | 97% | 90% | 97% |

All three policies look identical under nominal conditions (97%) and come apart under perturbation. The worst condition is the same for all of them: a change of table colour, i.e. the *background* they were never asked to ignore. Small camera shifts and physics changes cost little. A single nominal success rate hides all of this, and that's the argument for reporting a matrix. S13 notebook 131 fixes the visual weakness with randomized training data. The video shows one policy on the same episode in three of these worlds. Notebook 122 repeats the seed study for OpenVLA on LIBERO (Colab).

**Real-robot evaluation (book §15.5):** fix the reset protocol, randomize the order of policies, blind the operator, log videos, report time-to-completion and interventions, and still use intervals.

## Key takeaways
- LeRobotDataset v3 = Parquet + MP4 + metadata with stats; `delta_timestamps` defines chunks and history.
- Use Wilson intervals; to halve the width, run 4× the episodes.
- Small gaps need hundreds of paired episodes; robustness suites reveal what nominal success hides.

## References
- Cadene et al. 2024 (LeRobot); Liu et al. 2023 (LIBERO); Li et al. 2024 (SimplerEnv); Nasiriany et al. 2024 (RoboCasa)
- Wilson 1927; Agresti & Coull 1998
- Kandapagari, *Action Models for Robot Learning*, Ch. 15, Exercises 15.x.1–15.x.5
