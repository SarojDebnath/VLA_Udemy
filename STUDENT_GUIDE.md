# Student guide: set up, then work through the course

This guide takes you from an empty computer to the last capstone. Follow it in order the first time; later, use the
section table in Part 3 as your map.

---

## Part 1 · One-time setup (about 30 minutes)

### What you need
| | Minimum | Comfortable |
|---|---|---|
| Computer | Windows 10/11, Linux or macOS, 16 GB RAM | 32 GB RAM, 8+ CPU cores |
| GPU | none (S00–S07 run on CPU) | NVIDIA with 8 GB+ for S06–S17 |
| Disk | 20 GB free | 40 GB (datasets, pretrained models, videos) |
| Accounts | a free GitHub account is enough | a free Google account for the Colab notebooks |

No NVIDIA GPU? Do S00–S07 locally, download our trained weights for the later sections (step 4), and use Colab for
the notebooks that train big models.

### Step 1 · Install the tools
- **Git:** https://git-scm.com/downloads
- **Miniconda** (Python environment manager): https://docs.anaconda.com/miniconda/
- **An editor for notebooks:** VS Code with the *Python* and *Jupyter* extensions (recommended), or JupyterLab, which
  step 2 installs.
- **NVIDIA GPU only:** a recent driver (`nvidia-smi` in a terminal should print your GPU).

### Step 2 · Get the code and create the environment
Open a terminal (on Windows: *Anaconda Prompt*) and run:
```bash
git clone https://github.com/SarojDebnath/VLA_Udemy.git
cd VLA_Udemy
conda create -n vla python=3.11 -y
conda activate vla
```
Install PyTorch for your hardware (pick **one** line):
```bash
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124   # NVIDIA GPU
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu     # no GPU (Windows/Linux)
pip install torch==2.6.0 torchvision==0.21.0                                                     # macOS
```
Then the rest of the course:
```bash
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name vla --display-name "Python (vla)"
```
`pip install -e .` installs the course library `vlacourse` in editable mode: notebooks import it, and if you change
it, notebooks see the change after a kernel restart.

### Step 3 · Check that everything works
```bash
jupyter lab            # or open the VLA_Udemy folder in VS Code
```
Open `S00_Setup/00_env_check.ipynb`, choose the kernel **Python (vla)** (VS Code: *Select Kernel* at the top right),
and run all cells. It prints your MuJoCo, PyTorch and GPU versions, renders a test scene and the TableTop world, and
downloads every robot model used in the course. Each robot should print `ok`; if anything fails, see Part 5.

### Step 4 · Decide: train yourself, or download our trained models
Every notebook that needs a trained model first looks in its section's `checkpoints/` folder. If the model is there,
the notebook loads it; if not, it trains it and saves it there. So both options work, and you can mix them per
section.
```bash
python tools/weights.py list           # what exists, and what you already have
python tools/weights.py get            # everything (~750 MB)
python tools/weights.py get S08 S10    # only some sections
```
**Recommendation:** train yourself on 🟢 notebooks (that's where the learning is), and download for the 🟡 ones your
hardware can't handle. [TRAINING.md](TRAINING.md) lists every notebook that trains something and how long it took us.
To retrain a model you downloaded, delete its file in `checkpoints/` and re-run the notebook.

You never download datasets: the course's scripted experts record their demonstrations the first time a notebook
needs them and save them in `data/`. That takes a few minutes once.

---

## Part 2 · How to study one lecture

Each number in the course (for example **74 · ACT**) is either a notebook, a lecture note, or both:
- `S08_Generative_Action_Heads/74_act.ipynb`: the hands-on part.
- `S08_Generative_Action_Heads/lectures/71-74_multimodality_cvae_chunking_act.md`: the theory and math, covering 71 to 74.

For every lecture:
1. **Watch the video**, with the lecture note open. The notes have the equations, tables and references.
2. **Run the notebook top to bottom** (*Run All*), then go back and read each cell's output. Every claim in the video is
   a number or a plot in the notebook; check them against yours.
3. **Change one thing.** A seed, a learning rate, a number of demonstrations. Predict the result first, then run it.
4. **Do the exercises** at the end of the notebook. Projects (titled *Project N*) end with a deliverable: do it
   before moving on.

Tips:
- Run cells in order. If you jump around and something breaks, use *Restart Kernel and Run All*.
- Your success rates won't match ours exactly (GPU training isn't bit-for-bit reproducible). They should fall within
  the confidence intervals the notebooks report. Section S12 explains why.
- Videos play inline in the notebook and are saved to `outputs/` as MP4.
- If a notebook says *"This notebook uses '…', trained in …"*, it needs a model from an earlier notebook: run that
  one first, or download that section's weights with the command it prints.

---

## Part 3 · The course map

Tier: 🟢 laptop/CPU · 🟡 NVIDIA GPU or Colab · ☁️ Colab only. Notes are in each section's `lectures/` folder.

| Section | What you build | Notebooks | Tier |
|---|---|---|---|
| **S00 Setup** | environment check, a pretrained VLA driving a robot, tour of the robots | 00, 01 ☁️, 03 | 🟢 |
| **S01 The robot learning problem** | open vs closed loop, compounding error | 04, 05 | 🟢 |
| **S02 Math foundations** | linear algebra, rotations, SE(3) and cameras, autodiff, probability, KL, optimization, a training loop, fault drills, loss families | 08–17 | 🟢 |
| **S03 MuJoCo from zero** | MJCF, equations of motion, contacts, integrators, actuators, sensors, cameras, MjSpec, viewer, speed. **Project 1: TableTop**, the world used for the rest of the course | 19–29 | 🟢 |
| **S04 Classical control** | FK, Jacobian, IK, PD and gravity compensation, computed torque, impedance, trajectories, RRT, PDDL. **Projects 2–3:** PDDL to torque, the data engine. Cross-embodiment. **Showcase projects:** PID drone racer, MPC crowd navigation, LQR humanoid balance, Go2 trot | 30–41, 42, 42b, 42c, 42d | 🟢 |
| **S05 Reinforcement learning** | value/policy iteration, Q-learning, reward design, REINFORCE, PPO and SAC from scratch, domain randomization, MJX. **Project 4:** Go1 walks. **G1 humanoid balance** | 43–51 🟢, 52–53 ☁️, 53b 🟡 (own env) | 🟢/☁️ |
| **S06 Imitation learning** | teleoperation, LeRobot dataset format, BC from state and pixels, DAgger. **Project 5:** pixel BC | 55–62 | 🟢/🟡 |
| **S07 Sequence models** | attention, a transformer from scratch, Decision Transformer, return conditioning, stitching, action tokenization | 63–69 | 🟢/🟡 |
| **S08 Generative action heads** | multimodality, CVAE, action chunking, ACT, DDPM, Diffusion Policy, flow matching, receding horizon. **Project 6:** five heads compared | 71–84 | 🟡 |
| **S09 Vision-language foundations** | ViT, CLIP/SigLIP from scratch, zero-shot CLIP, DINOv2 vs SigLIP, VLM anatomy, language conditioning, grounding test | 85–91 | 🟡 |
| **S10 Building a VLA** | action tokenizers, FAST, OpenVLA inside out. **Project 7: MiniVLA** (SmolVLM + LoRA), data scaling | 93–95 🟢, 99–100 ☁️, 103–104 🟡 | 🟡 |
| **S11 Continuous VLAs, dual systems** | flow-matching action expert (π0-style), latency, async inference. **Project 8:** dual-system agent | 108–115 | 🟡 |
| **S12 Data, benchmarks, evaluation** | LeRobotDataset v3, success-rate statistics, comparing policies, evaluation harness, OpenVLA on LIBERO, robustness suite | 117 (own env), 119–121 🟢, 122 ☁️, 123 🟡 | 🟢/🟡 |
| **S13 Fine-tuning for your robot** | LoRA from scratch, demo coverage, SmolVLA and OpenVLA-7B fine-tuning, ablations, over-specialization, visual DR. **Project 9:** new embodiment | 124–126 🟢, 127–128 ☁️, 129–132 🟡 | 🟡 |
| **S14 World models** | MPPI with MuJoCo, mini-Dreamer, planning horizon vs model error, stochastic latents | 134–138 | 🟢/🟡 |
| **S15 Safety and deployment** | safety shield, runtime monitor, canary rollout, adversarial patch | 142–146 | 🟢/🟡 |
| **S16 Frontiers** | RL fine-tuning of a flow policy, whole-body humanoid control | 149, 151 | 🟡 |
| **S17 Capstones** | choose at least one: C1 language sorter, C2 ALOHA ACT, C3 SO-101 starter, C4 G1 humanoid VLA, C5 tower builder, C6 data engine, C7 mobile fetch. Briefs in `lectures/capstone_briefs.md` | C1–C7 | 🟡 |

**Dependencies between sections.** These notebooks load models trained earlier (or downloaded):
| Notebook | Needs |
|---|---|
| S11 114, S13 132, S15 144 | the flow policy from S08 81 |
| S11 115, S17 C1 | the language policy from S09 90 |
| S12 123 | ACT (S08 74), Diffusion Policy (S08 78), flow (S08 81) |
| S15 146 | the chunked policy from S08 84 |

**Short on time?** A solid path: S00 → S02 (skim what you know) → S03 29 → S04 30–40 → S06 57–62 → S08 → S09 90 →
S10 103 → S11 108 → S12 119–121 → one capstone.

---

## Part 4 · Notebooks that need a different environment

### ☁️ Colab notebooks (S00 01, S05 52–53, S10 99–100, S12 122, S13 127–128)
They need a large GPU or packages that conflict with the main environment.
1. Go to https://colab.research.google.com → *File → Open notebook → GitHub*, paste
   `https://github.com/SarojDebnath/VLA_Udemy`, and pick the notebook.
   (VS Code alternative: open the notebook locally, *Select Kernel → Colab*.)
2. *Runtime → Change runtime type → T4 GPU* (or L4/A100 if you have Colab Pro).
3. Run the first cell: it installs the packages and clones the course. Then *Run All*.

Specifics:
- **S05 53** saves the trained policy to your Google Drive, so a disconnect doesn't lose 25 minutes of training.
- **S13 127** needs the TableTop dataset in LeRobot format. Create it locally with S12 117, zip
  `data/lerobot_v3/tabletop_panda_v1`, and upload the zip when the notebook asks.
- **S12 122** takes about 1.5 hours on a T4; keep the tab open.

### S05 53b · G1 humanoid RL (MuJoCo Playground, JAX)
Needs JAX with CUDA: Linux with an NVIDIA GPU, or Colab. **On Colab:** open it as above and run all cells; the first
cell installs everything and downloads our trained policy. **On a Linux GPU machine:**
```bash
python -m venv ~/.pg
~/.pg/bin/pip install "playground==0.2.0" "jax[cuda12]==0.7.2" "flax==0.11.2" "orbax-checkpoint==0.11.22" mediapy ipykernel
~/.pg/bin/pip install --no-deps -e .  &&  ~/.pg/bin/pip install mujoco-menagerie opencv-python-headless scipy imageio imageio-ffmpeg
~/.pg/bin/python -m ipykernel install --user --name pg --display-name "Python (playground)"
```
Open the notebook with the kernel **Python (playground)**.

### S12 117 · LeRobot
LeRobot pins its own PyTorch version, so it gets its own environment:
```bash
conda create -n lerobot python=3.11 -y
conda activate lerobot
pip install "lerobot[smolvla]>=0.4,<0.5" ipykernel matplotlib
pip install --no-deps -e .
python -m ipykernel install --user --name lerobot --display-name "Python (lerobot)"
conda activate vla
```
Open S12 117 with the kernel **Python (lerobot)**. It converts the TableTop dataset that S04 40 records into
`data/tabletop_panda_v1`, so run S04 40 (in the `vla` environment) first.

---

## Part 5 · When something goes wrong

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: vlacourse` | Wrong kernel: select **Python (vla)**. Or you skipped `pip install -e .` in the `vla` environment. |
| `torch.cuda.is_available()` is False with an NVIDIA GPU | You installed the CPU wheel: rerun the `cu124` line of step 2. Check `nvidia-smi` works. |
| `CUDA out of memory` | Close other notebooks (*Kernel → Shut Down*), restart the kernel. On small GPUs, download the weights for that section instead of training. |
| `FileNotFoundError: This notebook uses '…'` | Run the notebook it names first, or `python tools/weights.py get <section>`. |
| Rendering errors on Linux (`EGL`, `GLFW`) | Run headless: `export MUJOCO_GL=egl` (NVIDIA) or `MUJOCO_GL=osmesa` (CPU). See the README's server notes. |
| A window pops up and blocks | That's the OpenCV video player: press `q`. Set `VLA_HEADLESS=1` to turn windows off. |
| Robot model download fails | Check your internet connection and rerun `00_env_check`. Models are cached after the first download. |
| Hugging Face model download is slow | First use of CLIP, SigLIP, DINOv2 and SmolVLM downloads ~5 GB in total into `~/.cache/huggingface`. Only once. |
| Results differ a bit from the video | Expected. Look at the confidence intervals, not the exact number. |
| Something else | Copy the full error message (the last 20 lines) into the Q&A of the course, with the notebook name and your OS/GPU. |

To start a notebook clean: delete its section's `outputs/`, and the model in `checkpoints/` if you want to retrain.
`data/` holds the recorded datasets; deleting it makes the notebooks record them again.
