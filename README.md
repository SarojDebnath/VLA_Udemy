# VLA Mastery: Vision-Language-Action Models, from First Principles to Deployment

Course code for the Udemy course: 127 Jupyter notebooks plus Markdown lecture notes. You build the whole stack of
robot learning in MuJoCo, from rotations and PD control to behaviour cloning, ACT, Diffusion Policy, flow matching
and your own small Vision-Language-Action model, with Menagerie robots (Franka Panda, UR5e, ALOHA, Unitree Go2/G1,
Skydio X2, SO-101), and a self-driving car in a procedural MuJoCo city.

**Start here: [STUDENT_GUIDE.md](STUDENT_GUIDE.md).** It covers the setup, the order to work through the course, and what to do when
something breaks.

## Layout
```
SXX_<section>/
    NN_<topic>.ipynb        runnable notebook (lab or project)
    lectures/NN_<topic>.md  theory and math notes for the same numbers
    checkpoints/            trained models (created when you run or download them)
    outputs/                videos the notebooks render
vlacourse/                  shared library: TableTop world, Panda IK, scripted experts, policies, evaluation, video tools
tools/weights.py            download the pretrained course weights
_notebook_src/              plain-Python sources of the notebooks (# %% cells)
```

## Quick setup
```bash
conda create -n vla python=3.11 -y && conda activate vla
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124   # NVIDIA GPU; see the guide for CPU/Mac
pip install -r requirements.txt
pip install -e .
python -m ipykernel install --user --name vla --display-name "Python (vla)"
jupyter lab
```
Then open `S00_Setup/00_env_check.ipynb` with the **Python (vla)** kernel and run it.

Two notebooks need their own environments (LeRobot for S12/117, MuJoCo Playground for S05/53b), and a few run on
Colab. Details are in the guide.

## Pretrained weights (optional)
Every notebook trains its own models when its `checkpoints/` folder is empty, so you can do the whole course from
scratch. To skip the training time, download ours (~750 MB, from the Hugging Face Hub):
```bash
python tools/weights.py get            # everything
python tools/weights.py get S08 S17    # only some sections
```
What each notebook trains and how long it takes: [TRAINING.md](TRAINING.md).

## Compute tiers
- 🟢 CPU or a 4 GB laptop GPU: most of the course.
- 🟡 A bigger GPU or a free Colab T4. Colab notebooks start with a setup cell that clones this repo.
- 🔴 A100-class (optional). Every 🔴 notebook has a 🟡 fallback (smaller model, 4-bit, or fewer steps).

## Conventions
- Notebooks display videos inline; some also open an OpenCV window (press `q` to close). Set `VLA_HEADLESS=1` to
  disable windows (servers, Colab).
- Every robot comes from MuJoCo Menagerie through the `mujoco-menagerie` package; they download on first use.

## Headless Linux servers
If rendering is slow and the GPU sits at 0%, EGL fell back to Mesa's CPU renderer (common when your user isn't in the
`video`/`render` groups). `vlacourse` points `__EGL_VENDOR_LIBRARY_FILENAMES` at NVIDIA's EGL automatically; for your
own scripts, export it yourself. Parallel data recorders use `spawn` workers: forking a process that holds a GPU
context can crash the driver.
