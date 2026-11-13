# %% [markdown]
# # 128 · QLoRA fine-tuning of OpenVLA-7B on TableTop demonstrations (Colab)
# **Runtime:** Runtime → Change runtime type → **A100** or **L4** GPU (recommended). A T4 works in 4-bit but trains about 5× slower.
#
# **Data:** the TableTop demonstrations are not in the GitHub repo (they are generated). Upload `data/tabletop_panda_v1.zip`
# from the course folder to the top of your Google Drive (`MyDrive/tabletop_panda_v1.zip`). If it isn't there, the data
# cell records a fresh dataset in Colab with the course's scripted expert instead (slower, ~10 min).
#
# What happens below:
# 1. load OpenVLA-7B with its frozen weights in 4-bit NF4 and add LoRA adapters (QLoRA);
# 2. turn every 7-D TableTop action into 7 action tokens, exactly as OpenVLA does (256 bins, the last 256 vocabulary ids);
# 3. fine-tune on 180 demonstrations; check the predictions open-loop on 20 held-out demonstrations;
# 4. run the fine-tuned model in the TableTop simulator and watch it.

# %%
# Colab now runs Python 3.13. OpenVLA's original pins (transformers 4.40.1 / tokenizers 0.19.1) have no wheels for it, so we
# use the closest versions that install cleanly; OpenVLA prints a version warning, which is expected and harmless here.
!pip install -q "transformers==4.47.1" "timm==0.9.16" "peft==0.13.2" bitsandbytes accelerate sentencepiece
!pip install -q mujoco mujoco-menagerie imageio imageio-ffmpeg
# (pip may print huggingface-hub conflicts with diffusers/gradio: Colab packages we don't use, safe to ignore)
# A /content/VLA_Udemy left over from an earlier session may be incomplete: move it aside and clone fresh.
!test -f /content/VLA_Udemy/pyproject.toml || { test -d /content/VLA_Udemy && mv /content/VLA_Udemy /content/VLA_Udemy_old_$(date +%s); git clone -q --depth 1 https://github.com/SarojDebnath/VLA_Udemy.git /content/VLA_Udemy; }
!pip install -q --no-deps -e /content/VLA_Udemy

# %%
import os, sys, glob, time, zipfile, numpy as np, torch
os.environ["MUJOCO_GL"] = "egl"                     # headless rendering on Colab's GPU
sys.path.insert(0, "/content/VLA_Udemy")
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

GPU = torch.cuda.get_device_name(0); MEM = torch.cuda.get_device_properties(0).total_memory / 2**30
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16      # T4 has no bfloat16
BS = 8 if MEM > 30 else 4
STEPS = 1500 if MEM > 30 else 600
print(f"{GPU}, {MEM:.0f} GB, {DTYPE}, batch {BS}, {STEPS} steps")

# %% [markdown]
# ## Data

# %%
DATA = "/content/tabletop_panda_v1"
if not os.path.exists(f"{DATA}/meta.json"):
    if not os.path.exists("/content/drive"):
        from google.colab import drive; drive.mount("/content/drive")
    zp = "/content/drive/MyDrive/tabletop_panda_v1.zip"
    if os.path.exists(zp):
        zipfile.ZipFile(zp).extractall("/content")
    else:
        print("no zip on Drive: recording 200 demonstrations with the scripted expert ...")
        from vlacourse.data import collect_parallel
        collect_parallel(DATA, 200, n_cubes=3, n_trays=1, image_size=(128, 128), cameras=("front", "wrist"))
files = sorted(glob.glob(f"{DATA}/episode_*.npz"))
episodes = []
for f in files:
    z = np.load(f)                                  # read each array ONCE (indexing z["front"] re-decompresses it)
    episodes.append(dict(front=z["front"], action=z["action"], instruction=str(z["instruction"])))
train_eps, test_eps = episodes[:180], episodes[180:]
frames = [(e["front"][t], e["instruction"], e["action"][t]) for e in train_eps for t in range(len(e["action"]))]
print(len(files), "episodes,", len(frames), "training frames; e.g.", repr(train_eps[0]["instruction"]))

# %% [markdown]
# ## OpenVLA-7B in 4-bit + LoRA

# %%
MID = "openvla/openvla-7b"
proc = AutoProcessor.from_pretrained(MID, trust_remote_code=True)
tok = proc.tokenizer
model = AutoModelForVision2Seq.from_pretrained(MID, torch_dtype=DTYPE, trust_remote_code=True, device_map={"": 0},
    low_cpu_mem_usage=True,
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=DTYPE))
model = prepare_model_for_kbit_training(model)        # also turns on gradient checkpointing
model = get_peft_model(model, LoraConfig(r=32, lora_alpha=16, lora_dropout=0.0, target_modules="all-linear", init_lora_weights="gaussian"))
model.print_trainable_parameters()
# prepare_model_for_kbit_training upcasts the (unquantized) vision encoder to float32, so images must be float32 too
PIX_DTYPE = next(p for n, p in model.named_parameters() if "patch_embed" in n).dtype
print(f"GPU memory after loading: {torch.cuda.memory_allocated() / 2**30:.1f} GB")

# %% [markdown]
# ## Actions → tokens (OpenVLA's action tokenizer)
# Normalize each dimension with its 1st/99th percentile to [-1, 1], cut into 256 bins, and use the **last 256 ids of the
# 32,000-token Llama vocabulary**. Careful: the model's embedding table is padded to 32,064 rows, but the action ids count
# down from the *tokenizer's* vocabulary size (32,000), the same as in OpenVLA's own code.

# %%
A = np.stack([f[2] for f in frames]); q01, q99 = np.percentile(A, 1, 0), np.percentile(A, 99, 0)
VOCAB = tok.vocab_size                                  # 32000
BINS = np.linspace(-1, 1, 256); CENTERS = (BINS[:-1] + BINS[1:]) / 2

def action_to_tokens(a):
    z = np.clip(2 * (a - q01) / np.maximum(q99 - q01, 1e-8) - 1, -1, 1)
    return VOCAB - np.digitize(z, BINS).clip(1, 255)

def tokens_to_action(t):
    z = CENTERS[np.clip(VOCAB - np.asarray(t) - 1, 0, 254)]
    return (z + 1) / 2 * (q99 - q01) + q01

def prompt_ids(instruction):
    ids = tok(f"In: What action should the robot take to {instruction}?\nOut:").input_ids
    return ids + [29871] if ids[-1] != 29871 else ids    # OpenVLA's predict_action appends this empty token too

a0 = frames[0][2]
print("action", a0.round(3), "\ntokens", action_to_tokens(a0), "\nround trip", tokens_to_action(action_to_tokens(a0)).round(3))

# %% [markdown]
# ## Fine-tune
# Each example is `[BOS] prompt [29871] a1 … a7 [EOS]`, right-padded. Only the action tokens and EOS carry a loss.

# %%
def batch(samples):
    seqs, labs = [], []
    for img, instr, a in samples:
        p, act = prompt_ids(instr), action_to_tokens(a).tolist() + [tok.eos_token_id]
        seqs.append(p + act); labs.append([-100] * len(p) + act)
    L = max(map(len, seqs))
    ids = torch.full((len(seqs), L), tok.pad_token_id); lab = torch.full((len(seqs), L), -100); att = torch.zeros((len(seqs), L), dtype=torch.long)
    for i, (s, l) in enumerate(zip(seqs, labs)):
        ids[i, :len(s)] = torch.tensor(s); lab[i, :len(l)] = torch.tensor(l); att[i, :len(s)] = 1
    pix = proc.image_processor([Image.fromarray(x[0]) for x in samples], return_tensors="pt")["pixel_values"]
    return ids.cuda(), att.cuda(), lab.cuda(), pix.to("cuda", PIX_DTYPE)

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
rng = np.random.default_rng(0); model.train(); t0 = time.time()
for step in range(STEPS):
    ids, att, lab, pix = batch([frames[i] for i in rng.integers(0, len(frames), BS)])
    out = model(input_ids=ids, attention_mask=att, pixel_values=pix, labels=lab)
    opt.zero_grad(); out.loss.backward(); opt.step()
    if step % 100 == 0 or step == STEPS - 1:
        # token accuracy on this batch: logits at position i predict token i+1; the vision patches are inserted after BOS
        n_patch = out.logits.shape[1] - ids.shape[1]
        pred = out.logits[:, n_patch:-1].argmax(-1); tgt = lab[:, 1:]; m = tgt[:, :] != -100
        acc = (pred[m] == tgt[m]).float().mean().item()
        print(f"step {step:5d}  loss {out.loss.item():.3f}  action-token accuracy {acc:.2f}  ({time.time() - t0:.0f}s)")
model.save_pretrained("/content/openvla_tabletop_lora")

# %% [markdown]
# ## Open-loop check on 20 held-out demonstrations
# Predict 7 tokens greedily from each frame and compare with the demonstrator's action. Baseline: always predicting the
# mean training action. If the model doesn't clearly beat the baseline here, there is no point running it on the robot.

# %%
@torch.no_grad()
def predict(img, instruction):
    p = torch.tensor([prompt_ids(instruction)], device="cuda")
    pix = proc.image_processor([Image.fromarray(img)], return_tensors="pt")["pixel_values"].to("cuda", PIX_DTYPE)
    out = model.generate(input_ids=p, pixel_values=pix, max_new_tokens=7, do_sample=False)
    return tokens_to_action(out[0, -7:].cpu().numpy())

model.eval()
errs, base = [], []
for e in test_eps:
    for t in range(0, len(e["action"]), 4):
        a_hat = predict(e["front"][t], e["instruction"])
        errs.append(np.abs(a_hat - e["action"][t])); base.append(np.abs(A.mean(0) - e["action"][t]))
errs, base = np.array(errs), np.array(base)
print("mean |error| per dim, model   :", errs.mean(0).round(4))
print("mean |error| per dim, baseline:", base.mean(0).round(4))
print(f"position error: model {1000 * errs[:, :3].mean():.1f} mm vs baseline {1000 * base[:, :3].mean():.1f} mm")

# %% [markdown]
# ## Closed loop in the simulator
# OpenVLA is a single-step token policy at ~3–6 Hz on Colab, so 10 episodes take several minutes. In S10 our own token
# VLA scored 0% here while its token accuracy looked fine: measure it, don't assume it.

# %%
from vlacourse import TableTop, evaluate
import imageio
from IPython.display import Video, display

class OpenVLAPolicy:
    def __init__(self, env): self.env = env
    def act(self, obs):
        a = predict(obs["images"]["front"], self.env.task["instruction"]).astype(np.float32)
        a[6] = float(a[6] > 0.5); return a

env = TableTop(n_cubes=3, n_trays=1, cameras=("front",), image_size=(128, 128), seed=0, max_steps=100)
t0 = time.time()
r = evaluate(env, OpenVLAPolicy(env), n_episodes=10, seed=50_000, record=3)
print(f"success {r['success']:.0%} (95% CI {r['ci'][0]:.0%}–{r['ci'][1]:.0%}), {time.time() - t0:.0f}s")
imageio.mimwrite("/content/openvla_tabletop.mp4", [f for v in r["videos"] for f in v], fps=10)
display(Video("/content/openvla_tabletop.mp4", embed=True, width=384))

# %% [markdown]
# **Exercises**
# 1. Compare the open-loop errors per dimension: which one is worst, and is it the one the closed-loop failures point to?
# 2. Train for 3× more steps. Does open-loop error keep dropping while closed-loop success doesn't? (Compounding errors, S06.)
# 3. Replace the single action with a chunk: predict 7 × 4 tokens (four future actions) and execute them all. What does it
#    cost in tokens per second, and what does it buy? (This is the road to FAST and π0, S10–S11.)
