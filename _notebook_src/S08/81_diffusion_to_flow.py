# %% [markdown]
# # 81 · Swap the diffusion head for flow matching: same backbone, same data; success vs network evaluations

# %%
import os, time, numpy as np, torch, matplotlib.pyplot as plt
from vlacourse import TableTop, evaluate
from vlacourse.torch_data import TableTopDataset
from vlacourse.chunk_policies import DiffusionPolicy, FlowPolicy, train_policy, ChunkPolicy

from vlacourse import scene_dataset
SCENE = "single"                    # TableTop scene tier (see vlacourse.SCENES)
DATA = scene_dataset(SCENE)          # 500 demos, recorded once with all CPU cores if missing
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.8)
H = 16

# %%
ds = TableTopDataset(DATA, episodes=range(450), horizon=H)
flow = FlowPolicy(horizon=H)
if os.path.exists("checkpoints/flow.pt"):
    flow.load_state_dict(torch.load("checkpoints/flow.pt", map_location=device)); flow.to(device)
else:
    t0 = time.time(); losses = train_policy(flow, ds, epochs=60, lr=1e-4, device=device, log_every=5)
    torch.save(flow.state_dict(), "checkpoints/flow.pt"); print(f"trained in {time.time() - t0:.0f}s")
diff = DiffusionPolicy(horizon=H)
diff.load_state_dict(torch.load("checkpoints/diffusion.pt", map_location=device)); diff.to(device)

# %%
env = TableTop(scene=SCENE, cameras=("front", "wrist"), image_size=(128, 128), seed=0)
nfes = [1, 2, 4, 10]
table = {}
for name, net in [("diffusion (DDIM)", diff), ("flow (Euler)", flow)]:
    table[name] = []
    for k in nfes:
        t0 = time.time()
        r = evaluate(env, ChunkPolicy(net, ds, env, device, execute=8, nfe=k), n_episodes=40, seed=50_000)
        table[name].append((r["success"], *r["ci"], time.time() - t0))
        print(f"{name:17s} NFE={k:2d}: success {r['success']:.2f} CI [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}]  eval time {time.time() - t0:.0f}s")

# %%
plt.figure(figsize=(7, 4.5))
for name, rows in table.items():
    rows = np.array(rows)
    plt.errorbar(nfes, rows[:, 0], yerr=[rows[:, 0] - rows[:, 1], rows[:, 2] - rows[:, 0]], marker="o", capsize=3, label=name)
plt.xscale("log"); plt.xlabel("network evaluations per chunk (NFE)"); plt.ylabel("success (40 episodes)"); plt.ylim(0, 1); plt.legend()
plt.title("the generator is interchangeable; the compute is not"); plt.show()

# %%
b = {k: v[None].to(device) for k, v in ds[0].items() if k != "pad_mask"}
for name, net in [("diffusion", diff), ("flow", flow)]:
    for k in [1, 10]:
        torch.cuda.synchronize() if device == "cuda" else None
        t0 = time.perf_counter()
        for _ in range(20):
            net.sample(b, nfe=k)
        torch.cuda.synchronize() if device == "cuda" else None
        print(f"{name:9s} NFE={k:2d}: {(time.perf_counter() - t0) / 20 * 1000:6.1f} ms per chunk")

# %% [markdown]
# ### Watch it: 10 denoising steps vs 1 flow step
# Same episode. Flow matching with a single network call per chunk against DDIM with ten.

# %%
from vlacourse import showcase as sc
sc.compare(env, {"Diffusion, DDIM 10 steps": ChunkPolicy(diff, ds, env, device, execute=8, nfe=10),
                  "Flow matching, 1 Euler step": ChunkPolicy(flow, ds, env, device, execute=8, nfe=1)}, seed=50_001, title="81 diffusion vs flow")
