# %% [markdown]
# # 74 · ACT (Action Chunking with Transformers) from scratch on TableTop

# %%
import os, time, numpy as np, torch, matplotlib.pyplot as plt
from vlacourse import TableTop, evaluate, show_video
from vlacourse.torch_data import TableTopDataset
from vlacourse.chunk_policies import ACT, train_policy, ChunkPolicy

from vlacourse import scene_dataset
SCENE = "single"                    # TableTop scene tier (see vlacourse.SCENES)
DATA = scene_dataset(SCENE)          # 500 demos, recorded once with all CPU cores if missing
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    torch.cuda.set_per_process_memory_fraction(0.8)
os.makedirs("checkpoints", exist_ok=True)
H = 16

# %%
ds = TableTopDataset(DATA, episodes=range(450), horizon=H)
net = ACT(horizon=H, kl_weight=10.0)
print(f"ACT parameters: {sum(p.numel() for p in net.parameters()) / 1e6:.1f}M  (ResNet-18 backbone {sum(p.numel() for p in net.obs.parameters()) / 1e6:.1f}M)")
if os.path.exists("checkpoints/act.pt"):
    net.load_state_dict(torch.load("checkpoints/act.pt", map_location=device)); net.to(device); losses = None
else:
    t0 = time.time(); losses = train_policy(net, ds, epochs=50, lr=1e-4, device=device, log_every=5)
    torch.save(net.state_dict(), "checkpoints/act.pt"); print(f"trained in {time.time() - t0:.0f}s")
    plt.figure(figsize=(7, 3.5)); plt.semilogy(np.convolve(losses, np.ones(50) / 50, "valid")); plt.xlabel("step"); plt.ylabel("L1 + 10·KL"); plt.show()

# %%
env = TableTop(scene=SCENE, cameras=("front", "wrist"), image_size=(128, 128), seed=0)
for name, pol in [("ACT, temporal ensembling", ChunkPolicy(net, ds, env, device, ensemble=True)), ("ACT, execute 8 of 16", ChunkPolicy(net, ds, env, device, execute=8))]:
    r = evaluate(env, pol, n_episodes=50, seed=50_000, record=2 if "ensembling" in name else 0)
    print(f"{name:26s} success {r['success']:.2f} CI [{r['ci'][0]:.2f}, {r['ci'][1]:.2f}] per task {r['per_task']}")
    if "ensembling" in name:
        from vlacourse import showcase as sc
        sc.watch(env, pol, seeds=[50_000 + i for i in range(4)], title="74 ACT with temporal ensembling")

# %%
b = {k: v[None].to(device) for k, v in ds[500].items() if k != "pad_mask"}
net.eval()
with torch.no_grad():
    mem = net._memory(b, torch.zeros(1, net.z_dim, device=device), augment=False)
    chunks = [net._decode(net._memory(b, torch.randn(1, net.z_dim, device=device) * 2, augment=False))[0].cpu().numpy() for _ in range(8)]
plt.figure(figsize=(8, 3.5))
for c in chunks:
    plt.plot(c[:, 0], alpha=0.6)
plt.plot(b["action"][0, :, 0].cpu(), "k--", lw=2, label="expert chunk")
plt.xlabel("chunk step"); plt.ylabel("dx (normalized)"); plt.legend(); plt.title("decoding with different latents z (training uses the CVAE posterior)"); plt.show()
