# %% [markdown]
# # 123 · Your own evaluation suite: robustness of ACT / Diffusion / Flow to camera, lighting, texture and friction changes

# %%
import os, json, numpy as np, torch, matplotlib.pyplot as plt
from vlacourse import TableTop
from vlacourse.torch_data import TableTopDataset
from vlacourse.chunk_policies import ACT, DiffusionPolicy, FlowPolicy, ChunkPolicy
from vlacourse.eval_suite import run_suite, SUITE

from vlacourse import scene_dataset
SCENE = "single"                    # TableTop scene tier (see vlacourse.SCENES)
DATA = scene_dataset(SCENE)          # 500 demos, recorded once with all CPU cores if missing
CK = "../S08_Generative_Action_Heads/checkpoints"
from vlacourse import need; [need(f"{CK}/{n}.pt", f"S08 notebook {k}") for n, k in [("act", 74), ("diffusion", 78), ("flow", 81)]]
device = "cuda" if torch.cuda.is_available() else "cpu"
H = 16

def load(cls, name):
    net = cls(horizon=H); net.load_state_dict(torch.load(f"{CK}/{name}.pt", map_location=device)); return net.to(device).eval()

# %%
ds = TableTopDataset(DATA, episodes=range(450), horizon=H)
env = TableTop(scene=SCENE, cameras=("front", "wrist"), image_size=(128, 128), seed=0)
policies = {"ACT": ChunkPolicy(load(ACT, "act"), ds, env, device, ensemble=True),
            "Diffusion": ChunkPolicy(load(DiffusionPolicy, "diffusion"), ds, env, device, execute=8, nfe=10),
            "Flow": ChunkPolicy(load(FlowPolicy, "flow"), ds, env, device, execute=8, nfe=4)}
results = {}
for name, pol in policies.items():
    results[name] = run_suite(env, pol, n_episodes=30)
    print(name, {k: round(v["success"], 2) for k, v in results[name].items()})

# %%
conds = [p.name for p in SUITE]
M = np.array([[results[n][c]["success"] for c in conds] for n in policies])
plt.figure(figsize=(11, 3)); plt.imshow(M, cmap="RdYlGn", vmin=0, vmax=1)
plt.xticks(range(len(conds)), conds, rotation=25, ha="right"); plt.yticks(range(len(policies)), list(policies))
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        plt.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center")
plt.title("robustness matrix (30 paired episodes per cell)"); plt.colorbar(); plt.show()

# %%
os.makedirs("results", exist_ok=True)
json.dump({n: {c: dict(success=results[n][c]["success"], ci=results[n][c]["ci"]) for c in conds} for n in policies}, open("results/robustness_matrix.json", "w"), indent=1)
drops = {n: {c: results[n]["nominal"]["success"] - results[n][c]["success"] for c in conds[1:]} for n in policies}
for n, d in drops.items():
    worst = max(d, key=d.get)
    print(f"{n:9s} largest drop: {worst} (-{d[worst]:.2f})")

# %% [markdown]
# ### Watch it: one policy, three worlds
# The same policy and episode under nominal conditions, a moved camera and dim light.

# %%
from vlacourse import showcase as sc
from vlacourse.eval_suite import _restore_all
def perturbed(fn):
    return lambda e: (_restore_all(e), fn(e))
sfx = {p.name: p.apply_fn for p in SUITE}
pol = policies["Flow"]
sc.compare(env, {"nominal": pol, "camera moved 6 cm": pol, "dim light": pol}, seed=200_002, size=(300, 400),
           per_policy_kw={"nominal": dict(on_reset=perturbed(sfx["nominal"])), "camera moved 6 cm": dict(on_reset=perturbed(sfx["camera +6 cm / lower"])),
                          "dim light": dict(on_reset=perturbed(sfx["dim light (x0.5)"]))}, title="123 one policy, three worlds")
_restore_all(env)
