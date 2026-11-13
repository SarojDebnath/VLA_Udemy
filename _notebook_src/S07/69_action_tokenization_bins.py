# %% [markdown]
# # 69 · Tokenize the action and pay the resolution bill: Decision Transformer with 256 → 4 bins

# %%
import os, time, importlib.util, numpy as np, torch, matplotlib.pyplot as plt

spec = importlib.util.spec_from_file_location("dtnb", os.path.abspath("dt_lib.py"))
dtnb = importlib.util.module_from_spec(spec); spec.loader.exec_module(dtnb)

# %%
data = list(np.load("generated/pendulum_offline.npy", allow_pickle=True))
best = max(e["rew"].sum() for e in data)
cont = dtnb.DecisionTransformer().to(dtnb.device)
cont.load_state_dict(torch.load("generated/dt_pendulum.pt", map_location=dtnb.device)); cont.eval()
base = np.mean([dtnb.rollout_dt(cont, best, seed=3000 + s) for s in range(5)])
print(f"continuous-action DT (MSE head): {base:.0f}")

# %%
results = {}
for bins in [256, 64, 16, 8, 4, 2]:
    t0 = time.time()
    dt = dtnb.train_dt(data, steps=4000, bins=bins); dt.eval()
    results[bins] = [dtnb.rollout_dt(dt, best, seed=3000 + s) for s in range(5)]
    print(f"{bins:4d} bins (resolution {2 / bins:.3f} N·m-units): achieved {np.mean(results[bins]):7.0f} ± {np.std(results[bins]):4.0f}   ({time.time() - t0:.0f}s)")

# %%
b = sorted(results)
plt.figure(figsize=(7, 4))
plt.errorbar(b, [np.mean(results[k]) for k in b], yerr=[np.std(results[k]) for k in b], marker="o", capsize=4, label="tokenized actions (cross-entropy)")
plt.axhline(base, c="k", ls="--", label="continuous actions (MSE)")
plt.xscale("log", base=2); plt.xlabel("bins per action dimension"); plt.ylabel("achieved return"); plt.legend(); plt.title("too few bins: the controller cannot be precise"); plt.show()
