# %% [markdown]
# # 120 · Comparing two policies honestly: tests, paired seeds, and how many episodes you need

# %%
import os, json, numpy as np, matplotlib.pyplot as plt
from scipy.stats import norm, fisher_exact

def two_prop_z(k1, n1, k2, n2):
    p = (k1 + k2) / (n1 + n2); se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    z = (k1 / n1 - k2 / n2) / se if se > 0 else 0.0
    return z, 2 * (1 - norm.cdf(abs(z)))

def episodes_needed(p1, p2, alpha=0.05, power=0.8):
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power); pb = (p1 + p2) / 2
    return int(np.ceil((za * np.sqrt(2 * pb * (1 - pb)) + zb * np.sqrt(p1 * (1 - p1) + p2 * (1 - p2))) ** 2 / (p1 - p2) ** 2))

def bootstrap_diff(a, b, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(n)]
    return np.percentile(d, [2.5, 97.5])

# %%
for (k1, n1, k2, n2) in [(21, 25, 18, 25), (42, 50, 36, 50), (84, 100, 72, 100)]:
    z, p = two_prop_z(k1, n1, k2, n2); _, pf = fisher_exact([[k1, n1 - k1], [k2, n2 - k2]])
    print(f"A {k1}/{n1}={k1 / n1:.0%} vs B {k2}/{n2}={k2 / n2:.0%}: z-test p={p:.3f}, Fisher exact p={pf:.3f}  -> {'significant' if pf < 0.05 else 'NOT significant'}")
print("same 12-point gap: invisible at n=25 and n=50, still borderline at n=100")

# %%
gaps = [(0.5, 0.6), (0.7, 0.8), (0.8, 0.9), (0.85, 0.9), (0.9, 0.95)]
for p1, p2 in gaps:
    print(f"to detect {p1:.0%} vs {p2:.0%} (α=0.05, power 0.8): {episodes_needed(p1, p2):5d} episodes PER policy")

# %%
rng = np.random.default_rng(1)
difficulty = rng.uniform(0, 1, 100)
a = (rng.uniform(0, 1, 100) > difficulty * 0.6).astype(float)
b = (rng.uniform(0, 1, 100) > difficulty * 0.6 + 0.08).astype(float)
print(f"A {a.mean():.2f} vs B {b.mean():.2f} on the SAME 100 initial conditions")
print("unpaired bootstrap 95% CI of the difference:", bootstrap_diff(a, b).round(3))
d = a - b; boot = [rng.choice(d, len(d)).mean() for _ in range(10000)]
print("paired bootstrap 95% CI of the difference:  ", np.percentile(boot, [2.5, 97.5]).round(3), "<- pairing removes scene difficulty from the noise")
both = ((a == 1) & (b == 0)).sum(), ((a == 0) & (b == 1)).sum()
print(f"McNemar-style discordant pairs: A-only wins {both[0]}, B-only wins {both[1]}")

# %%
path = "../S08_Generative_Action_Heads/results/project6_heads.json"
if os.path.exists(path):
    res = json.load(open(path)); names = list(res)
    print("Project 6 pairwise comparisons (100 episodes each, Fisher exact p-values):")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            k1, k2 = round(res[names[i]]["nominal"] * 100), round(res[names[j]]["nominal"] * 100)
            _, pf = fisher_exact([[k1, 100 - k1], [k2, 100 - k2]])
            print(f"  {names[i]:28s} {k1:3d}%  vs  {names[j]:28s} {k2:3d}%   p={pf:.3f} {'*' if pf < 0.05 else ''}")
else:
    print("run S08 project 84 first to compare its policies here")
