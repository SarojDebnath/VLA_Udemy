# %% [markdown]
# # 119 · Success-rate statistics: a success rate is a draw from a distribution

# %%
import numpy as np, matplotlib.pyplot as plt
from scipy.stats import norm, binom
from vlacourse import wilson

def wald(k, n, conf=0.95):
    z = norm.ppf(1 - (1 - conf) / 2); p = k / n; h = z * np.sqrt(p * (1 - p) / n)
    return p - h, p + h

def coverage(interval, p, n, trials=20000, seed=0):
    k = np.random.default_rng(seed).binomial(n, p, trials)
    lo_hi = np.array([interval(ki, n) for ki in range(n + 1)])
    return np.mean((lo_hi[k, 0] <= p) & (p <= lo_hi[k, 1]))

# %%
true_p, n = 0.75, 25
rng = np.random.default_rng(0)
seeds = rng.binomial(n, true_p, 5)
print("five seeds of the SAME policy (true success 0.75, 25 episodes each):", [f"{k}/{n}" for k in seeds])
for k in seeds:
    print(f"  {k}/{n} = {k / n:.0%}   Wilson 95% CI [{wilson(k, n)[0]:.0%}, {wilson(k, n)[1]:.0%}]")
K, N = seeds.sum(), n * len(seeds)
print(f"pooled {K}/{N} = {K / N:.0%}  CI [{wilson(K, N)[0]:.0%}, {wilson(K, N)[1]:.0%}]  (width shrinks like 1/sqrt(n))")

# %%
ps = np.linspace(0.02, 0.98, 49)
plt.figure(figsize=(8, 4))
for name, f in [("Wald (textbook normal)", wald), ("Wilson", wilson)]:
    plt.plot(ps, [coverage(f, p, 25) for p in ps], label=name)
plt.axhline(0.95, c="k", ls="--"); plt.ylim(0.5, 1.0); plt.xlabel("true success rate"); plt.ylabel("actual coverage of the '95%' interval")
plt.title("n = 25: Wald collapses near 0 and 1, Wilson stays honest"); plt.legend(); plt.show()
print("Wald interval for 25/25 successes:", wald(25, 25), "  <- zero width, absurd;  Wilson:", np.round(wilson(25, 25), 3))

# %%
ns = np.arange(5, 401, 5)
plt.figure(figsize=(8, 4))
for p in [0.5, 0.8, 0.95]:
    plt.plot(ns, [wilson(int(round(p * n)), n)[1] - wilson(int(round(p * n)), n)[0] for n in ns], label=f"p = {p}")
plt.xlabel("episodes"); plt.ylabel("95% CI width"); plt.legend(); plt.title("to halve the interval, run 4x the episodes"); plt.show()
for n in [10, 25, 50, 100, 200]:
    lo, hi = wilson(int(0.7 * n), n)
    print(f"70% measured with n={n:3d}: CI [{lo:.2f}, {hi:.2f}]  width {hi - lo:.2f}")
