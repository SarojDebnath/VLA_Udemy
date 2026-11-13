# %% [markdown]
# # 68 · Where stitching breaks: return-conditioned imitation vs offline Q-learning

# %%
import numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt

torch.set_num_threads(4)
L = 6
# states: path A (start S, 0..L-1) -> hub X (index 2L) ; path B (start Z, L..2L-1) -> X ; X -> path to G (good) or Y (bad)
S, Z, X = 0, L, 2 * L
N_STATES = 2 * L + 1 + 2 * L
def step(s, a):
    """action 0/1. Paths A and B move forward with either action. At X, a=1 leads to the goal branch, a=0 to the dead branch."""
    if s < 2 * L:
        nxt = s + 1 if (s + 1) % L else X
        return nxt, (1.0 if s < L and nxt == X else 0.0), False
    if s == X:
        return (X + 1 if a == 1 else X + 1 + L), 0.0, False
    if X + 1 <= s < X + L:
        return s + 1, (1.0 if s + 1 == X + L else 0.0), s + 1 == X + L
    if X + 1 + L <= s < X + 2 * L:
        return s + 1, 0.0, s + 1 == X + 2 * L
    return s, 0.0, True

def trajectory(start, choice):
    s, out, done = start, [], False
    while not done:
        a = choice if s == X else np.random.randint(2)
        s2, r, done = step(s, a); out.append((s, a, r)); s = s2
    return out

DATA = [trajectory(S, 0) for _ in range(200)] + [trajectory(Z, 1) for _ in range(200)]
print("dataset: start S -> X -> dead end (return 1)  and  start Z -> X -> goal (return 1)")
print("the best behavior from S (S -> X -> goal, return 2) never appears in any single trajectory")

# %%
def offline_q(data, gamma=0.99, iters=200):
    Q = np.zeros((N_STATES, 2))
    trans = [(s, a, r, (traj[i + 1][0] if i + 1 < len(traj) else None)) for traj in data for i, (s, a, r) in enumerate(traj)]
    for _ in range(iters):
        for s, a, r, s2 in trans:
            Q[s, a] = r + (gamma * Q[s2].max() if s2 is not None else 0.0)
    return Q

def run_policy(choose, start=S):
    s, total, done, hist = start, 0.0, False, []
    while not done:
        a = choose(s, hist); s2, r, done = step(s, a); hist.append((s, a, r)); total += r; s = s2
    return total

Q = offline_q(DATA)
print("offline Q-learning from S: return", run_policy(lambda s, h: int(Q[s].argmax())), "(stitches through X)")

# %%
K = 3 * L
class DT(nn.Module):
    def __init__(self, d=64):
        super().__init__()
        self.eR, self.es, self.ea, self.et = nn.Linear(1, d), nn.Embedding(N_STATES, d), nn.Embedding(2, d), nn.Embedding(64, d)
        self.tf = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, 4 * d, 0.0, batch_first=True, norm_first=True), 2)
        self.head = nn.Linear(d, 2)
    def forward(self, R, s, a, t):
        B, T = s.shape; te = self.et(t)
        x = torch.stack([self.eR(R) + te, self.es(s) + te, self.ea(a) + te], 2).reshape(B, 3 * T, -1)
        h = self.tf(x, mask=torch.triu(torch.ones(3 * T, 3 * T, dtype=torch.bool), 1)).reshape(B, T, 3, -1)
        return self.head(h[:, :, 1])

def encode(traj):
    r = np.array([x[2] for x in traj]); rtg = np.cumsum(r[::-1])[::-1].copy()
    return rtg, np.array([x[0] for x in traj]), np.array([x[1] for x in traj])

dt = DT(); opt = torch.optim.AdamW(dt.parameters(), 1e-3)
enc = [encode(t) for t in DATA]
for it in range(1500):
    batch = [enc[i] for i in np.random.randint(len(enc), size=64)]
    T = max(len(b[1]) for b in batch)
    R = torch.zeros(64, T, 1); Sx = torch.zeros(64, T, dtype=torch.long); A = torch.zeros(64, T, dtype=torch.long); M = torch.zeros(64, T)
    for i, (rt, s, a) in enumerate(batch):
        n = len(s); R[i, :n, 0] = torch.tensor(rt, dtype=torch.float32); Sx[i, :n] = torch.tensor(s); A[i, :n] = torch.tensor(a); M[i, :n] = 1
    logits = dt(R, Sx, A, torch.arange(T)[None].repeat(64, 1))
    loss = (nn.functional.cross_entropy(logits.reshape(-1, 2), A.reshape(-1), reduction="none") * M.reshape(-1)).sum() / M.sum()
    opt.zero_grad(); loss.backward(); opt.step()

def dt_policy(target):
    state = {"R": [target]}
    def choose(s, hist):
        rt = [target - sum(h[2] for h in hist[:i]) for i in range(len(hist) + 1)]
        ss = [h[0] for h in hist] + [s]; aa = [h[1] for h in hist] + [0]
        with torch.no_grad():
            lg = dt(torch.tensor(rt, dtype=torch.float32)[None, :, None], torch.tensor(ss)[None], torch.tensor(aa)[None], torch.arange(len(ss))[None])
        return int(lg[0, -1].argmax())
    return choose

for target in [1.0, 2.0]:
    rets = [run_policy(dt_policy(target)) for _ in range(20)]
    print(f"Decision Transformer from S, requested return {target}: mean achieved {np.mean(rets):.2f}  (choice at X -> goal in {np.mean([r == 2 for r in rets]):.0%})")

# %%
with torch.no_grad():
    for target in [1.0, 2.0]:
        hist = [(S + i, 0, 0.0) for i in range(L - 1)] + [(L - 1, 0, 1.0)]
        rt = [target - sum(h[2] for h in hist[:i]) for i in range(len(hist) + 1)]
        ss = [h[0] for h in hist] + [X]; aa = [h[1] for h in hist] + [0]
        p = dt(torch.tensor(rt, dtype=torch.float32)[None, :, None], torch.tensor(ss)[None], torch.tensor(aa)[None], torch.arange(len(ss))[None])[0, -1].softmax(-1)
        print(f"at hub X after coming from S (RTG left = {rt[-1]:.0f}): P(goal branch) = {p[1]:.2f}")
print("offline Q at X:", Q[X].round(3), "-> greedy picks the goal branch regardless of how X was reached")
