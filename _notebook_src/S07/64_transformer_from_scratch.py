# %% [markdown]
# # 64 · A GPT-style transformer from scratch: a sorting sanity check, then next-action prediction on TableTop

# %%
import math, time, numpy as np, torch, torch.nn as nn, matplotlib.pyplot as plt
from vlacourse import TableTop, ScriptedExpert, evaluate
from vlacourse.features import privileged_obs, PRIV_DIM

torch.set_num_threads(4)
device = "cuda" if torch.cuda.is_available() else "cpu"

class Block(nn.Module):
    def __init__(self, d, n_head, dropout=0.0):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj, self.n_head = nn.Linear(d, 3 * d), nn.Linear(d, d), n_head
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d), nn.Dropout(dropout))

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.n_head, D // self.n_head).permute(2, 0, 3, 1, 4)
        y = nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(y.transpose(1, 2).reshape(B, T, D))
        return x + self.mlp(self.ln2(x))

class GPT(nn.Module):
    def __init__(self, vocab=None, d_in=None, d_out=None, d=128, n_layer=3, n_head=4, max_T=128):
        super().__init__()
        self.tok = nn.Embedding(vocab, d) if vocab else nn.Linear(d_in, d)
        self.pos = nn.Embedding(max_T, d)
        self.blocks = nn.Sequential(*[Block(d, n_head) for _ in range(n_layer)])
        self.ln, self.head = nn.LayerNorm(d), nn.Linear(d, vocab or d_out)

    def forward(self, x):
        h = self.tok(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        return self.head(self.ln(self.blocks(h)))

def sort_batch(B, n=8, vocab=10):
    x = torch.randint(0, vocab, (B, n))
    seq = torch.cat([x, x.sort(-1).values], 1)
    return seq[:, :-1], seq[:, 1:]

# %%
torch.manual_seed(0)
gpt = GPT(vocab=10, d=64, n_layer=2, max_T=16).to(device)
opt = torch.optim.AdamW(gpt.parameters(), 1e-3)
for step in range(2000):
    x, y = sort_batch(256)
    logits = gpt(x.to(device))
    loss = nn.functional.cross_entropy(logits[:, 7:].reshape(-1, 10), y[:, 7:].reshape(-1).to(device))
    opt.zero_grad(); loss.backward(); opt.step()
x, y = sort_batch(1000)
seq = x[:, :8].to(device)
with torch.no_grad():
    for _ in range(8):
        seq = torch.cat([seq, gpt(seq)[:, -1].argmax(-1, keepdim=True)], 1)
acc = (seq[:, 8:].cpu() == y[:, 7:]).all(1).float().mean()
print(f"sorting 8 digits autoregressively: {acc:.1%} of sequences exactly right (params {sum(p.numel() for p in gpt.parameters()) / 1e3:.0f}k)")

# %%
def collect(env, n, seed=0):
    eps = []
    for e in range(n):
        env.reset(seed=seed + e); ex = ScriptedExpert(env, seed=e, speed=np.random.default_rng(e).uniform(0.6, 1.0)); done = False; O, A = [], []
        while not done:
            o = privileged_obs(env); a = ex.act(); O.append(o); A.append(a); _, _, done, _ = env.step(a)
        eps.append((np.array(O), np.array(A)))
    return eps

def windows(eps, K):
    X, Y = [], []
    for O, A in eps:
        Op = np.concatenate([np.repeat(O[:1], K - 1, 0), O]); Ap = np.concatenate([np.repeat(A[:1], K - 1, 0), A])
        for t in range(len(O)):
            X.append(Op[t:t + K]); Y.append(Ap[t:t + K])
    return np.array(X, np.float32), np.array(Y, np.float32)

env = TableTop(cameras=(), seed=0)
eps = collect(env, 150)
K = 8
X, Y = windows(eps, K)
mx, sx, my, sy = X.reshape(-1, PRIV_DIM).mean(0), X.reshape(-1, PRIV_DIM).std(0) + 1e-6, Y.reshape(-1, 7).mean(0), Y.reshape(-1, 7).std(0) + 1e-6
Xt, Yt = torch.tensor((X - mx) / sx), torch.tensor((Y - my) / sy)
print("training windows:", Xt.shape, "->", Yt.shape)

# %%
torch.manual_seed(0)
policy_net = GPT(d_in=PRIV_DIM, d_out=7, d=128, n_layer=3, max_T=K).to(device)
opt = torch.optim.AdamW(policy_net.parameters(), 3e-4, weight_decay=1e-4)
t0, losses = time.time(), []
for epoch in range(25):
    for idx in torch.randperm(len(Xt)).split(256):
        loss = nn.functional.mse_loss(policy_net(Xt[idx].to(device)), Yt[idx].to(device))
        opt.zero_grad(); loss.backward(); opt.step()
    losses.append(loss.item())
print(f"trained in {time.time() - t0:.0f}s, final loss {losses[-1]:.4f}")

class HistoryPolicy:
    def __init__(self, env):
        self.env = env
    def reset(self):
        self.hist = []
    def act(self, obs):
        self.hist.append((privileged_obs(self.env) - mx) / sx)
        h = self.hist[-K:]; h = [h[0]] * (K - len(h)) + h
        with torch.no_grad():
            a = policy_net(torch.tensor(np.array(h), dtype=torch.float32)[None].to(device))[0, -1].cpu().numpy() * sy + my
        a[6] = float(a[6] > 0.5)
        return a

res = evaluate(env, HistoryPolicy(env), n_episodes=60, seed=50_000)
print(f"causal transformer over the last {K} states: success {res['success']:.2f} CI {np.round(res['ci'], 2)} per task {res['per_task']}")
