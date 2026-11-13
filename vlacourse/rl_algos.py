"""Compact PPO and SAC implementations (the same algorithms written from scratch in S05 notebooks 48 and 50)."""
import copy

import numpy as np
import torch
import torch.nn as nn


def evaluate_policy(env, act_fn, n=5, seed=10 ** 6):
    rets = []
    for k in range(n):
        obs, _ = env.reset(seed=seed + k)
        done, R = False, 0.0
        while not done:
            obs, r, te, tr, _ = env.step(act_fn(obs))
            R += r
            done = te or tr
        rets.append(R)
    return float(np.mean(rets))


# ----------------------------- PPO -----------------------------
class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, discrete, hidden=64):
        super().__init__()
        self.discrete = discrete
        self.pi = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, act_dim))
        self.v = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.log_std = None if discrete else nn.Parameter(-0.5 * torch.ones(act_dim))

    def dist(self, obs):
        out = self.pi(obs)
        return torch.distributions.Categorical(logits=out) if self.discrete else torch.distributions.Normal(out, self.log_std.exp())

    def logp(self, d, a):
        return d.log_prob(a) if self.discrete else d.log_prob(a).sum(-1)

    def act(self, obs):
        with torch.no_grad():
            out = self.pi(torch.as_tensor(obs, dtype=torch.float32))
        return int(out.argmax()) if self.discrete else out.numpy()


def ppo(env, discrete, total_steps, n_steps=2048, epochs=10, mb=64, clip=0.2, lr=3e-4, gamma=0.99, lam=0.95,
        seed=0, eval_env_fn=None, eval_every=None):
    torch.manual_seed(seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n if discrete else env.action_space.shape[0]
    ac = ActorCritic(obs_dim, act_dim, discrete)
    opt = torch.optim.Adam(ac.parameters(), lr=lr)
    obs, _ = env.reset(seed=seed)
    ep_ret, ep_len, finished, log, steps, it = 0.0, 0, [], [], 0, 0
    while steps < total_steps:
        O, A, LP, R, D, V = [], [], [], [], [], []
        for _ in range(n_steps):
            o = torch.as_tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                d = ac.dist(o); a = d.sample(); lp = ac.logp(d, a); v = ac.v(o)[0]
            obs2, r, te, tr, info = env.step(a.item() if discrete else a.numpy())
            ep_ret += r; ep_len += 1
            if tr and not te:
                with torch.no_grad():
                    r = r + gamma * ac.v(torch.as_tensor(obs2, dtype=torch.float32))[0].item()
            O.append(o); A.append(a); LP.append(lp); R.append(r); D.append(float(te or tr)); V.append(v)
            if te or tr:
                finished.append((ep_ret, ep_len, info.get("success", False))); ep_ret, ep_len = 0.0, 0
                obs2, _ = env.reset()
            obs = obs2
        steps += n_steps; it += 1
        with torch.no_grad():
            last_v = ac.v(torch.as_tensor(obs, dtype=torch.float32))[0]
        O, A, LP, V = torch.stack(O), torch.stack(A), torch.stack(LP), torch.stack(V)
        R, D = torch.tensor(R, dtype=torch.float32), torch.tensor(D)
        adv, g = torch.zeros_like(R), 0.0
        for t in reversed(range(n_steps)):
            nv = last_v if t == n_steps - 1 else V[t + 1]
            delta = R[t] + gamma * nv * (1 - D[t]) - V[t]
            g = delta + gamma * lam * (1 - D[t]) * g
            adv[t] = g
        ret = adv + V
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        kls = []
        for _ in range(epochs):
            for idx in torch.randperm(n_steps).split(mb):
                d = ac.dist(O[idx]); lp = ac.logp(d, A[idx]); ratio = (lp - LP[idx]).exp()
                pg = -torch.min(ratio * adv[idx], ratio.clamp(1 - clip, 1 + clip) * adv[idx]).mean()
                vl = ((ac.v(O[idx]).squeeze(-1) - ret[idx]) ** 2).mean()
                opt.zero_grad(); (pg + 0.5 * vl).backward(); nn.utils.clip_grad_norm_(ac.parameters(), 0.5); opt.step()
                kls.append(((ratio - 1) - (lp - LP[idx])).mean().item())
        recent = finished[-20:]
        entry = dict(steps=steps, ret=np.mean([x[0] for x in recent]) if recent else np.nan,
                     success=np.mean([x[2] for x in recent]) if recent else np.nan, kl=float(np.mean(kls)))
        if eval_env_fn is not None and eval_every and it % eval_every == 0:
            entry["eval"] = evaluate_policy(eval_env_fn(), ac.act)
        log.append(entry)
    return ac, log


# ----------------------------- SAC -----------------------------
def _mlp(i, o, h=256):
    return nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h), nn.ReLU(), nn.Linear(h, o))


class SquashedGaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = _mlp(obs_dim, 2 * act_dim)

    def forward(self, obs, deterministic=False):
        mu, log_std = self.net(obs).chunk(2, -1)
        std = log_std.clamp(-5, 2).exp()
        u = mu if deterministic else mu + std * torch.randn_like(mu)
        logp = torch.distributions.Normal(mu, std).log_prob(u).sum(-1) - (2 * (np.log(2) - u - nn.functional.softplus(-2 * u))).sum(-1)
        return torch.tanh(u), logp

    def act(self, obs):
        with torch.no_grad():
            return self(torch.as_tensor(obs, dtype=torch.float32), deterministic=True)[0].numpy()


def sac(env, total_steps=20_000, start=1000, batch=256, gamma=0.99, tau=0.005, lr=3e-4, seed=0):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    od, ad = env.observation_space.shape[0], env.action_space.shape[0]
    actor, q1, q2 = SquashedGaussianActor(od, ad), _mlp(od + ad, 1), _mlp(od + ad, 1)
    q1_t, q2_t = copy.deepcopy(q1), copy.deepcopy(q2)
    log_alpha = torch.zeros(1, requires_grad=True)
    opt_a = torch.optim.Adam(actor.parameters(), lr)
    opt_q = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr)
    opt_al = torch.optim.Adam([log_alpha], lr)
    N = total_steps
    B = dict(o=np.zeros((N, od), np.float32), a=np.zeros((N, ad), np.float32), r=np.zeros(N, np.float32),
             o2=np.zeros((N, od), np.float32), d=np.zeros(N, np.float32))
    obs, _ = env.reset(seed=seed)
    for t in range(N):
        if t < start:
            a = env.action_space.sample()
        else:
            with torch.no_grad():
                a = actor(torch.as_tensor(obs, dtype=torch.float32))[0].numpy()
        obs2, r, te, tr, _ = env.step(a)
        for k, v in zip("o a r o2 d".split(), (obs, a, r, obs2, float(te))):
            B[k][t] = v
        obs = obs2
        if te or tr:
            obs, _ = env.reset()
        if t >= start:
            idx = rng.integers(0, t + 1, batch)
            o, a_b, r_b, o2, d = (torch.as_tensor(B[k][idx]) for k in "o a r o2 d".split())
            alpha = log_alpha.exp().detach()
            with torch.no_grad():
                a2, lp2 = actor(o2)
                y = r_b + gamma * (1 - d) * (torch.min(q1_t(torch.cat([o2, a2], -1)), q2_t(torch.cat([o2, a2], -1))).squeeze(-1) - alpha * lp2)
            oa = torch.cat([o, a_b], -1)
            lq = ((q1(oa).squeeze(-1) - y) ** 2).mean() + ((q2(oa).squeeze(-1) - y) ** 2).mean()
            opt_q.zero_grad(); lq.backward(); opt_q.step()
            a_new, lp = actor(o)
            la = (alpha * lp - torch.min(q1(torch.cat([o, a_new], -1)), q2(torch.cat([o, a_new], -1))).squeeze(-1)).mean()
            opt_a.zero_grad(); la.backward(); opt_a.step()
            lal = -(log_alpha * (lp.detach() - ad)).mean()
            opt_al.zero_grad(); lal.backward(); opt_al.step()
            with torch.no_grad():
                for net, tgt in [(q1, q1_t), (q2, q2_t)]:
                    for p, pt in zip(net.parameters(), tgt.parameters()):
                        pt.mul_(1 - tau).add_(tau * p)
    return actor
