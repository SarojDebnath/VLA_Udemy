"""Capstone C8: a mini driving VLA, in the spirit of NVIDIA's Alpamayo, at laptop scale.

    camera frames (now, 0.4 s ago) ─ ResNet-18 ─┐
    speed, curvature ───────────────── MLP ─────┼─ context ─┬─ flow-matching head → K sampled 2-s plans of (a, κ)
    instruction ─ frozen CLIP text ─── linear ──┘           └─ reasoning head → why ("red light: stopping at the line")

Alpamayo 2 Edge: 6 cameras × up to 5 frames, ego history, navigation text → a 2.45B VLM writes a chain-of-causation
explanation, and a flow-matching action expert outputs 6.4 s of acceleration and curvature. Same recipe, ~1000× smaller.
"""
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .city import A_MAX, A_MIN, DT, K_MAX, PARAPHRASES, REASONS

H = 20                         # plan horizon: 20 steps of 0.1 s = 2 s
HIST = 4                       # the second camera frame is 4 control steps (0.4 s) older
A_SCALE, K_SCALE = 3.0, 0.3    # label normalization
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


# ---------------------------------------------------------------- data
class DriveData:
    """All recorded episodes in memory. A sample = (frame, frame 0.4 s earlier, speed, curvature, command,
    future plan of H (a, κ) labels, reason label). Images stay uint8 (move them to the GPU once with .to(device))."""

    def __init__(self, root, episodes=None, max_episodes=None):
        files = sorted(glob.glob(os.path.join(root, "ep_*.npz")))
        if episodes is not None:
            files = [files[i] for i in episodes]
        if max_episodes:
            files = files[:max_episodes]
        imgs, idx, prev, speed, kap, cmd, plan, reason, ep_id = [], [], [], [], [], [], [], [], []
        n_img = 0
        for e, f in enumerate(files):
            z = np.load(f)
            a, k, steps = z["a"], z["k"], z["img_step"]
            T = len(a)
            k_exec = np.r_[0.0, z["k"][:-1]]            # curvature the car had when the frame was taken (≈ previous label)
            for j, st in enumerate(steps):
                fut = np.arange(st, st + H).clip(max=T - 1)
                plan.append(np.stack([a[fut] / A_SCALE, k[fut] / K_SCALE], 1))
                earlier = np.searchsorted(steps, st - HIST, side="right") - 1
                prev.append(n_img + max(earlier, 0))
                idx.append(n_img + j)
                speed.append(z["speed"][st]); kap.append(k_exec[st]); cmd.append(z["cmd"][st]); reason.append(z["reason"][st])
                ep_id.append(e)
            imgs.append(z["img"]); n_img += len(steps)
        self.img = torch.from_numpy(np.concatenate(imgs))                  # (N, H, W, 3) uint8
        self.idx, self.prev = torch.tensor(idx), torch.tensor(prev)
        self.speed = torch.tensor(np.array(speed), dtype=torch.float32)
        self.kappa = torch.tensor(np.array(kap), dtype=torch.float32)
        self.cmd, self.reason = torch.tensor(np.array(cmd)), torch.tensor(np.array(reason))
        self.plan = torch.tensor(np.array(plan), dtype=torch.float32)       # (N, H, 2) normalized
        self.ep = np.array(ep_id)
        self.n_episodes = len(files)

    def __len__(self):
        return len(self.idx)

    def to(self, device):
        for k in ("img", "idx", "prev", "speed", "kappa", "cmd", "reason", "plan"):
            setattr(self, k, getattr(self, k).to(device))
        return self

    def weights(self):
        """Sampling weights that stop red-light waiting (the most common frame) from dominating training."""
        counts = torch.bincount(self.reason.cpu(), minlength=len(REASONS)).float()
        w = (counts.sum() / counts.clamp(min=1)).sqrt()
        return w[self.reason.cpu()]


# ---------------------------------------------------------------- model
def time_embed(t, dim=64):
    f = torch.exp(torch.linspace(0, np.log(1000.0), dim // 2, device=t.device))
    x = t[:, None] * f[None]
    return torch.cat([x.sin(), x.cos()], 1)


class DriveNet(nn.Module):
    def __init__(self, text_dim=512, cmd_texts=None, ctx=512, img_hw=(128, 224)):
        super().__init__()
        import torchvision
        r = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        self.trunk = nn.Sequential(*list(r.children())[:-2])              # (B, 512, H/32, W/32)
        fh, fw = img_hw[0] // 32, img_hw[1] // 32
        self.squeeze = nn.Sequential(nn.Conv2d(1024, 64, 1), nn.GELU())    # both frames, stacked on channels
        self.img_proj = nn.Sequential(nn.Flatten(), nn.Linear(64 * fh * fw, 512), nn.GELU())
        self.ego = nn.Sequential(nn.Linear(2, 64), nn.GELU())
        self.text = nn.Linear(text_dim, 128)
        self.fuse = nn.Sequential(nn.Linear(512 + 64 + 128, ctx), nn.GELU(), nn.Linear(ctx, ctx), nn.GELU())
        self.vel = nn.Sequential(nn.Linear(H * 2 + 64 + ctx, 1024), nn.GELU(), nn.Linear(1024, 1024), nn.GELU(),
                                 nn.Linear(1024, H * 2))
        self.why = nn.Linear(ctx, len(REASONS))
        self.register_buffer("cmd_emb", torch.zeros(4, 4, text_dim))     # 4 commands × 4 paraphrases (CLIP, frozen)

    def set_commands(self, embedder):
        with torch.no_grad():
            for c, key in enumerate("LSRF"):
                self.cmd_emb[c] = embedder(PARAPHRASES[key]).to(self.cmd_emb.device)

    def context(self, img, img_prev, speed, kappa, cmd, augment=False, paraphrase=None):
        x = torch.cat([img, img_prev], 0).permute(0, 3, 1, 2).float() / 255.0
        if augment:
            x = _augment(x)
        x = (x - MEAN.to(x.device)) / STD.to(x.device)
        f = self.trunk(x)
        B = img.shape[0]
        f = self.img_proj(self.squeeze(torch.cat([f[:B], f[B:]], 1)))
        e = self.ego(torch.stack([speed / 10.0, kappa / K_SCALE], 1))
        p = paraphrase if paraphrase is not None else torch.zeros_like(cmd)
        t = self.text(self.cmd_emb[cmd, p])
        return self.fuse(torch.cat([f, e, t], 1))

    def velocity(self, x, t, c):
        return self.vel(torch.cat([x.flatten(1), time_embed(t), c], 1)).view(-1, H, 2)

    def loss(self, b, augment=True):
        c = self.context(b["img"], b["prev"], b["speed"], b["kappa"], b["cmd"], augment=augment, paraphrase=b.get("para"))
        x1 = b["plan"]
        x0 = torch.randn_like(x1)
        t = torch.rand(len(x1), device=x1.device)
        xt = (1 - t)[:, None, None] * x0 + t[:, None, None] * x1
        flow = F.mse_loss(self.velocity(xt, t, c), x1 - x0)
        why = F.cross_entropy(self.why(c), b["reason"])
        return flow + 0.2 * why, dict(flow=flow.item(), why=why.item())

    @torch.no_grad()
    def sample(self, c, n=6, nfe=10, generator=None):
        """n plans per context: (B, n, H, 2) in physical units (a [m/s²], κ [1/m]); plus reason probabilities."""
        B = c.shape[0]
        cc = c.repeat_interleave(n, 0)
        x = torch.randn(B * n, H, 2, device=c.device, generator=generator)
        for i in range(nfe):
            t = torch.full((B * n,), i / nfe, device=c.device)
            x = x + self.velocity(x, t, cc) / nfe
        x = x.view(B, n, H, 2) * torch.tensor([A_SCALE, K_SCALE], device=c.device)
        return x, F.softmax(self.why(c), -1)


def _augment(x):
    """Random shift (±6 px) and brightness/contrast jitter, per image."""
    B, C, Hh, W = x.shape
    pad = 6
    xp = F.pad(x, (pad, pad, pad, pad), mode="replicate")
    ox = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    oy = torch.randint(0, 2 * pad + 1, (B,), device=x.device)
    ar_y = torch.arange(Hh, device=x.device)[None, :] + oy[:, None]
    ar_x = torch.arange(W, device=x.device)[None, :] + ox[:, None]
    x = xp[torch.arange(B, device=x.device)[:, None, None, None], torch.arange(C, device=x.device)[None, :, None, None],
           ar_y[:, None, :, None], ar_x[:, None, None, :]]
    g = 1 + 0.25 * (torch.rand(B, 1, 1, 1, device=x.device) - 0.5)
    o = 0.08 * (torch.rand(B, 1, 1, 1, device=x.device) - 0.5)
    return ((x - 0.5) * g + 0.5 + o).clamp(0, 1)


def batch(data, ii):
    return dict(img=data.img[data.idx[ii]], prev=data.img[data.prev[ii]], speed=data.speed[ii], kappa=data.kappa[ii],
                cmd=data.cmd[ii], plan=data.plan[ii], reason=data.reason[ii],
                para=torch.randint(0, 4, (len(ii),), device=data.speed.device))


def train(net, data, steps=20000, bs=128, lr=3e-4, device="cuda", log_every=1000, val=None):
    """AdamW + cosine schedule, mixed precision, reason-balanced sampling. Data must already be on `device`."""
    net.to(device).train()
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.05)
    w = data.weights().to(device)
    log, t0 = [], time.time()
    use_amp = device == "cuda"
    for step in range(1, steps + 1):
        ii = torch.multinomial(w, bs, replacement=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
            loss, parts = net.loss(batch(data, ii))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step(); sched.step()
        if step % log_every == 0 or step == steps:
            row = dict(step=step, loss=loss.item(), **parts, seconds=time.time() - t0)
            if val is not None:
                row["val_flow"] = val_loss(net, val)
                net.train()
            log.append(row)
            print("  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in row.items()), flush=True)
    net.eval()
    return log


@torch.no_grad()
def val_loss(net, data, n=2048, seed=0):
    net.eval()
    g = torch.Generator(device=data.speed.device).manual_seed(seed)
    ii = torch.randint(0, len(data), (n,), device=data.speed.device, generator=g)
    tot = 0.0
    for k in range(0, n, 256):
        b = batch(data, ii[k:k + 256])
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=data.speed.is_cuda):
            tot += net.loss(b, augment=False)[1]["flow"] * len(ii[k:k + 256])
    return tot / n


# ---------------------------------------------------------------- closed loop
def plan_path(xy, yaw, v, plan, dt=DT):
    """Integrate an (a, κ) plan with the unicycle model: (H+1, 2) positions. This is how Alpamayo's action expert
    turns acceleration and curvature into the trajectory it outputs."""
    pts, x, y, th, vv = [xy.copy()], xy[0], xy[1], yaw, v
    for a, k in plan:
        vv = max(0.0, vv + a * dt)
        th += vv * k * dt
        x += vv * np.cos(th) * dt; y += vv * np.sin(th) * dt
        pts.append(np.array([x, y]))
    return np.array(pts)


class DrivePolicy:
    """Runs DriveNet at 10 Hz: samples n plans, executes the first step of one of them (receding horizon), and keeps
    the plans, the chosen one and the reasoning for visualization.
    Plan selection: the sample closest to the previous chosen plan (shifted one step), for temporal consistency."""

    def __init__(self, net, device="cuda", n=6, nfe=10, seed=0):
        self.net, self.device, self.n, self.nfe = net.to(device).eval(), device, n, nfe
        self.gen = torch.Generator(device=device).manual_seed(seed)
        self.reset()

    def reset(self):
        self.frames, self.prev_plan, self.last = [], None, None

    @torch.no_grad()
    def act(self, obs, env):
        img = obs["images"]["front"]
        self.frames.append(img)
        self.frames = self.frames[-(HIST + 1):]
        old = self.frames[0]
        dev = self.device
        t = lambda a: torch.as_tensor(np.asarray(a)[None], device=dev)
        c = self.net.context(t(img), t(old), torch.tensor([float(env.ego["v"])], device=dev),
                             torch.tensor([float(env.ego["kappa"])], device=dev),
                             torch.tensor(["LSRF".index(obs["command"])], device=dev))
        plans, why = self.net.sample(c, n=self.n, nfe=self.nfe, generator=self.gen)
        plans = plans[0].float().cpu().numpy()
        if self.prev_plan is not None:
            ref = np.r_[self.prev_plan[1:], self.prev_plan[-1:]]
            k = int((((plans - ref) / np.array([A_SCALE, K_SCALE])) ** 2).sum((1, 2)).argmin())
        else:
            k = 0
        self.prev_plan = plans[k]
        why = why[0].float().cpu().numpy()
        self.last = dict(plans=plans, chosen=k, why=why, reason=REASONS[int(why.argmax())])
        return np.array([np.clip(plans[k][0, 0], A_MIN, A_MAX), np.clip(plans[k][0, 1], -K_MAX, K_MAX)])


class ExpertDriver:
    """The privileged expert behind the same interface (for evaluation and films)."""

    def reset(self):
        self.last = None

    def act(self, obs, env):
        a, reason = env.expert()
        self.last = dict(plans=None, chosen=0, why=None, reason=reason)
        return a


# ---------------------------------------------------------------- evaluation
SCENARIOS = ["random", "red", "jaywalker", "left", "right"]


def run_episode(env, policy, seed, scenario=None, on_step=None):
    obs = env.reset(seed=seed, scenario=None if scenario == "random" else scenario)
    policy.reset()
    done = False
    while not done:
        a = policy.act(obs, env)
        obs, done, info = env.step(a)
        if on_step is not None:
            on_step(env, policy)
    return info


def _eval_worker(args):
    ckpt, seeds, scenario, device, kw = args
    os.environ.setdefault("MUJOCO_GL", "egl")
    from .city import DriveEnv
    env = DriveEnv(**kw)
    if ckpt == "expert":
        pol = ExpertDriver()
    else:
        net = DriveNet()
        net.load_state_dict(torch.load(ckpt, map_location=device))
        pol = DrivePolicy(net, device=device)
    out = []
    for sd in seeds:
        info = run_episode(env, pol, sd, scenario)
        out.append(dict(seed=int(sd), scenario=scenario, **{k: (v if not isinstance(v, (np.floating, np.integer)) else v.item())
                                                            for k, v in info.items()}))
    return out


def evaluate_driver(ckpt, n_per_scenario=20, scenarios=SCENARIOS, workers=8, device="cuda", seed0=900_000, env_kw=None):
    """Closed-loop evaluation in parallel (spawned workers). ckpt = path to DriveNet weights, or 'expert'.
    Returns a list of per-episode records."""
    import multiprocessing as mp
    jobs = []
    for s_i, sc in enumerate(scenarios):
        seeds = list(range(seed0 + 1000 * s_i, seed0 + 1000 * s_i + n_per_scenario))
        per = max(1, int(np.ceil(len(seeds) / max(1, workers // len(scenarios) or 1))))
        for k in range(0, len(seeds), per):
            jobs.append((ckpt, seeds[k:k + per], sc, device, env_kw or {}))
    with mp.get_context("spawn").Pool(workers) as pool:
        res = [r for part in pool.map(_eval_worker, jobs) for r in part]
    return res


def summarize(records):
    """Per-scenario success rate, incidents, and km per incident (Alpamayo's closed-loop metric)."""
    from .evaluation import wilson
    rows = {}
    for sc in sorted({r["scenario"] for r in records}, key=lambda s: SCENARIOS.index(s) if s in SCENARIOS else 99):
        rs = [r for r in records if r["scenario"] == sc]
        k = sum(r["success"] for r in rs)
        lo, hi = wilson(k, len(rs))
        rows[sc] = dict(n=len(rs), success=k / len(rs), ci=(lo, hi),
                        collision_car=sum(r["collision"] == "car" for r in rs),
                        collision_ped=sum(r["collision"] == "pedestrian" for r in rs),
                        offroad=sum(bool(r["offroad"]) for r in rs), wrong_way=sum(bool(r["offroute"]) and not r["offroad"] for r in rs),
                        red_light=sum(r["red_light"] for r in rs), timeout=sum((not r["success"]) and not r["collision"] and not r["offroad"]
                                                                              and not r["offroute"] for r in rs))
    km = sum(r["distance"] for r in records) / 1000
    inc = sum(bool(r["collision"]) or bool(r["offroad"]) for r in records)
    return rows, km, inc
