"""Action-chunk policies for TableTop (S08): chunked regression, ACT (transformer CVAE), Diffusion Policy, flow matching.

All share the same observation encoder (ResNet-18 per camera + state + task one-hot) so that the comparison in
S08 project 84 isolates the action head. Each net exposes .loss(batch) and .sample(batch, nfe) -> (B, H, 7) normalized actions.
"""
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .models import ImageEncoder, random_shift
from .torch_data import task_vector


# ---------------------------------------------------------------- observation encoder
class ObsEncoder(nn.Module):
    def __init__(self, cameras=("front", "wrist"), state_dim=8, task_dim=15, spatial=False):
        super().__init__()
        self.cameras, self.spatial = cameras, spatial
        self.backbone = ImageEncoder(pretrained=True, spatial=spatial)
        self.out_dim = self.backbone.out_dim * len(cameras) + state_dim + task_dim

    def forward(self, batch, augment=False):
        feats = []
        for c in self.cameras:
            x = batch[c]
            if augment:
                x = random_shift(x)
            feats.append(self.backbone(x))
        if self.spatial:
            return feats, batch["state"], batch["task"]
        return torch.cat(feats + [batch["state"], batch["task"]], -1)


def sinusoidal(t, dim):
    half = dim // 2
    f = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * f[None]
    return torch.cat([a.sin(), a.cos()], -1)


# ---------------------------------------------------------------- 1) chunked regression
class ChunkRegression(nn.Module):
    def __init__(self, horizon=16, act_dim=7, cameras=("front", "wrist")):
        super().__init__()
        self.H, self.A = horizon, act_dim
        self.obs = ObsEncoder(cameras)
        self.head = nn.Sequential(nn.Linear(self.obs.out_dim, 512), nn.ReLU(), nn.Linear(512, 512), nn.ReLU(), nn.Linear(512, horizon * act_dim))

    def loss(self, b):
        pred = self.head(self.obs(b, augment=True)).view(-1, self.H, self.A)
        return F.l1_loss(pred, b["action"])

    @torch.no_grad()
    def sample(self, b, nfe=None):
        return self.head(self.obs(b)).view(-1, self.H, self.A)


# ---------------------------------------------------------------- 2) ACT: transformer CVAE
class ACT(nn.Module):
    """Action Chunking with Transformers (Zhao et al. 2023), compact version.
    CVAE encoder: [CLS, state, a_1..a_H] -> z.  Policy: transformer encoder over [z, state, task, image patches]
    + transformer decoder with H learned queries -> action chunk.  Inference uses z = 0 (the prior mean)."""

    def __init__(self, horizon=16, act_dim=7, cameras=("front", "wrist"), d=256, z_dim=32, layers=4, kl_weight=10.0, state_dim=8, task_dim=15):
        super().__init__()
        self.H, self.A, self.z_dim, self.kl_w = horizon, act_dim, z_dim, kl_weight
        self.obs = ObsEncoder(cameras, state_dim=state_dim, task_dim=task_dim, spatial=True)
        self.img_proj = nn.Conv2d(512, d, 1)
        self.img_pos = nn.Parameter(torch.randn(1, 16 * len(cameras), d) * 0.02)
        self.state_proj, self.task_proj, self.z_proj = nn.Linear(state_dim, d), nn.Linear(task_dim, d), nn.Linear(z_dim, d)
        enc = lambda: nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 8, 4 * d, 0.1, batch_first=True, norm_first=True), layers)
        self.cvae_enc, self.policy_enc = enc(), enc()
        self.cls, self.act_proj = nn.Parameter(torch.zeros(1, 1, d)), nn.Linear(act_dim, d)
        self.cvae_pos = nn.Parameter(torch.randn(1, horizon + 2, d) * 0.02)
        self.to_latent = nn.Linear(d, 2 * z_dim)
        self.queries = nn.Parameter(torch.randn(1, horizon, d) * 0.02)
        self.decoder = nn.TransformerDecoder(nn.TransformerDecoderLayer(d, 8, 4 * d, 0.1, batch_first=True, norm_first=True), layers)
        self.out = nn.Linear(d, act_dim)

    def _memory(self, b, z, augment):
        imgs, state, task = self.obs(b, augment)
        tok = [self.img_proj(f).flatten(2).transpose(1, 2) for f in imgs]
        img = torch.cat(tok, 1)
        img = img + self.img_pos[:, : img.shape[1]]
        mem = torch.cat([self.z_proj(z)[:, None], self.state_proj(state)[:, None], self.task_proj(task)[:, None], img], 1)
        return self.policy_enc(mem)

    def _decode(self, mem):
        q = self.queries.expand(mem.shape[0], -1, -1)
        return self.out(self.decoder(q, mem))

    def loss(self, b):
        B = b["action"].shape[0]
        seq = torch.cat([self.cls.expand(B, -1, -1), self.state_proj(b["state"])[:, None], self.act_proj(b["action"])], 1) + self.cvae_pos
        mu, logvar = self.to_latent(self.cvae_enc(seq)[:, 0]).chunk(2, -1)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)
        pred = self._decode(self._memory(b, z, augment=True))
        kl = -0.5 * (1 + logvar - mu ** 2 - logvar.exp()).sum(-1).mean()
        return F.l1_loss(pred, b["action"]) + self.kl_w * kl

    @torch.no_grad()
    def sample(self, b, nfe=None):
        B = b["state"].shape[0]
        return self._decode(self._memory(b, torch.zeros(B, self.z_dim, device=b["state"].device), augment=False))


# ---------------------------------------------------------------- 3/4) Diffusion Policy and flow matching
class ResBlock1D(nn.Module):
    def __init__(self, cin, cout, cond_dim):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv1d(cin, cout, 5, padding=2), nn.GroupNorm(8, cout), nn.Mish())
        self.b2 = nn.Sequential(nn.Conv1d(cout, cout, 5, padding=2), nn.GroupNorm(8, cout), nn.Mish())
        self.film = nn.Linear(cond_dim, 2 * cout)
        self.skip = nn.Conv1d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, cond):
        h = self.b1(x)
        scale, bias = self.film(cond).chunk(2, -1)
        h = h * (1 + scale[..., None]) + bias[..., None]
        return self.b2(h) + self.skip(x)


class ConditionalUnet1D(nn.Module):
    """Temporal U-Net over the action chunk (channels = action dims), FiLM-conditioned on observation + time."""

    def __init__(self, act_dim, cond_dim, c1=128, c2=256):
        super().__init__()
        self.t_mlp = nn.Sequential(nn.Linear(128, 256), nn.Mish(), nn.Linear(256, 128))
        cd = cond_dim + 128
        self.d1 = nn.ModuleList([ResBlock1D(act_dim, c1, cd), ResBlock1D(c1, c1, cd)])
        self.down = nn.Conv1d(c1, c1, 3, stride=2, padding=1)
        self.d2 = nn.ModuleList([ResBlock1D(c1, c2, cd), ResBlock1D(c2, c2, cd)])
        self.mid = ResBlock1D(c2, c2, cd)
        self.u2 = ResBlock1D(2 * c2, c2, cd)
        self.up = nn.ConvTranspose1d(c2, c1, 4, stride=2, padding=1)
        self.u1 = nn.ModuleList([ResBlock1D(2 * c1, c1, cd), ResBlock1D(c1, c1, cd)])
        self.final = nn.Conv1d(c1, act_dim, 1)

    def forward(self, x, t, cond):
        c = torch.cat([cond, self.t_mlp(sinusoidal(t, 128))], -1)
        x = x.transpose(1, 2)
        for blk in self.d1:
            x = blk(x, c)
        s1 = x
        x = self.down(x)
        for blk in self.d2:
            x = blk(x, c)
        s2 = x
        x = self.mid(x, c)
        x = self.u2(torch.cat([x, s2], 1), c)
        x = self.up(x)
        for i, blk in enumerate(self.u1):
            x = blk(torch.cat([x, s1], 1) if i == 0 else x, c)
        return self.final(x).transpose(1, 2)


class DiffusionPolicy(nn.Module):
    def __init__(self, horizon=16, act_dim=7, cameras=("front", "wrist"), T=100):
        super().__init__()
        self.H, self.A, self.T = horizon, act_dim, T
        self.obs = ObsEncoder(cameras)
        self.unet = ConditionalUnet1D(act_dim, self.obs.out_dim)
        s = 0.008
        steps = torch.linspace(0, T, T + 1) / T
        ab = torch.cos((steps + s) / (1 + s) * math.pi / 2) ** 2
        self.register_buffer("abar", (ab / ab[0])[1:].clamp(1e-5, 1.0))

    def loss(self, b):
        a = b["action"]; B = a.shape[0]
        cond = self.obs(b, augment=True)
        t = torch.randint(0, self.T, (B,), device=a.device)
        eps = torch.randn_like(a)
        ab = self.abar[t][:, None, None]
        return F.mse_loss(self.unet(ab.sqrt() * a + (1 - ab).sqrt() * eps, t, cond), eps)

    @torch.no_grad()
    def sample(self, b, nfe=10):
        cond = self.obs(b); B = cond.shape[0]
        x = torch.randn(B, self.H, self.A, device=cond.device)
        ts = torch.linspace(self.T - 1, 0, nfe, device=cond.device).long()
        for i, t in enumerate(ts):
            eps = self.unet(x, t.repeat(B), cond)
            ab = self.abar[t]
            x0 = ((x - (1 - ab).sqrt() * eps) / ab.sqrt()).clamp(-5, 5)
            ab_prev = self.abar[ts[i + 1]] if i + 1 < len(ts) else torch.tensor(1.0, device=x.device)
            x = ab_prev.sqrt() * x0 + (1 - ab_prev).sqrt() * eps
        return x


class FlowPolicy(nn.Module):
    """Same backbone as DiffusionPolicy; trained with (rectified) flow matching, sampled with Euler steps."""

    def __init__(self, horizon=16, act_dim=7, cameras=("front", "wrist")):
        super().__init__()
        self.H, self.A = horizon, act_dim
        self.obs = ObsEncoder(cameras)
        self.unet = ConditionalUnet1D(act_dim, self.obs.out_dim)

    def loss(self, b):
        a1 = b["action"]; B = a1.shape[0]
        cond = self.obs(b, augment=True)
        a0 = torch.randn_like(a1)
        tau = torch.rand(B, device=a1.device)
        at = (1 - tau)[:, None, None] * a0 + tau[:, None, None] * a1
        return F.mse_loss(self.unet(at, tau * 100, cond), a1 - a0)

    @torch.no_grad()
    def sample(self, b, nfe=10):
        cond = self.obs(b); B = cond.shape[0]
        x = torch.randn(B, self.H, self.A, device=cond.device)
        for k in range(nfe):
            tau = torch.full((B,), k / nfe, device=x.device)
            x = x + self.unet(x, tau * 100, cond) / nfe
        return x


# ---------------------------------------------------------------- training and deployment
def gpu_cache(ds, device="cuda"):
    """Put a whole TableTopDataset on the GPU once (images stay uint8): every batch is then a GPU index,
    no per-sample CPU decoding. ~10-20x faster training than a DataLoader with num_workers=0."""
    from .torch_data import TableTopDataset
    from .lang import LangDataset, render_instruction
    if isinstance(ds, LangDataset):              # language: cache the base dataset + a table of instruction embeddings
        out = gpu_cache(ds.base, device)
        if out is None:
            return None
        texts = []
        for ep in ds.base.eps:
            t = ep["task"]
            opts = [render_instruction(t, tpl) for tpl in ds.templates[t["type"]]] if ds.paraphrase else [ep["instruction"]]
            texts.append(opts)
        K = max(len(o) for o in texts)
        table = np.stack([np.stack([np.asarray(ds.emb[o[k % len(o)]], np.float32) for k in range(K)]) for o in texts])   # (E, K, D)
        ep_id = np.concatenate([np.full(len(ep["action"]), e) for e, ep in enumerate(ds.base.eps)])
        out["_lang_table"], out["_ep_id"] = torch.from_numpy(table).to(device), torch.from_numpy(ep_id).to(device)
        return out
    if not (isinstance(ds, TableTopDataset) and type(ds).__getitem__ is TableTopDataset.__getitem__):
        return None
    H, cache = ds.horizon, {k: [] for k in ["state", "task", "action", "pad_mask"] + (list(ds.cameras) if ds.use_images else [])}
    for e, ep in enumerate(ds.eps):
        T = len(ep["action"])
        t = np.arange(T)
        cache["state"].append(ds.state_norm.norm(ep["state"]).astype(np.float32))
        cache["task"].append(np.repeat(task_vector(ep["task"])[None], T, 0))
        cache["action"].append(ds.act_norm.norm(ds.episode_chunks(e)).astype(np.float32))
        cache["pad_mask"].append(t[:, None] + np.arange(H)[None] >= T)
        if ds.use_images:
            for c in ds.cameras:
                cache[c].append(ep[c])
    out = {k: torch.from_numpy(np.concatenate(v)).to(device) for k, v in cache.items()}
    if ds.use_images:
        for c in ds.cameras:
            out[c] = out[c].permute(0, 3, 1, 2).contiguous()          # uint8 (N, 3, H, W)
    if H == 1:
        out["action"] = out["action"][:, 0]
    return out


def _gpu_batches(cache, cameras, bs, device):
    N = len(cache["state"])
    perm = torch.randperm(N, device=device)
    for i in range(0, N - bs + 1, bs):
        j = perm[i:i + bs]
        b = {k: v[j] for k, v in cache.items() if not k.startswith("_")}
        if "_lang_table" in cache:               # a random paraphrase per sample, like LangDataset.__getitem__
            e = cache["_ep_id"][j]
            k = torch.randint(0, cache["_lang_table"].shape[1], (len(j),), device=device)
            b["lang"] = cache["_lang_table"][e, k]
        for c in cameras:
            b[c] = b[c].float().div_(255.0)
        yield b


def train_policy(net, ds, epochs=20, bs=64, lr=1e-4, device="cuda", log_every=None, weight_decay=1e-6):
    net.to(device)
    backbone = [p for n, p in net.named_parameters() if "backbone" in n]
    rest = [p for n, p in net.named_parameters() if "backbone" not in n]
    opt = torch.optim.AdamW([{"params": backbone, "lr": lr}, {"params": rest, "lr": lr * 3}], weight_decay=weight_decay)
    n_steps = epochs * (len(ds) // bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[lr, lr * 3], total_steps=n_steps, pct_start=0.05)
    losses, t0 = [], time.time()
    cache = gpu_cache(ds, device) if str(device).startswith("cuda") else None
    base = getattr(ds, "base", ds)
    cams = list(base.cameras) if cache is not None and base.use_images else []
    for ep in range(epochs):
        batches = _gpu_batches(cache, cams, bs, device) if cache is not None else \
            DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=0)
        for b in batches:
            b = {k: v.to(device) for k, v in b.items()}
            if b["action"].dim() == 2:                # horizon-1 datasets give (B, 7); chunk heads predict (B, H=1, 7).
                b["action"] = b["action"][:, None]    # Without this, (B,1,7) vs (B,7) BROADCASTS to (B,B,7) and the net learns the batch mean.
            loss = net.loss(b)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); sched.step()
            losses.append(loss.item())
        if log_every and (ep + 1) % log_every == 0:
            print(f"  epoch {ep + 1}/{epochs}  loss {np.mean(losses[-50:]):.4f}  ({time.time() - t0:.0f}s)")
    return np.array(losses)


class ChunkPolicy:
    """Deploys a chunk net in TableTop: receding-horizon execution of the first `execute` actions,
    or ACT-style temporal ensembling (query every step, exponentially weighted average of overlapping chunks)."""

    def __init__(self, net, ds, env, device="cuda", execute=8, ensemble=False, k=0.01, nfe=10):
        self.net, self.ds, self.env, self.device = net.eval(), ds, env, device
        self.execute, self.ensemble, self.k, self.nfe = execute, ensemble, k, nfe
        self.reset()

    def reset(self):
        self.queue, self.history, self.t = [], [], 0

    def _batch(self, obs):
        b = {"state": torch.from_numpy(self.ds.state_norm.norm(obs["state"]).astype(np.float32))[None],
             "task": torch.from_numpy(task_vector(self.env.task))[None]}
        for c in self.net.obs.cameras:
            b[c] = torch.from_numpy(np.ascontiguousarray(obs["images"][c])).permute(2, 0, 1).float().div(255)[None]
        return {k: v.to(self.device) for k, v in b.items()}

    def _chunk(self, obs):
        """Predicted chunk in COMMAND space (absolute targets for action_mode='rel', raw deltas for 'delta')."""
        out = self.net.sample(self._batch(obs), nfe=self.nfe)[0].cpu().numpy()
        out = np.stack([self.ds.act_norm.unnorm(a) for a in out])
        return self.ds.to_absolute(out, obs) if hasattr(self.ds, "to_absolute") else out

    def act(self, obs):
        if self.ensemble:
            self.history.append((self.t, self._chunk(obs)))
            preds = [(c[self.t - s], s) for s, c in self.history if self.t - s < len(c)]
            w = np.exp(-self.k * np.arange(len(preds)))
            a = np.sum([p * wi for (p, _), wi in zip(preds, w)], 0) / w.sum()
            self.history = [(s, c) for s, c in self.history if self.t - s < len(c) - 1]
        else:
            if not self.queue:
                self.queue = list(self._chunk(obs)[: self.execute])
            a = self.queue.pop(0)
        self.t += 1
        a = self.ds.to_command(a, self.env) if hasattr(self.ds, "to_command") else a.copy()
        a[6] = float(a[6] > 0.5)
        return a
