"""PyTorch datasets over TableTop episodes (vlacourse.data format)."""
import json
import os

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import list_episodes, load_episode, load_meta

COLORS = ["red", "green", "blue", "yellow", "purple", "orange"]
TASKS = ["pick", "place", "stack"]


def task_vector(task):
    """One-hot task encoding: type (3) + object color (6) + target color (6). Language replaces this in S09."""
    v = np.zeros(15, np.float32)
    v[TASKS.index(task["type"])] = 1
    v[3 + COLORS.index(task["obj"])] = 1
    if "target" in task:
        v[9 + COLORS.index(task["target"])] = 1
    return v


class Normalizer:
    def __init__(self, stats, mode="meanstd"):
        self.mode = mode
        if mode == "meanstd":
            self.a, self.b = np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)
        else:
            lo, hi = np.asarray(stats["q01"], np.float32), np.asarray(stats["q99"], np.float32)
            self.a, self.b = (lo + hi) / 2, np.maximum((hi - lo) / 2, 1e-6)
        # Constant dimensions (e.g. roll/pitch that the expert never uses) keep a tiny scale, so the network's
        # small prediction noise on them is crushed when un-normalizing instead of being passed through as real motion.
        self.b = np.maximum(self.b, 1e-6).astype(np.float32)

    def norm(self, x):
        return (x - self.a) / self.b

    def unnorm(self, x):
        return x * self.b + self.a


# ---------------------------------------------------------------- action representations
POSE_DIMS = np.array([1, 1, 1, 0, 0, 1, 0], np.float32)       # x, y, z, yaw are poses; roll/pitch unused; grip absolute


def commanded_targets(ep):
    """Rebuild the COMMANDED target pose after each step, [x, y, z, 0, 0, yaw, grip], from the executed deltas.
    (Not the achieved gripper pose: the arm lags its target while moving, so achieved poses make every step too short.)"""
    from .tabletop import MAX_DPOS, MAX_DROT, TARGET_LO, TARGET_HI, yaw_of
    s, ex = ep["state"], (ep["executed"] if "executed" in ep else ep["action"])
    p, y = s[0, :3].astype(np.float64).copy(), yaw_of(s[0, 3:7])
    out = np.zeros((len(ex), 7), np.float32)
    for t in range(len(ex)):
        p = np.clip(p + np.clip(ex[t, :3], -MAX_DPOS, MAX_DPOS), TARGET_LO, TARGET_HI)
        y += float(np.clip(ex[t, 5], -MAX_DROT, MAX_DROT))
        out[t, :3], out[t, 5], out[t, 6] = p, y, ep["action"][t, 6]
    return out


def ref_pose(state):
    """[x, y, z, 0, 0, yaw, 0] of the gripper in state(s) (..., 8): the origin of relative action chunks."""
    from .tabletop import yaw_of
    st = np.asarray(state)
    flat = st.reshape(-1, st.shape[-1])
    r = np.zeros((len(flat), 7), np.float32)
    r[:, :3] = flat[:, :3]
    r[:, 5] = [yaw_of(q) for q in flat[:, 3:7]]
    return r.reshape(st.shape[:-1] + (7,))


class TableTopDataset(Dataset):
    """Samples (observation at t, action chunk for t .. t+horizon-1).

    action_mode
      "rel"   (default) each chunk element is the commanded target pose RELATIVE to the gripper pose at t
              (Diffusion Policy / UMI style). The label says "the cube is 7 cm to my left", so the network has to look.
      "delta" the raw 7-D per-step EE deltas the expert sent (velocity-like; the S06 baseline).
    Policies call ds.to_absolute(chunk, obs) and ds.to_command(target, env) to turn predictions into env actions.

    obs keys: camera images (uint8 HxWx3 -> float CHW in [0,1]), 'state' (normalized), 'task' (one-hot), 'pad_mask'.
    """

    def __init__(self, data_dir, episodes=None, horizon=1, cameras=("front", "wrist"), use_images=True, privileged=False,
                 action_mode="rel"):
        self.meta = load_meta(data_dir)
        paths = list_episodes(data_dir)
        if episodes is not None:
            paths = [paths[i] for i in episodes]
        self.eps = [load_episode(p) for p in paths]
        self.horizon, self.cameras, self.use_images, self.privileged = horizon, cameras, use_images, privileged
        self.action_mode = action_mode
        self.state_norm = Normalizer(self.meta["state_stats"])
        if action_mode == "delta":
            self.act_norm = Normalizer(self.meta["action_stats"])
        else:
            for ep in self.eps:
                ep["target"] = commanded_targets(ep)
            C = np.concatenate([self.episode_chunks(e) for e in range(len(self.eps))]).reshape(-1, 7)
            self.act_norm = Normalizer(dict(mean=C.mean(0).tolist(), std=C.std(0).tolist()))
        self.index = [(e, t) for e, ep in enumerate(self.eps) for t in range(len(ep["action"]))]

    def episode_chunks(self, e):
        """(T, horizon, 7) un-normalized action chunks of episode e in this dataset's action_mode."""
        ep = self.eps[e]
        T = len(ep["action"])
        idx = np.minimum(np.arange(T)[:, None] + np.arange(self.horizon)[None], T - 1)
        if self.action_mode == "delta":
            return ep["action"][idx]
        rel = ep["target"][idx] - ref_pose(ep["state"])[:, None, :] * POSE_DIMS
        rel[..., 5] = (rel[..., 5] + np.pi) % (2 * np.pi) - np.pi          # yaw difference, wrapped
        return rel

    def to_absolute(self, chunk, obs):
        """Un-normalized predicted chunk (H, 7) -> what the policy should command (absolute targets for 'rel')."""
        if self.action_mode == "delta":
            return chunk
        return chunk + ref_pose(obs["state"]) * POSE_DIMS

    def to_command(self, a, env):
        """One element of to_absolute(...) -> the 7-D EE-delta action TableTop.step expects."""
        if self.action_mode == "delta":
            return np.asarray(a, np.float32).copy()
        from .tabletop import yaw_of
        d = np.zeros(7, np.float32)
        d[:3] = a[:3] - env.target_pos
        d[5] = (a[5] - yaw_of(env.target_quat) + np.pi) % (2 * np.pi) - np.pi
        d[6] = a[6]
        return d

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        e, t = self.index[i]
        ep = self.eps[e]
        T = len(ep["action"])
        if "_chunks" not in ep:
            ep["_chunks"] = self.act_norm.norm(self.episode_chunks(e)).astype(np.float32)
        out = {
            "state": torch.from_numpy(self.state_norm.norm(ep["state"][t]).astype(np.float32)),
            "task": torch.from_numpy(task_vector(ep["task"])),
            "action": torch.from_numpy(ep["_chunks"][t]),
            "pad_mask": torch.from_numpy((np.arange(t, t + self.horizon) >= T)),
        }
        if self.horizon == 1:
            out["action"] = out["action"][0]
        if self.use_images:
            for c in self.cameras:
                out[c] = torch.from_numpy(ep[c][t]).permute(2, 0, 1).float() / 255.0
        return out


def images_to_tensor(img, device="cpu"):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(device)
