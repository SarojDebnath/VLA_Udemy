"""S16 notebook 150: NVIDIA Alpamayo 2 Edge in our MuJoCo city (zero-shot), open and closed loop.

Runs in the 'alpamayo' environment (NVIDIA's pinned PyTorch 2.8 / Transformers 5.17 + MuJoCo + vlacourse).

    AlpamayoInputs   six cameras on our ego car (Alpamayo's rig: cross-left/right 120°, front-wide 120°, rear-left/
                     right 70°, front-tele 30°), a 2-second frame history at Alpamayo's sampling times, the last 20
                     ego poses in the ego frame, and a navigation instruction ("Turn left in 30m"): the same input
                     dictionary NVIDIA's PhysicalAI-AV loader builds from real driving logs
    AlpamayoDriver   closed loop: every second, ask Alpamayo for 6 trajectories + reasoning, follow candidate 1 with
                     pure pursuit and its speed profile. The simulation waits while the model thinks (see latency).
"""
import time

import numpy as np
import torch

from .city import AV_RIG, DT, K_MAX, A_MAX, A_MIN

CAM_IDS = [0, 1, 2, 3, 5, 6]
CAM_NAMES = {c: rig[0] for c, rig in zip(CAM_IDS, AV_RIG)}
FRONT_OFFSETS = [-2.0, -1.0, -0.5, -0.1, 0.0]
OTHER_OFFSETS = [-1.0, -0.1, 0.0]
CAM_LABEL = {0: "cross left 120°", 1: "front wide 120°", 2: "cross right 120°", 3: "rear left 70°", 5: "rear right 70°",
             6: "front tele 30°"}


def nav_text(env):
    """Alpamayo's navigation format ('Turn right in 30m'), from our route."""
    J = env.route.next_junction(env.ego["s"], margin=3.0)
    if J is None or J["turn"] == "S":
        return "Continue straight"
    d = max(5, int(round((J["s_in"] - env.ego["s"]) / 5.0) * 5))
    return f"Turn {'left' if J['turn'] == 'L' else 'right'} in {d}m"


def _rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class AlpamayoInputs:
    """Keeps a 2.1-second buffer of the six camera images and the ego pose, and builds Alpamayo's input dict."""

    def __init__(self, env, size=(360, 640)):
        import mujoco
        self.env = env
        self.r = mujoco.Renderer(env.model, *size)
        self.buf = []

    def reset(self):
        self.buf = []

    def record(self):
        e = self.env.ego
        imgs = {}
        for c in CAM_IDS:
            self.r.update_scene(self.env.data, CAM_NAMES[c])
            imgs[c] = self.r.render().copy()
        self.buf.append((self.env.t, imgs, (e["xy"].copy(), float(e["yaw"]))))
        self.buf = self.buf[-24:]

    def _at(self, t):
        ts = np.array([b[0] for b in self.buf])
        ok = np.where(ts <= t + 1e-6)[0]
        return self.buf[ok[-1]] if len(ok) else self.buf[0]

    def ready(self):
        return len(self.buf) >= 21

    def sample(self, nav):
        from alpamayo2_edge.sample import CAMERA_PIXEL_BOUNDS, validate_sample
        t0 = self.buf[-1][0]
        frames, cams, stamps = [], [], []
        for c in CAM_IDS:
            for o in (FRONT_OFFSETS if c in (1, 6) else OTHER_OFFSETS):
                t, imgs, _ = self._at(t0 + o)
                frames.append(imgs[c]); cams.append(c); stamps.append(int(round(t * 1e6)))
        p0, y0 = self.buf[-1][2]
        R0 = _rot(y0)
        xyz, rot = [], []
        for k in range(19, -1, -1):
            _, _, (p, y) = self._at(t0 - 0.1 * k)
            xyz.append(R0.T @ np.r_[p - p0, 0.0]); rot.append(R0.T @ _rot(y))
        data = {
            "image_frames": torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).unsqueeze(1).contiguous(),
            "camera_indices": torch.tensor(cams, dtype=torch.int64),
            "absolute_timestamps": torch.tensor(stamps, dtype=torch.int64),
            "ego_history_xyz": torch.tensor(np.array(xyz), dtype=torch.float32).unsqueeze(0),
            "ego_history_rot": torch.tensor(np.array(rot), dtype=torch.float32).unsqueeze(0),
            "nav_text": [nav], "camera_pixel_bounds": dict(CAMERA_PIXEL_BOUNDS), "min_pixels": 163840, "max_pixels": 196608,
            "input_profile": "camera_subsampled", "history_num_steps": 20, "clip_id": "mujoco-city", "t0_us": int(round(t0 * 1e6)),
            "provenance": {"dataset": "vlacourse MuJoCo city (simulated)"},
        }
        validate_sample(data)
        return data


SETTINGS = dict(seed=43, top_p=0.98, top_k=None, temperature=0.6, num_traj_sets=6, num_traj_samples=1,
                max_generation_length=256, diffusion_kwargs={"inference_step": 10})


def run_alpamayo(model, data, settings=SETTINGS, device="cuda"):
    """One Alpamayo call: (n, 64, 3) trajectories in the ego frame at t0, the n reasoning texts, seconds taken."""
    from alpamayo2_edge.helper import prepare_model_inputs, to_device
    from alpamayo2_edge.inference import predict
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    inputs = to_device(prepare_model_inputs(data, model.config, model.tokenizer), device=device)
    xyz, _, extra = predict(model, inputs, settings, device)
    torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    traj = xyz[0].reshape(-1, 64, 3).float().cpu().numpy()
    cot = [str(c) for c in np.asarray(extra["cot"]).reshape(-1)]
    return traj, cot, sec


class AlpamayoDriver:
    """Alpamayo in the loop. The expert drives the first 2 s (to fill the camera history); then Alpamayo replans every
    `replan` steps (6 candidates + reasoning).

    shadow=None   closed loop: the car follows candidate `pick` (pure pursuit + the trajectory's speed profile)
    shadow=policy shadow mode: `policy` drives (e.g. ExpertDriver()), Alpamayo only plans and explains; every plan
                  is later scored against what the car really did (shadow_metrics)."""

    def __init__(self, model, inputs, replan=10, pick=0, settings=SETTINGS, shadow=None):
        self.model, self.inp, self.replan, self.pick, self.settings, self.shadow = model, inputs, replan, pick, settings, shadow
        self.reset()

    def reset(self):
        self.inp.reset()
        self.step, self.world, self.since, self.log, self.last, self.track = 0, None, 0, [], None, []
        if self.shadow is not None:
            self.shadow.reset()

    def act(self, obs, env):
        self.inp.record()
        e = env.ego
        self.track.append((env.t, e["xy"].copy()))
        if not self.inp.ready():
            a = self.shadow.act(obs, env) if self.shadow is not None else env.expert()[0]
            self.last = dict(plans=None, paths=None, chosen=0, why=None, reason=None, text="warm-up: 2 s of camera history "
                             "before Alpamayo's first plan", mode="warmup")
            self.step += 1
            return a
        if self.world is None or self.since >= self.replan:
            nav = nav_text(env)
            traj, cot, sec = run_alpamayo(self.model, self.inp.sample(nav), self.settings)
            R0 = _rot(e["yaw"])[:2, :2]
            self.world = [np.vstack([e["xy"], (R0 @ t[:, :2].T).T + e["xy"]]) for t in traj]
            self.since, self.v_plan = 0, [np.r_[np.linalg.norm(np.diff(w, axis=0), axis=1) / DT] for w in self.world]
            J = env.route.next_junction(e["s"])
            light = env.lights.state(J["node"], J["d_in"], env.t) if J is not None and J["s_stop"] - e["s"] < 45 else None
            self.log.append(dict(t=env.t, step=len(self.track) - 1, nav=nav, cot=cot, seconds=sec, world=self.world, light=light,
                                 speed=e["v"]))
            self.last = dict(plans=None, paths=self.world, chosen=self.pick, why=None, reason=None, text=cot[self.pick],
                             nav=nav, latency=sec, mode="shadow" if self.shadow is not None else "alpamayo")
        if self.shadow is not None:                     # shadow mode: someone else drives
            self.since += 1; self.step += 1
            info = self.last
            a = self.shadow.act(obs, env)
            self.last = info
            return a
        path, v = self.world[self.pick], self.v_plan[self.pick]
        # steering: pure pursuit on the chosen trajectory; speed: the trajectory's speed 0.3 s ahead
        i = int(np.argmin(np.linalg.norm(path - e["xy"], axis=1)))
        ld = float(np.clip(0.6 * e["v"] + 3.0, 3.5, 9.0))
        seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path[i:], axis=0), axis=1))]
        j = i + int(np.searchsorted(seg, ld))
        tgt = path[min(j, len(path) - 1)]
        d = tgt - e["xy"]
        alpha = (np.arctan2(d[1], d[0]) - e["yaw"] + np.pi) % (2 * np.pi) - np.pi
        k = float(np.clip(2 * np.sin(alpha) / max(np.linalg.norm(d), 1.0), -K_MAX, K_MAX)) if np.linalg.norm(d) > 0.5 else 0.0
        v_t = float(v[min(self.since + 3, len(v) - 1)])
        a = float(np.clip(1.5 * (v_t - e["v"]), A_MIN, A_MAX))
        self.since += 1; self.step += 1
        return np.array([a, k])


def shadow_metrics(log, track, horizon=3.0):
    """Score every Alpamayo plan against what the car really did next (only plans with `horizon` seconds of future).
    Returns per-plan dicts: ADE of candidate 1 and minADE over the 6 candidates (NVIDIA's open-loop metric), and
    whether candidate 1 and the real car agree on stop (< 2 m in `horizon` s) vs go."""
    pos = np.array([p for _, p in track])
    n = int(round(horizon / DT))
    out = []
    for L in log:
        k0 = L["step"]
        if k0 + n >= len(pos):
            continue
        real = pos[k0 + 1:k0 + n + 1]
        ade = [float(np.linalg.norm(w[1:n + 1] - real, axis=1).mean()) for w in L["world"]]
        moved_real = np.linalg.norm(real[-1] - pos[k0]) > 2.0
        moved_plan = np.linalg.norm(L["world"][0][n] - L["world"][0][0]) > 2.0
        out.append(dict(ade1=ade[0], minade=min(ade), stop_go_agree=bool(moved_real == moved_plan), real_moves=bool(moved_real)))
    return out
