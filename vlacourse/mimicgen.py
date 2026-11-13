"""Capstone C6: a MimicGen-style data engine for TableTop (Mandlekar et al. 2023, "MimicGen").

A handful of (simulated-)human demonstrations is turned into hundreds of new ones:
  1. record source demos together with the object poses at the start (record_source),
  2. split each demo into object-centric segments: everything up to the lift happens relative to the CUBE,
     everything after it relative to the TRAY (segment),
  3. for a new layout, move each segment rigidly with its object (SE(2): translation + yaw about the object),
  4. connect the segments with short linear interpolations, replay the commanded poses in simulation,
  5. keep only the replays that succeed (generate). The result is saved in the normal dataset format.
"""
import json
import os
import time

import numpy as np

from .data import _stats, load_episode
from .expert import ScriptedExpert, wrap
from .tabletop import TableTop, yaw_of


class SimulatedHuman:
    """The scripted expert seen through a human (S06 notebook 55): reaction lag, correlated hand tremor, hesitations,
    and a personal approach style (comes in from the left or right of the object)."""

    def __init__(self, env, rng):
        self.env, self.rng = env, rng
        self.expert = ScriptedExpert(env, seed=int(rng.integers(1 << 30)), speed=rng.uniform(0.4, 0.8), hover=rng.uniform(0.06, 0.16), tol=8e-3)
        side = rng.choice([-1, 1]) * rng.uniform(0.03, 0.06)
        p = self.expert.waypoints[0]
        self.expert.waypoints.insert(0, (p[0] + np.array([0, side, 0.05]), p[1], 0.0, 0))
        self.lag, self.noise, self.queue = int(rng.integers(0, 3)), np.zeros(3), []

    def act(self, obs):
        a = self.expert.act(obs)
        self.noise = 0.8 * self.noise + self.rng.normal(0, 0.003, 3)
        a[:3] += self.noise
        if self.rng.random() < 0.05:
            a[:6] = 0
        self.queue.append(a)
        return self.queue.pop(0) if len(self.queue) > self.lag else np.r_[np.zeros(6), a[6]]


def _poses(env):
    t = env.task
    out = dict(cube=np.r_[env.cube_pos(t["obj"])[:2], env.cube_yaw(t["obj"])])
    if t["type"] == "place":
        out["tray"] = env.tray_pos(t["target"])[:2].copy()
    return out


def record_source(env, n, seed=0, max_tries=200, out_dir=None):
    """n successful SimulatedHuman demos. Each keeps: task, start object poses, the COMMANDED target pose sequence
    [x, y, z, yaw, grip] after every step (what the human asked the arm to do) and the cube height (to find the lift).
    With out_dir, the same demos are also saved with images in the normal dataset format (the '10 demos only' baseline)."""
    rng, demos, tries, eps = np.random.default_rng(seed), [], 0, []
    while len(demos) < n and tries < max_tries:
        obs = env.reset(seed=seed * 7919 + tries); tries += 1
        op, poses, traj, cz, done = SimulatedHuman(env, rng), _poses(env), [], [], False
        rec = {c: [] for c in env.cameras}; rec.update(state=[], qpos=[], action=[], executed=[])
        while not done:
            a = op.act(obs)
            for c in env.cameras:
                rec[c].append(obs["images"][c])
            rec["state"].append(obs["state"]); rec["qpos"].append(obs["qpos"]); rec["action"].append(a); rec["executed"].append(a)
            obs, _, done, info = env.step(a)
            traj.append(np.r_[env.target_pos, yaw_of(env.target_quat), env.grip]); cz.append(env.cube_pos(env.task["obj"])[2])
        if info["success"]:
            demos.append(dict(task=dict(env.task), poses=poses, traj=np.array(traj), cube_z=np.array(cz)))
            eps.append((dict(env.task), rec))
    if out_dir:
        _save_dataset(out_dir, eps, env)
    return demos, tries


def _save_dataset(out_dir, eps, env):
    os.makedirs(out_dir, exist_ok=True)
    actions, states, lengths = [], [], []
    for i, (task, rec) in enumerate(eps):
        arrays = {k: (np.asarray(v, np.uint8) if k in env.cameras else np.asarray(v, np.float32)) for k, v in rec.items()}
        np.savez_compressed(os.path.join(out_dir, f"episode_{i:05d}.npz"), **arrays, instruction=np.array(task["instruction"]),
                            task=np.array(json.dumps({k: str(v) for k, v in task.items()})), success=np.array(True))
        actions.append(arrays["action"]); states.append(arrays["state"]); lengths.append(len(arrays["action"]))
    meta = dict(robot="panda", image_size=[env.h, env.w], cameras=list(env.cameras), control_hz=env.control_hz, task_types=list(env.task_types),
                n_episodes=len(eps), n_frames=int(sum(lengths)), episode_lengths=lengths, action_stats=_stats(actions), state_stats=_stats(states),
                action_names=["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"], env_kw={})
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)


def segment(demo, approach_radius=0.10):
    """Object-centric segments: [('cube', poses...), ('tray', poses...)].
    The cube segment starts when the gripper first comes within `approach_radius` (horizontally) of the cube: the
    free-space motion before that isn't about the object and is replaced by an interpolation."""
    tr, c = demo["traj"], demo["poses"]["cube"]
    near = np.where(np.linalg.norm(tr[:, :2] - c[:2], axis=1) < approach_radius)[0]
    start = int(near[0]) if len(near) else 0
    if demo["task"]["type"] == "pick":
        return [("cube", tr[start:])]
    lifted = np.where((tr[:, 4] > 0.5) & (demo["cube_z"] > 0.06))[0]
    cut = int(lifted[0]) if len(lifted) else len(tr)
    return [("cube", tr[start:cut]), ("tray", tr[cut:])]


def _transform(seg, frame, src, dst):
    out = seg.copy()
    if frame == "tray":                                       # trays are square and don't rotate: translation only
        out[:, :2] += dst["tray"] - src["tray"]
        return out
    d = wrap(dst["cube"][2] - src["cube"][2])
    d = (d + np.pi / 4) % (np.pi / 2) - np.pi / 4             # a cube looks the same every 90 degrees: smallest turn
    c, s = np.cos(d), np.sin(d)
    rel = seg[:, :2] - src["cube"][:2]
    out[:, :2] = dst["cube"][:2] + rel @ np.array([[c, s], [-s, c]])
    out[:, 3] = seg[:, 3] + d
    return out


def _go_to(env, pose, grip, steps_max=40, tol=0.004):
    """Drive the commanded target to `pose` [x, y, z, yaw] in straight-line steps (the interpolation between segments)."""
    acts = []
    for _ in range(steps_max):
        dp = pose[:3] - env.target_pos
        dy = wrap(pose[3] - yaw_of(env.target_quat))
        if np.linalg.norm(dp) < tol and abs(dy) < 0.02:
            break
        n = np.linalg.norm(dp)
        dp = dp / n * min(n, env.max_dpos) if n > 0 else dp
        acts.append(np.r_[dp, 0, 0, np.clip(dy, -env.max_drot, env.max_drot), grip].astype(np.float32))
        yield acts[-1]


def replay_plan(env, demo, dst):
    """Generator of actions: interpolate to each transformed segment, then follow it pose by pose."""
    for frame, seg in segment(demo):
        if len(seg) == 0:
            continue
        seg = _transform(seg, frame, demo["poses"], dst)
        yield from _go_to(env, seg[0, :4], float(env.grip))
        for p in seg:
            dp = p[:3] - env.target_pos
            n = np.linalg.norm(dp)
            if n > env.max_dpos:
                dp *= env.max_dpos / n
            dy = np.clip(wrap(p[3] - yaw_of(env.target_quat)), -env.max_drot, env.max_drot)
            yield np.r_[dp, 0, 0, dy, p[4]].astype(np.float32)
    for _ in range(4):                                        # settle (lets a released cube come to rest)
        yield np.r_[0, 0, 0, 0, 0, 0, float(env.grip)].astype(np.float32)


def generate(out_dir, demos, n, env_kw, seed=0, verbose=True):
    """Generate n new successful demos from `demos` (each new layout uses a random source demo of the same task type).
    Returns stats: attempts, successes, yield. Episodes are saved in the vlacourse dataset format."""
    os.makedirs(out_dir, exist_ok=True)
    env = TableTop(**env_kw, seed=seed)
    rng = np.random.default_rng(seed)
    saved, attempts, t0 = 0, 0, time.time()
    while saved < n:
        obs = env.reset(seed=seed * 104729 + attempts); attempts += 1
        same = [d for d in demos if d["task"]["type"] == env.task["type"]]
        if not same:
            continue
        src = same[rng.integers(len(same))]
        dst = _poses(env)
        rec = {c: [] for c in env.cameras}; rec.update(state=[], qpos=[], action=[], executed=[])
        info = {"success": False}
        for a in replay_plan(env, src, dst):
            for c in env.cameras:
                rec[c].append(obs["images"][c])
            rec["state"].append(obs["state"]); rec["qpos"].append(obs["qpos"]); rec["action"].append(a); rec["executed"].append(a)
            obs, _, done, info = env.step(a)
            if done:
                break
        if not info["success"]:
            continue
        arrays = {k: (np.asarray(v, np.uint8) if k in env.cameras else np.asarray(v, np.float32)) for k, v in rec.items()}
        np.savez_compressed(os.path.join(out_dir, f"episode_{saved:05d}.npz"), **arrays, instruction=np.array(env.task["instruction"]),
                            task=np.array(json.dumps({k: str(v) for k, v in env.task.items()})), success=np.array(True))
        saved += 1
        if verbose and saved % 50 == 0:
            print(f"  generated {saved}/{n} ({attempts} attempts, {time.time() - t0:.0f}s)")
    return dict(attempts=attempts, saved=saved)


def _gen_part(args):
    out, demos, n, env_kw, seed = args
    if os.name != "nt":
        os.environ.setdefault("MUJOCO_GL", "egl")
    return generate(out, demos, n, env_kw, seed=seed, verbose=False)


def generate_parallel(out_dir, demos, n, env_kw, workers=None, seed=0):
    import glob
    import shutil
    from concurrent.futures import ProcessPoolExecutor
    workers = workers or max(1, min(os.cpu_count() - 1, 20))
    per = [n // workers + (i < n % workers) for i in range(workers)]
    parts = [os.path.join(out_dir, f"_part_{i:02d}") for i in range(workers)]
    t0 = time.time()
    # spawn, not fork: a forked worker inherits the parent's GPU/EGL rendering context, which can crash the driver
    with ProcessPoolExecutor(workers, mp_context=__import__('multiprocessing').get_context('spawn')) as ex:
        infos = list(ex.map(_gen_part, [(p, demos, k, env_kw, seed * 1000 + 13 * i + 3) for i, (p, k) in enumerate(zip(parts, per)) if k > 0]))
    k, actions, states, lengths = 0, [], [], []
    for p in parts:
        for f in sorted(glob.glob(os.path.join(p, "episode_*.npz"))):
            dst = os.path.join(out_dir, f"episode_{k:05d}.npz"); shutil.move(f, dst); k += 1
            ep = load_episode(dst); actions.append(ep["action"]); states.append(ep["state"]); lengths.append(len(ep["action"]))
        shutil.rmtree(p, ignore_errors=True)
    attempts = sum(i["attempts"] for i in infos)
    meta = dict(robot="panda", image_size=list(env_kw.get("image_size", (128, 128))), cameras=list(env_kw.get("cameras", ("front", "wrist"))),
                control_hz=10, task_types=list(env_kw.get("task_types", ("pick", "place"))), n_episodes=k, attempts=attempts,
                generated_from=len(demos), yield_=k / max(1, attempts), n_frames=int(sum(lengths)), episode_lengths=lengths,
                action_stats=_stats(actions), state_stats=_stats(states), seed=seed,
                action_names=["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"], env_kw={})
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print(f"generated {k} demos from {len(demos)} sources: {attempts} attempts, yield {k / max(1, attempts):.0%}, {time.time() - t0:.0f}s ({workers} workers)")
    return meta
