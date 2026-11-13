"""Demonstration recording and loading for TableTop.

On-disk format (one directory per dataset):
    meta.json                env config, action/state statistics, collection settings
    episode_00000.npz        per-episode arrays (compressed):
        <camera>   (T, H, W, 3) uint8      observation images, one key per camera
        state      (T, 8) float32          ee pos (3) + ee quat (4) + gripper width (1)
        qpos       (T, n_arm) float32      arm joint angles
        action     (T, 7) float32          clean expert action (the label)
        executed   (T, 7) float32          action actually sent to the env (differs under DART noise)
        instruction ()  str
        task       ()   str (json)
Observation t is recorded *before* action t is applied.
"""
import json
import os
import time

import numpy as np

from .expert import ScriptedExpert
from .tabletop import TASK_TYPES, TableTop


def _stats(x):
    x = np.concatenate(x, 0).astype(np.float64)
    return {k: v.tolist() for k, v in dict(mean=x.mean(0), std=x.std(0) + 1e-8, min=x.min(0), max=x.max(0),
                                              q01=np.percentile(x, 1, 0), q99=np.percentile(x, 99, 0)).items()}


def collect(out_dir, n_episodes, robot="panda", image_size=(128, 128), cameras=("front", "wrist"),
            task_types=TASK_TYPES, n_cubes=3, n_trays=1, dart_noise=0.0, vary_speed=True, seed=0,
            only_success=True, max_steps=200, verbose=True, demonstrator=None, **env_kw):
    """demonstrator(env, rng, attempt) -> object with act(obs); defaults to the randomized ScriptedExpert."""
    os.makedirs(out_dir, exist_ok=True)
    env = TableTop(n_cubes=n_cubes, n_trays=n_trays, image_size=image_size, cameras=cameras, seed=seed,
                   max_steps=max_steps, task_types=task_types, robot=robot, **env_kw)
    rng = np.random.default_rng(seed)
    actions, states, lengths, attempts, saved, t0 = [], [], [], 0, 0, time.time()
    while saved < n_episodes:
        obs = env.reset(seed=seed * 100003 + attempts)
        attempts += 1
        if demonstrator is not None:
            expert = demonstrator(env, rng, attempts)
        else:
            expert = ScriptedExpert(env, seed=attempts, speed=rng.uniform(0.6, 1.0) if vary_speed else 1.0,
                                    hover=rng.uniform(0.08, 0.14) if vary_speed else 0.12)
        rec = {c: [] for c in cameras}
        rec.update(state=[], qpos=[], action=[], executed=[])
        done = False
        while not done:
            a = expert.act(obs)
            a_exec = a.copy()
            if dart_noise > 0:
                a_exec[:3] += rng.normal(0, dart_noise, 3)
            for c in cameras:
                rec[c].append(obs["images"][c])
            rec["state"].append(obs["state"]); rec["qpos"].append(obs["qpos"])
            rec["action"].append(a); rec["executed"].append(a_exec)
            obs, _, done, info = env.step(a_exec)
        if only_success and not info["success"]:
            continue
        arrays = {k: np.asarray(v) for k, v in rec.items()}
        arrays = {k: (v.astype(np.uint8) if k in cameras else v.astype(np.float32)) for k, v in arrays.items()}
        np.savez_compressed(os.path.join(out_dir, f"episode_{saved:05d}.npz"), **arrays,
                            instruction=np.array(env.task["instruction"]), task=np.array(json.dumps(env.task)),
                            success=np.array(bool(info["success"])))
        actions.append(arrays["action"]); states.append(arrays["state"]); lengths.append(len(arrays["action"]))
        saved += 1
        if verbose and (saved % 25 == 0 or saved == n_episodes):
            print(f"  saved {saved}/{n_episodes}  (attempts {attempts}, {time.time() - t0:.0f}s)")
    meta = dict(robot=robot, image_size=list(image_size), cameras=list(cameras), control_hz=env.control_hz,
                task_types=list(task_types), n_cubes=n_cubes, n_trays=n_trays, dart_noise=dart_noise,
                n_episodes=saved, attempts=attempts, n_frames=int(sum(lengths)), episode_lengths=lengths,
                action_stats=_stats(actions), state_stats=_stats(states), seed=seed,
                action_names=["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"], env_kw={k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in env_kw.items()})
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    return meta


def _collect_part(args):
    out_dir, n, seed, kw = args
    if os.name != "nt":
        os.environ.setdefault("MUJOCO_GL", "egl")
    return collect(out_dir, n, seed=seed, verbose=False, **kw)


def collect_parallel(out_dir, n_episodes, workers=None, seed=0, **kw):
    """collect() split over CPU cores (each worker its own seeds), then merged into one dataset directory.
    A 32-core machine records ~1000 TableTop episodes in about 90 s."""
    import glob
    import shutil
    from concurrent.futures import ProcessPoolExecutor
    workers = workers or max(1, min(os.cpu_count() - 1, 20))
    per = [n_episodes // workers + (i < n_episodes % workers) for i in range(workers)]
    parts = [os.path.join(out_dir, f"_part_{i:02d}") for i in range(workers)]
    t0 = time.time()
    # spawn, not fork: a forked worker inherits the parent's GPU/EGL rendering context, which can crash the driver
    with ProcessPoolExecutor(workers, mp_context=__import__('multiprocessing').get_context('spawn')) as ex:
        metas = list(ex.map(_collect_part, [(p, n, seed * 1000 + 17 * i + 1, kw) for i, (p, n) in enumerate(zip(parts, per)) if n > 0]))
    k, actions, states, lengths = 0, [], [], []
    for p in parts:
        for f in sorted(glob.glob(os.path.join(p, "episode_*.npz"))):
            dst = os.path.join(out_dir, f"episode_{k:05d}.npz"); shutil.move(f, dst); k += 1
            ep = load_episode(dst); actions.append(ep["action"]); states.append(ep["state"]); lengths.append(len(ep["action"]))
        shutil.rmtree(p, ignore_errors=True)
    meta = dict(metas[0]); meta.update(n_episodes=k, attempts=sum(m["attempts"] for m in metas), n_frames=int(sum(lengths)),
                                       episode_lengths=lengths, action_stats=_stats(actions), state_stats=_stats(states), seed=seed)
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print(f"collected {k} episodes ({sum(lengths)} frames) with {workers} workers in {time.time() - t0:.0f}s")
    return meta


def scene_dataset(scene, n_episodes=500, root=None, **kw):
    """Path of the standard dataset for a TableTop scene ('single', 'colors', 'full'); recorded once if missing.
    root defaults to the course's data/ folder (next to the section folders)."""
    root = root or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    path = os.path.join(root, f"tabletop_{scene}_v1")
    if not os.path.exists(os.path.join(path, "meta.json")):
        print(f"recording the '{scene}' dataset ({n_episodes} demos), once ...")
        collect_parallel(path, n_episodes, scene=scene, **kw)
    return path


def list_episodes(data_dir):
    return sorted(os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.startswith("episode_") and f.endswith(".npz"))


def load_episode(path):
    with np.load(path, allow_pickle=False) as z:
        ep = {k: z[k] for k in z.files}
    ep["instruction"] = str(ep["instruction"])
    ep["task"] = json.loads(str(ep["task"]))
    return ep


def load_meta(data_dir):
    return json.load(open(os.path.join(data_dir, "meta.json")))
