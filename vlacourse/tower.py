"""Capstone C5: three-cube towers on TableTop (long-horizon, language, task-level feedback).

    collect_towers(out_dir, n)        tower demonstrations, one stored episode per stacking step ("stack the X on the Y")
    tower_height(env, order)          how many levels of the requested tower are correctly built right now
    TowerExecutive(env, policy, ...)  runs a plan step by step, CHECKS the tower after every step and re-plans from what
                                      is actually built (retries a fallen cube, rebuilds a knocked-over level)
"""
import json
import os
import time

import numpy as np

from .data import _stats, load_episode
from .expert import ScriptedExpert
from .tabletop import CUBE_HALF, TableTop


def stack_task(top, bottom):
    return dict(type="stack", obj=str(top), target=str(bottom), instruction=f"stack the {top} cube on the {bottom} cube")


def on_top(env, top, bottom):
    p, q = env.cube_pos(top), env.cube_pos(bottom)
    return bool(np.linalg.norm(p[:2] - q[:2]) < 0.02 and 0.03 < p[2] - q[2] < 0.05)


def tower_height(env, order):
    """Number of correctly stacked levels of `order` (bottom -> top), counted from the bottom cube resting on the table."""
    if env.cube_pos(order[0])[2] > CUBE_HALF + 0.01:
        return 0
    k = 0
    while k + 1 < len(order) and on_top(env, order[k + 1], order[k]):
        k += 1
    return k


def retract(env, home, frames=None, lift=5, back=15):
    """Release, lift straight up (fingers clear the cube), then return to the home pose: every step starts like a demo."""
    obs = None
    for _ in range(lift):
        obs, *_ = env.step(np.r_[0, 0, 0.03, 0, 0, 0, 0.0])
    for _ in range(back):
        obs, *_ = env.step(np.r_[np.clip(home - env.target_pos, -0.05, 0.05), 0, 0, 0, 0.0])
    return obs


def tower_env(**kw):
    return TableTop(n_cubes=3, n_trays=0, task_types=("stack",), **kw)


def collect_towers(out_dir, n_towers, image_size=(128, 128), cameras=("front", "wrist"), seed=0, verbose=True):
    """Scripted 3-cube towers. Each stacking step is saved as its own episode (same format as vlacourse.collect), so the
    level-2 step ('stack the blue cube on the green cube' while green already sits on red) is in the data too."""
    os.makedirs(out_dir, exist_ok=True)
    env = tower_env(image_size=image_size, cameras=cameras, seed=seed)
    rng = np.random.default_rng(seed)
    saved, attempts, towers, t0 = 0, 0, 0, time.time()
    while towers < n_towers:
        obs = env.reset(seed=seed * 100003 + attempts); attempts += 1
        order = [str(c) for c in rng.permutation(env.cube_colors)]
        home, segs = env.target_pos.copy(), []
        for lvl in (1, 2):
            env.task = stack_task(order[lvl], order[lvl - 1])
            ex = ScriptedExpert(env, seed=attempts * 10 + lvl, speed=rng.uniform(0.6, 1.0), hover=rng.uniform(0.08, 0.14))
            rec = {c: [] for c in cameras}; rec.update(state=[], qpos=[], action=[], executed=[])
            k = 0
            while not ex.done and k < 150:
                a = ex.act(obs)
                for c in cameras:
                    rec[c].append(obs["images"][c])
                rec["state"].append(obs["state"]); rec["qpos"].append(obs["qpos"]); rec["action"].append(a); rec["executed"].append(a)
                obs, *_ = env.step(a); k += 1
            segs.append((dict(env.task, level=lvl, order=order), rec, env.success()))
            obs = retract(env, home, lift=0)
        if not all(s[2] for s in segs) or tower_height(env, order) != 2:
            continue                                        # keep complete towers only
        towers += 1
        for task, rec, _ in segs:
            arrays = {k: (np.asarray(v, np.uint8) if k in cameras else np.asarray(v, np.float32)) for k, v in rec.items()}
            np.savez_compressed(os.path.join(out_dir, f"episode_{saved:05d}.npz"), **arrays, instruction=np.array(task["instruction"]),
                                task=np.array(json.dumps(task)), success=np.array(True))
            saved += 1
        if verbose and towers % 25 == 0:
            print(f"  {towers}/{n_towers} towers ({attempts} attempts, {time.time() - t0:.0f}s)")
    return dict(attempts=attempts, towers=towers)


def _part(args):
    out, n, seed, kw = args
    if os.name != "nt":
        os.environ.setdefault("MUJOCO_GL", "egl")
    return collect_towers(out, n, seed=seed, verbose=False, **kw)


def collect_towers_parallel(out_dir, n_towers, workers=None, seed=0, **kw):
    import glob
    import shutil
    from concurrent.futures import ProcessPoolExecutor
    workers = workers or max(1, min(os.cpu_count() - 1, 20))
    per = [n_towers // workers + (i < n_towers % workers) for i in range(workers)]
    parts = [os.path.join(out_dir, f"_part_{i:02d}") for i in range(workers)]
    t0 = time.time()
    # spawn, not fork: a forked worker inherits the parent's GPU/EGL rendering context, which can crash the driver
    with ProcessPoolExecutor(workers, mp_context=__import__('multiprocessing').get_context('spawn')) as ex:
        infos = list(ex.map(_part, [(p, n, seed * 1000 + 31 * i + 7, kw) for i, (p, n) in enumerate(zip(parts, per)) if n > 0]))
    k, actions, states, lengths = 0, [], [], []
    for p in parts:
        for f in sorted(glob.glob(os.path.join(p, "episode_*.npz"))):
            dst = os.path.join(out_dir, f"episode_{k:05d}.npz"); shutil.move(f, dst); k += 1
            ep = load_episode(dst); actions.append(ep["action"]); states.append(ep["state"]); lengths.append(len(ep["action"]))
        shutil.rmtree(p, ignore_errors=True)
    attempts = sum(i["attempts"] for i in infos)
    meta = dict(robot="panda", image_size=list(kw.get("image_size", (128, 128))), cameras=list(kw.get("cameras", ("front", "wrist"))),
                control_hz=10, task_types=["stack"], n_cubes=3, n_trays=0, n_episodes=k, attempts=attempts, towers=n_towers,
                n_frames=int(sum(lengths)), episode_lengths=lengths, action_stats=_stats(actions), state_stats=_stats(states), seed=seed,
                action_names=["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"], env_kw={})
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print(f"{n_towers} towers ({k} stacking episodes, {sum(lengths)} frames) from {attempts} attempts "
          f"(expert tower success {n_towers / attempts:.0%}) with {workers} workers in {time.time() - t0:.0f}s")
    return meta


def tower_dataset(n_towers=400, root=None):
    root = root or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    path = os.path.join(root, "tabletop_tower_v1")
    if not os.path.exists(os.path.join(path, "meta.json")):
        print(f"recording {n_towers} tower demonstrations, once ...")
        collect_towers_parallel(path, n_towers)
    return path


class TowerExecutive:
    """Task-level closed loop. After every step it measures the tower and asks: what is the next cube that belongs on
    top of what is built? A cube that fell is simply attempted again; a knocked-down level is rebuilt.
    replan=False executes the plan once, open loop (the baseline)."""

    def __init__(self, env, policy, shield=None, step_budget=90, max_steps_total=8, replan=True):
        self.env, self.policy, self.shield = env, policy, shield
        self.step_budget, self.max_steps_total, self.replan = step_budget, max_steps_total, replan

    def run(self, order, seed, on_step=None):
        env = self.env
        env.reset(seed=seed); home = env.target_pos.copy()
        log, n = [], 0
        plan = [stack_task(order[i], order[i - 1]) for i in range(1, len(order))]
        while n < self.max_steps_total:
            h = tower_height(env, order)
            if h == len(order) - 1:
                break
            if self.replan:
                st = stack_task(order[h + 1], order[h])          # the next cube on top of what is ACTUALLY built
            else:
                if n >= len(plan):
                    break
                st = plan[n]
            env.task = st; self.policy.reset(); obs = env.observe(); k = 0
            while not (on_top(env, st["obj"], st["target"]) and env.grip < 0.5) and k < self.step_budget:
                a = self.policy.act(obs)
                if self.shield is not None:
                    a = self.shield(env, a)
                obs, *_ = env.step(a); k += 1
                if on_step is not None:
                    on_step(env, st, log)
            obs = retract(env, home)
            log.append(dict(step=st["instruction"], ok=on_top(env, st["obj"], st["target"]), height=tower_height(env, order)))
            n += 1
        return tower_height(env, order) == len(order) - 1, log
