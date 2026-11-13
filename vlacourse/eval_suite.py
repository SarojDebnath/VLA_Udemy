# Perturbation-based evaluation suite for TableTop policies (S12).
import mujoco
import numpy as np

from .evaluation import evaluate


class Perturbation:
    def __init__(self, name, apply):
        self.name, self.apply_fn = name, apply


def _cam_shift(dx, dy, dz):
    def f(env):
        cid = env.model.camera("front").id
        env._cam0 = env.model.cam_pos[cid].copy() if not hasattr(env, "_cam0") else env._cam0
        env.model.cam_pos[cid] = env._cam0 + np.array([dx, dy, dz])
    return f


def _light(scale):
    def f(env):
        env._light0 = env.model.light_diffuse.copy() if not hasattr(env, "_light0") else env._light0
        env.model.light_diffuse[:] = np.clip(env._light0 * scale, 0, 1)
    return f


def _table_color(rgba):
    def f(env):
        gid = env.model.geom("table").id
        env._table0 = env.model.geom_rgba[gid].copy() if not hasattr(env, "_table0") else env._table0
        env.model.geom_rgba[gid] = rgba
    return f


def _cube_mass(scale):
    def f(env):
        ids = [env.model.body(f"cube_{c}").id for c in env.cube_colors]
        if not hasattr(env, "_mass0"):
            env._mass0, env._inertia0 = env.model.body_mass[ids].copy(), env.model.body_inertia[ids].copy()
        env.model.body_mass[ids] = env._mass0 * scale
        env.model.body_inertia[ids] = env._inertia0 * scale
        mujoco.mj_setConst(env.model, env.data)
    return f


def _cube_friction(mu):
    def f(env):
        ids = [env.model.geom(f"cube_{c}").id for c in env.cube_colors]
        env._fric0 = env.model.geom_friction[ids].copy() if not hasattr(env, "_fric0") else env._fric0
        env.model.geom_friction[ids, 0] = mu
    return f


def _restore_all(env):
    if hasattr(env, "_cam0"):
        env.model.cam_pos[env.model.camera("front").id] = env._cam0
    if hasattr(env, "_light0"):
        env.model.light_diffuse[:] = env._light0
    if hasattr(env, "_table0"):
        env.model.geom_rgba[env.model.geom("table").id] = env._table0
    if hasattr(env, "_mass0"):
        ids = [env.model.body(f"cube_{c}").id for c in env.cube_colors]
        env.model.body_mass[ids], env.model.body_inertia[ids] = env._mass0, env._inertia0
        mujoco.mj_setConst(env.model, env.data)
    if hasattr(env, "_fric0"):
        env.model.geom_friction[[env.model.geom(f"cube_{c}").id for c in env.cube_colors]] = env._fric0
    mujoco.mj_forward(env.model, env.data)


SUITE = [
    Perturbation("nominal", lambda env: None),
    Perturbation("camera +3 cm", _cam_shift(0.0, 0.03, 0.0)),
    Perturbation("camera +6 cm / lower", _cam_shift(0.03, 0.05, -0.04)),
    Perturbation("dim light (x0.5)", _light(0.5)),
    Perturbation("bright light (x1.6)", _light(1.6)),
    Perturbation("grey table", _table_color([0.5, 0.5, 0.5, 1])),
    Perturbation("slippery cubes (μ=0.5)", _cube_friction(0.5)),
]


def run_suite(env, policy, n_episodes=30, seed=200_000, suite=SUITE):
    """Same initial conditions (seeds) for every perturbation, so differences are paired."""
    out = {}
    for p in suite:
        _restore_all(env)
        out[p.name] = evaluate(env, policy, n_episodes=n_episodes, seed=seed, on_reset=p.apply_fn)
        _restore_all(env)
    return out
