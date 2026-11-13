"""Privileged, object-centric state features for TableTop (used by state-based BC / DAgger in S06)."""
import numpy as np

from .tabletop import CUBE_HALF, yaw_of

TASK_TYPES = ["pick", "place", "stack"]


def privileged_obs(env):
    """[ee state (8) | obj - ee (3) | dest - ee (3) | obj yaw sin/cos(4θ) (2) | task one-hot (3) | grip cmd (1)] = 20-D."""
    t = env.task
    s = env.state()
    ee = s[:3]
    obj = env.cube_pos(t["obj"])
    if t["type"] == "place":
        dest = env.tray_pos(t["target"]) + [0, 0, 0.06]
    elif t["type"] == "stack":
        dest = env.cube_pos(t["target"]) + [0, 0, 2 * CUBE_HALF + 0.003]
    else:
        dest = np.r_[obj[:2], 0.2]
    yaw = env.cube_yaw(t["obj"]) - yaw_of(env.target_quat)
    onehot = np.eye(3)[TASK_TYPES.index(t["type"])]
    return np.r_[s, obj - ee, dest - ee, np.sin(4 * yaw), np.cos(4 * yaw), onehot, env.grip].astype(np.float32)


PRIV_DIM = 20
