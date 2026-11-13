"""Scripted expert for TableTop. Uses privileged state (object poses) and outputs the same 7-D
end-effector-delta action a learned policy would, so its rollouts are directly usable as demonstrations."""
import numpy as np

from .tabletop import CUBE_HALF, yaw_of


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class ScriptedExpert:
    def __init__(self, env, noise=0.0, speed=1.0, seed=0, hover=0.12, tol=1e-3, track=False):
        """tol: how close the commanded target must get to a waypoint before advancing (raise it when someone else
        perturbs or drives the robot: DART noise, DAgger shadow labelling). track: re-plan waypoints from the current
        object poses every step (a proper DAgger oracle)."""
        self.env, self.noise, self.speed, self.hover, self.tol, self.track = env, noise, speed, hover, tol, track
        self.rng = np.random.default_rng(seed)
        self.reset()

    def grasp_yaw(self, color):
        cur = yaw_of(self.env.target_quat)
        base = self.env.cube_yaw(color)
        cands = [base + k * np.pi / 2 for k in range(-4, 5)]
        return min(cands, key=lambda y: abs(wrap(y - cur)))

    def _plan(self):
        env, t, h, yaw = self.env, self.env.task, self.hover, self.yaw
        p = env.cube_pos(t["obj"])
        wps = [
            (np.r_[p[:2] + self.jitter, p[2] + h], yaw, 0.0, 0),
            (np.r_[p[:2], p[2] - 0.005], yaw, 0.0, 0),
            (np.r_[p[:2], p[2] - 0.005], yaw, 1.0, 6),
            (np.r_[p[:2], 0.2], yaw, 1.0, 0),
        ]
        if t["type"] == "place":
            g = env.tray_pos(t["target"])
            wps += [(np.r_[g[:2], 0.2], yaw, 1.0, 0),
                    (np.r_[g[:2], 0.06], yaw, 1.0, 0),
                    (np.r_[g[:2], 0.06], yaw, 0.0, 4),
                    (np.r_[g[:2], 0.2], yaw, 0.0, 0)]
        elif t["type"] == "stack":
            q = env.cube_pos(t["target"])
            y2 = yaw_of(env.data.body(f"cube_{t['target']}").xquat)
            y2 = min([y2 + k * np.pi / 2 for k in range(-4, 5)], key=lambda y: abs(wrap(y - yaw)))
            top = q[2] + 2 * CUBE_HALF + 0.003
            wps += [(np.r_[q[:2], top + h], y2, 1.0, 0),
                    (np.r_[q[:2], top], y2, 1.0, 0),
                    (np.r_[q[:2], top], y2, 0.0, 4),
                    (np.r_[q[:2], top + h], y2, 0.0, 0)]
        return wps

    def reset(self):
        self.yaw = self.grasp_yaw(self.env.task["obj"])
        self.jitter = self.rng.normal(0, 0.01, 2) * (self.noise > 0)
        self.waypoints = self._plan()
        self.i, self.wait = 0, 0

    @property
    def done(self):
        return self.i >= len(self.waypoints)

    def act(self, obs=None):
        env = self.env
        if self.track and not self.done:
            self.waypoints = self._plan()
        if self.done:
            pos, yaw, grip = env.target_pos, yaw_of(env.target_quat), self.waypoints[-1][2]
        else:
            pos, yaw, grip, hold = self.waypoints[self.i]
        dpos = np.asarray(pos) - env.target_pos
        dyaw = wrap(yaw - yaw_of(env.target_quat))
        step = env.max_dpos * self.speed
        n = np.linalg.norm(dpos)
        if n > step:
            dpos = dpos / n * step
        dyaw = float(np.clip(dyaw, -env.max_drot, env.max_drot))
        if not self.done:
            ee = env.state()[:3]
            reached = np.linalg.norm(np.asarray(pos) - env.target_pos) < self.tol and np.linalg.norm(np.asarray(pos) - ee) < 0.015
            grip_ok = abs(env.grip - grip) < 1e-6
            if reached and grip_ok:
                if self.wait >= hold:
                    self.i, self.wait = self.i + 1, 0
                else:
                    self.wait += 1
        if self.noise > 0:
            dpos = dpos + self.rng.normal(0, self.noise, 3)
        return np.r_[dpos, 0.0, 0.0, dyaw, grip].astype(np.float32)


class ReactiveExpert:
    """A STATELESS expert: its action is a function of the current simulator state only (no waypoint counter).

    That is what interactive imitation learning needs. DAgger asks "what would the expert do HERE?" in states the
    learner wandered into; DART executes noisy actions. A waypoint state machine (ScriptedExpert) falls out of sync
    in both cases and produces contradictory labels; this one can label any state. Supports pick and place."""

    def __init__(self, env, hover=0.12, speed=1.0):
        self.env, self.hover, self.speed = env, hover, speed

    def reset(self):
        pass

    def _go(self, target, yaw, grip):
        env = self.env
        d = np.asarray(target, float) - env.target_pos
        n, step = np.linalg.norm(d), env.max_dpos * self.speed
        if n > step:
            d = d / n * step
        dyaw = float(np.clip(wrap(yaw - yaw_of(env.target_quat)), -env.max_drot, env.max_drot))
        return np.r_[d, 0.0, 0.0, dyaw, grip].astype(np.float32)

    def act(self, obs=None):
        """Funnel controller: the target height shrinks smoothly with horizontal distance to the goal, so the action is
        a CONTINUOUS function of the state almost everywhere (only the gripper toggles). Smooth labels are what an
        interactive learner (DAgger) can actually fit."""
        env, t = self.env, self.env.task
        ee, tp, cur_yaw = env.state()[:3], env.target_pos, yaw_of(env.target_quat)
        obj, w = env.cube_pos(t["obj"]), env.gripper_width()
        near = np.linalg.norm(obj[:2] - ee[:2]) < 0.02 and abs(ee[2] - (obj[2] - 0.005)) < 0.02
        holding = env.grip > 0.5 and near                           # hysteresis: once closed around the cube, stay closed
        carry_z = 0.2
        if t["type"] == "place":
            dest = env.tray_pos(t["target"])
            placed = np.all(np.abs(obj[:2] - dest[:2]) < 0.035) and obj[2] < 0.06 and not holding
            if placed:                                              # done: open and retreat upwards
                return self._go(np.r_[tp[:2], carry_z] if w > 0.045 else tp, cur_yaw, 0.0)
        if holding:
            if w > 0.047:                                           # fingers still closing: wait
                return self._go(tp, cur_yaw, 1.0)
            if t["type"] == "pick":
                return self._go(np.r_[tp[:2], carry_z], cur_yaw, 1.0)
            dxy = np.linalg.norm(ee[:2] - dest[:2])
            z = 0.06 + (carry_z - 0.06) * min(1.0, dxy / 0.04)      # funnel down onto the tray
            if dxy < 0.02 and ee[2] < 0.075:
                return self._go(tp, cur_yaw, 0.0)                   # release
            return self._go(np.r_[dest[:2], z if ee[2] > z - 0.03 or dxy < 0.04 else ee[2] + 0.05], cur_yaw, 1.0)
        # reach: funnel down onto the cube while turning to the grasp yaw
        base = env.cube_yaw(t["obj"])
        yaw = min([base + k * np.pi / 2 for k in range(-4, 5)], key=lambda y: abs(wrap(y - cur_yaw)))
        dxy = np.linalg.norm(tp[:2] - obj[:2])
        grasp_z = obj[2] - 0.005
        z = grasp_z + self.hover * min(1.0, dxy / 0.03)
        ee_dxy = np.linalg.norm(ee[:2] - obj[:2])
        if ee_dxy < 0.012 and abs(ee[2] - grasp_z) < 0.012:          # close when the GRIPPER (not the command) is at the cube:
            return self._go(np.r_[obj[:2], grasp_z], yaw, 1.0)      # one consistent label at the grasp pose, noise or not
        return self._go(np.r_[obj[:2], z], yaw, 0.0)
