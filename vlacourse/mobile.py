"""Capstone C7: mobile manipulation with the Stanford TidyBot (holonomic base + Kinova Gen3 + Robotiq 2F-85, Menagerie).

"Bring the green cube to the box": drive to the table, pick the right cube (a learned, language-conditioned skill
from the wrist and base cameras), drive to the box, drop it in.

    MobileManip         the room: a table with 3 cubes, a box, the robot. Two control modes, both 10 Hz:
                          drive(v_xy, w)        base velocity (limited speed and acceleration, like 42b)
                          step(a7)              arm: [dx, dy, dz, 0, 0, dyaw, grip] of the pinch point (IK inside),
                                                the same 7-D action as TableTop, expressed in the BASE frame
    PickExpert          scripted top-down grasp (aligned to the cube's yaw), the demonstrator for the pick skill
    navigate(...)       a simple holonomic go-to-pose controller (P + limits)
    collect_picks(...)  pick demonstrations (cameras 'mast' + 'wrist', TableTop dataset format)
    Follow              1280x720 camera that follows the robot around the room
"""
import json
import os
import time

import mujoco
import mujoco_menagerie as mm
import numpy as np

COLORS = {"red": [0.9, 0.18, 0.18, 1], "green": [0.2, 0.75, 0.3, 1], "blue": [0.2, 0.4, 0.95, 1]}
TABLE = dict(pos=(2.2, 0.0), half=(0.25, 0.45), z=0.45)
BOX = dict(pos=(0.3, 2.0), half=0.14, h=0.12)
HALF = 0.02
V_MAX, W_MAX, A_MAX, DT = 0.6, 1.0, 1.0, 0.1
STANDOFF = 0.62                                        # base centre to cube, along the base's x axis, when picking
ARM_HOME = np.array([0.0, 0.26, 3.14, -2.27, 0.0, 0.96, 1.57])


def yaw_of(q):
    return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class MobileManip:
    def __init__(self, image_size=(128, 128), cameras=("mast", "wrist"), seed=0, max_steps=150):
        spec = mm.get("stanford_tidybot").spec("scene")
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
        spec.visual.quality.shadowsize = 4096
        wb = spec.worldbody
        (tx, ty), (hx, hy), tz = TABLE["pos"], TABLE["half"], TABLE["z"]
        wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, tz / 2], pos=[tx, ty, tz / 2], rgba=[0.62, 0.47, 0.34, 1])
        for c, rgba in COLORS.items():
            b = wb.add_body(name=f"cube_{c}", pos=[tx, 0, tz + HALF])
            b.add_freejoint(name=f"cube_{c}")
            b.add_geom(name=f"cube_{c}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[HALF] * 3, rgba=rgba, mass=0.05, friction=[1.5, 0.02, 0.002])
        (bx, by), bh, t, h = BOX["pos"], BOX["half"], 0.01, BOX["h"]
        box = wb.add_body(name="box", pos=[bx, by, 0])
        box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[bh, bh, t / 2], pos=[0, 0, t / 2], rgba=[0.55, 0.4, 0.25, 1])
        for sx, sy, w, l in [(1, 0, t, bh), (-1, 0, t, bh), (0, 1, bh, t), (0, -1, bh, t)]:
            box.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[w, l, h / 2], pos=[sx * (bh - t), sy * (bh - t), h / 2], rgba=[0.55, 0.4, 0.25, 1])
        # a camera mast: the stock base camera sits below the tabletop and sees only the table's side
        from .tabletop import lookat_xyaxes
        bl = spec.body("base_link")
        bl.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.015, 0.5], pos=[-0.12, 0, 0.85], rgba=[0.3, 0.3, 0.32, 1], contype=0, conaffinity=0)
        bl.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.03, 0.06, 0.025], pos=[-0.1, 0, 1.36], rgba=[0.15, 0.15, 0.17, 1], contype=0, conaffinity=0)
        bl.add_camera(name="mast", pos=[-0.07, 0, 1.36], xyaxes=lookat_xyaxes([-0.07, 0, 1.36], [0.66, 0, 0.45]), fovy=62)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.arm = [m.joint(f"joint_{i}").id for i in range(1, 8)]
        self.qadr = [m.jnt_qposadr[j] for j in self.arm]
        self.dadr = [m.jnt_dofadr[j] for j in self.arm]
        self.site = m.site("pinch_site").id
        self.grip_act = m.actuator("fingers_actuator").id
        self.n_sub = int(round(DT / m.opt.timestep))
        self.cameras, self.max_steps = tuple(cameras), max_steps
        self.renderer = mujoco.Renderer(m, *image_size) if cameras else None
        self.scratch = mujoco.MjData(m)
        self.rng = np.random.default_rng(seed)
        self._vid = None
        self.reset(seed)

    # ------------------------------------------------------------------ episode
    def reset(self, seed=None, start=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        (tx, ty), (hx, hy), tz = TABLE["pos"], TABLE["half"], TABLE["z"]
        placed = []
        for c in COLORS:
            for _ in range(500):
                xy = np.array([tx - hx + self.rng.uniform(0.08, 0.16), ty + self.rng.uniform(-hy + 0.1, hy - 0.1)])
                if all(np.linalg.norm(xy - p) > 0.12 for p in placed):
                    break
            placed.append(xy)
            adr = m.jnt_qposadr[m.joint(f"cube_{c}").id]
            yaw = self.rng.uniform(-0.6, 0.6)
            d.qpos[adr:adr + 7] = [xy[0], xy[1], tz + HALF, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        start = start if start is not None else np.r_[self.rng.uniform(-0.3, 0.3), self.rng.uniform(-0.3, 0.3), self.rng.uniform(-np.pi, np.pi)]
        d.qpos[:3] = start
        d.ctrl[:3] = start
        d.qpos[self.qadr] = ARM_HOME
        d.ctrl[3:10] = ARM_HOME
        mujoco.mj_forward(m, d)
        self.v = np.zeros(3)
        self.grip = 0.0
        self.target = self.pinch_base()                 # commanded pinch position, base frame
        self.target_yaw = 0.0
        self.t = 0
        c = str(self.rng.choice(list(COLORS)))
        self.task = dict(type="bring", obj=c, target="box", instruction=f"bring the {c} cube to the box")
        return self.observe()

    # ------------------------------------------------------------------ frames
    def base_pose(self):
        return self.data.qpos[:3].copy()

    def to_base(self, p):
        x, y, th = self.base_pose()
        c, s = np.cos(th), np.sin(th)
        d = np.asarray(p)[:2] - [x, y]
        return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1], p[2]])

    def to_world(self, p):
        x, y, th = self.base_pose()
        c, s = np.cos(th), np.sin(th)
        return np.array([x + c * p[0] - s * p[1], y + s * p[0] + c * p[1], p[2]])

    def pinch_base(self):
        return self.to_base(self.data.site_xpos[self.site])

    def cube_pos(self, c):
        return self.data.body(f"cube_{c}").xpos.copy()

    def cube_yaw(self, c):
        return yaw_of(self.data.body(f"cube_{c}").xquat)

    # ------------------------------------------------------------------ control
    def _physics(self, base_target=None, arm_q=None):
        d = self.data
        q0 = d.ctrl[3:10].copy()
        b0 = d.ctrl[:3].copy()
        for k in range(self.n_sub):
            f = (k + 1) / self.n_sub
            if base_target is not None:
                d.ctrl[:3] = b0 + (base_target - b0) * f
            if arm_q is not None:
                d.ctrl[3:10] = q0 + (arm_q - q0) * f
            d.ctrl[self.grip_act] = 255.0 * self.grip
            mujoco.mj_step(self.model, d)
            if self._vid is not None and k % self._vid["every"] == 0:
                self._vid["frames"].append(self._vid["film"].render())
        self.t += 1

    def drive(self, v_xy, w=0.0):
        """World-frame base velocity command (m/s, rad/s), limited in speed and acceleration. The arm holds still."""
        cmd = np.r_[np.asarray(v_xy, float), w]
        n = np.linalg.norm(cmd[:2])
        if n > V_MAX:
            cmd[:2] *= V_MAX / n
        cmd[2] = np.clip(cmd[2], -W_MAX, W_MAX)
        dv = cmd - self.v
        nd = np.linalg.norm(dv[:2])
        if nd > A_MAX * DT:
            dv[:2] *= A_MAX * DT / nd
        self.v = self.v + dv
        target = self.data.ctrl[:3] + self.v * DT
        self._physics(base_target=target)
        return self.observe()

    def _ik(self, target_world, yaw_world, iters=80):
        m, dk = self.model, self.scratch
        dk.qpos[:] = self.data.qpos
        zdir = np.array([0, 0, -1.0])
        xdir = np.array([-np.sin(yaw_world), np.cos(yaw_world), 0.0])   # finger closing axis follows the cube's yaw
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, dk); mujoco.mj_comPos(m, dk)
            R = dk.site_xmat[self.site].reshape(3, 3)
            e = np.r_[target_world - dk.site_xpos[self.site], 0.5 * (np.cross(R[:, 2], zdir) + np.cross(R[:, 0], xdir))]
            if np.linalg.norm(e) < 1e-4:
                break
            mujoco.mj_jacSite(m, dk, jp, jr, self.site)
            J = np.vstack([jp[:, self.dadr], 0.5 * jr[:, self.dadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
            for a, q, j in zip(self.qadr, dq, self.arm):
                dk.qpos[a] += q
                if m.jnt_limited[j]:
                    dk.qpos[a] = np.clip(dk.qpos[a], *m.jnt_range[j])
        return np.array([dk.qpos[a] for a in self.qadr])

    def step(self, a):
        """Arm action in the BASE frame: [dx, dy, dz, 0, 0, dyaw, grip] (TableTop's convention). The base holds still."""
        a = np.asarray(a, float)
        self.target = np.clip(self.target + np.clip(a[:3], -0.05, 0.05), [0.25, -0.45, 0.25], [0.9, 0.45, 1.0])
        self.target_yaw = wrap(self.target_yaw + float(np.clip(a[5], -0.3, 0.3)))
        self.grip = float(np.clip(a[6], 0, 1))
        q = self._ik(self.to_world(self.target), self.base_pose()[2] + self.target_yaw)
        self._physics(arm_q=q)
        obs = self.observe()
        ok = self.picked()
        return obs, float(ok), ok or self.t >= self.max_steps, {"success": ok}

    def retract(self, carry=(0.42, 0.0, 0.80), steps=20):
        """Lift the gripper (still pointing down, still holding) to a carry pose above the base: smooth, no wrist flip.
        (Jumping straight to the folded home pose whips the wrist around and throws the cube.)"""
        for _ in range(steps):
            d = np.clip(np.asarray(carry) - self.target, -0.05, 0.05)
            self.step(np.r_[d, 0, 0, -np.clip(self.target_yaw, -0.3, 0.3), self.grip])
            if np.linalg.norm(np.asarray(carry) - self.target) < 1e-3 and abs(self.target_yaw) < 1e-3:
                break

    # ------------------------------------------------------------------ checks + sensing
    def picked(self):
        return bool(self.cube_pos(self.task["obj"])[2] > TABLE["z"] + 0.08)

    def delivered(self):
        p, (bx, by) = self.cube_pos(self.task["obj"]), BOX["pos"]
        return bool(abs(p[0] - bx) < BOX["half"] - 0.02 and abs(p[1] - by) < BOX["half"] - 0.02 and p[2] < BOX["h"])

    def state(self):
        """8-D like TableTop: pinch position (base frame), pinch orientation as a quaternion of its yaw, gripper opening."""
        p = self.pinch_base()
        y = self.target_yaw
        width = 1.0 - self.grip
        return np.r_[p, np.cos(y / 2), 0, 0, np.sin(y / 2), width].astype(np.float32)

    def render(self, cam):
        self.renderer.update_scene(self.data, cam)
        return self.renderer.render().copy()

    def observe(self):
        return dict(images={c: self.render(c) for c in self.cameras}, state=self.state(), instruction=self.task["instruction"])

    def start_video(self, size=(720, 1280), fps=50, **kw):
        self._vid = dict(film=Follow(self, size=size, **kw), frames=[], every=max(1, int(round(1 / (fps * self.model.opt.timestep)))))

    def stop_video(self):
        f = self._vid["frames"] if self._vid else []
        self._vid = None
        return f


def navigate(env, goal_xy, goal_th, tol=0.03, max_steps=200, on_step=None):
    """Holonomic go-to-pose: P control on position and heading, with the base's speed/acceleration limits; slows down
    near the goal so it stops on target. Returns True when it arrives."""
    for _ in range(max_steps):
        x, y, th = env.base_pose()
        e = np.asarray(goal_xy) - [x, y]
        de = wrap(goal_th - th)
        if np.linalg.norm(e) < tol and abs(de) < 0.03 and np.linalg.norm(env.v[:2]) < 0.05:
            return True
        env.drive(1.5 * e, 2.0 * de)
        if on_step is not None:
            on_step(env)
    return False


READY = np.array([0.36, 0.0, 0.80])                   # pinch "ready" pose before a pick: gripper down, close to the base


def ready_arm(env, steps=25):
    """Bring the gripper (pointing down) to READY in the base frame: out of the mast camera's view of the table."""
    env.target, env.target_yaw = env.pinch_base(), 0.0
    for _ in range(steps):
        d = np.clip(READY - env.target, -0.05, 0.05)
        env.step(np.r_[d, 0, 0, 0, env.grip])
        if np.linalg.norm(READY - env.target) < 1e-3:
            break
    for _ in range(5):
        env.step(np.zeros(7))


def pick_pose(env, color=None, jitter=True):
    """Where the base parks to pick: a FIXED spot in front of the table (small random error, like a real base's
    localization). It does not depend on where the cube is: finding the right cube is the learned policy's job."""
    (tx, ty), (hx, hy), _ = TABLE["pos"], TABLE["half"], TABLE["z"]
    j = env.rng.uniform(-1, 1, 3) * [0.03, 0.03, 0.05] if jitter else np.zeros(3)
    return np.array([tx - hx - 0.40 + j[0], ty + j[1]]), j[2]


class PickExpert:
    """Top-down grasp: above the cube, down, close, lift. Actions in the base frame (the 7-D TableTop convention)."""

    def __init__(self, env, seed=0):
        rng = np.random.default_rng(seed)
        self.env, self.speed, self.hover = env, rng.uniform(0.6, 1.0), rng.uniform(0.08, 0.14)
        e, c = env, env.task["obj"]
        p = e.to_base(e.cube_pos(c))
        yaw = wrap(e.cube_yaw(c) - e.base_pose()[2])
        yaw = min([yaw + k * np.pi / 2 for k in range(-2, 3)], key=abs)
        self.plan = [(p + [0, 0, self.hover], yaw, 0.0, 0), (p + [0, 0, 0.0], yaw, 0.0, 0), (p, yaw, 1.0, 6), (p + [0, 0, 0.18], yaw, 1.0, 0)]
        self.i, self.wait = 0, 0

    @property
    def done(self):
        return self.i >= len(self.plan)

    def act(self, obs=None):
        e = self.env
        if self.done:
            return np.r_[0, 0, 0, 0, 0, 0, e.grip].astype(np.float32)
        pos, yaw, grip, hold = self.plan[self.i]
        d = pos - e.target
        n = np.linalg.norm(d)
        step = 0.05 * self.speed
        d = d / n * step if n > step else d
        dy = float(np.clip(wrap(yaw - e.target_yaw), -0.3, 0.3))
        if n < 2e-3 and abs(wrap(yaw - e.target_yaw)) < 0.02 and abs(e.grip - grip) < 1e-6:
            if self.wait >= hold:
                self.i, self.wait = self.i + 1, 0
            else:
                self.wait += 1
        return np.r_[d, 0, 0, dy, grip].astype(np.float32)


def collect_picks(out_dir, n, seed=0, image_size=(128, 128), verbose=True):
    """Pick demonstrations in the TableTop dataset format (cameras 'base' and 'wrist'). Each episode: the robot is
    driven to the pick pose (not recorded), then the scripted pick is recorded until the cube is lifted."""
    from .data import _stats
    os.makedirs(out_dir, exist_ok=True)
    env = MobileManip(image_size=image_size, cameras=("mast", "wrist"), seed=seed)
    saved, attempts, t0, acts, states, lens = 0, 0, time.time(), [], [], []
    while saved < n:
        env.reset(seed=seed * 100003 + attempts); attempts += 1
        xy, th = pick_pose(env, env.task["obj"])
        if not navigate(env, xy, th):
            continue
        ready_arm(env)
        env.target, env.target_yaw, env.t = env.pinch_base(), 0.0, 0
        obs = env.observe()
        ex = PickExpert(env, seed=attempts)
        rec = dict(mast=[], wrist=[], state=[], action=[], executed=[])
        done = False
        while not done:
            a = ex.act(obs)
            rec["mast"].append(obs["images"]["mast"]); rec["wrist"].append(obs["images"]["wrist"])
            rec["state"].append(obs["state"]); rec["action"].append(a); rec["executed"].append(a)
            obs, _, done, info = env.step(a)
            if ex.done:
                break
        if not env.picked():
            continue
        arrays = {k: (np.asarray(v, np.uint8) if k in ("mast", "wrist") else np.asarray(v, np.float32)) for k, v in rec.items()}
        np.savez_compressed(os.path.join(out_dir, f"episode_{saved:05d}.npz"), **arrays, instruction=np.array(env.task["instruction"]),
                            task=np.array(json.dumps(dict(type="pick", obj=env.task["obj"]))), success=np.array(True))
        acts.append(arrays["action"]); states.append(arrays["state"]); lens.append(len(arrays["action"]))
        saved += 1
        if verbose and saved % 25 == 0:
            print(f"  {saved}/{n} pick demos ({attempts} attempts, {time.time() - t0:.0f}s)")
    return dict(saved=saved, attempts=attempts, actions=acts, states=states, lengths=lens)


def _part(args):
    out, n, seed = args
    if os.name != "nt":
        os.environ.setdefault("MUJOCO_GL", "egl")
    r = collect_picks(out, n, seed=seed, verbose=False)
    return dict(saved=r["saved"], attempts=r["attempts"])


def collect_picks_parallel(out_dir, n, workers=8, seed=0):
    import glob
    import shutil
    from concurrent.futures import ProcessPoolExecutor
    from .data import _stats, load_episode
    per = [n // workers + (i < n % workers) for i in range(workers)]
    parts = [os.path.join(out_dir, f"_part_{i:02d}") for i in range(workers)]
    t0 = time.time()
    # spawn, not fork: a forked worker inherits the parent's GPU/EGL rendering context, which can crash the driver
    with ProcessPoolExecutor(workers, mp_context=__import__('multiprocessing').get_context('spawn')) as ex:
        infos = list(ex.map(_part, [(p, k, seed * 1000 + 11 * i + 5) for i, (p, k) in enumerate(zip(parts, per)) if k > 0]))
    k, acts, states, lens = 0, [], [], []
    for p in parts:
        for f in sorted(glob.glob(os.path.join(p, "episode_*.npz"))):
            dst = os.path.join(out_dir, f"episode_{k:05d}.npz"); shutil.move(f, dst); k += 1
            ep = load_episode(dst); acts.append(ep["action"]); states.append(ep["state"]); lens.append(len(ep["action"]))
        shutil.rmtree(p, ignore_errors=True)
    attempts = sum(i["attempts"] for i in infos)
    meta = dict(robot="stanford_tidybot", image_size=[128, 128], cameras=["mast", "wrist"], control_hz=10, task_types=["pick"], n_cubes=3, n_trays=0,
                n_episodes=k, attempts=attempts, n_frames=int(sum(lens)), episode_lengths=lens, action_stats=_stats(acts), state_stats=_stats(states),
                action_names=["dx", "dy", "dz", "droll", "dpitch", "dyaw", "grip"], env_kw={})
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print(f"{k} pick demos from {attempts} attempts ({k / attempts:.0%}) in {time.time() - t0:.0f}s with {workers} workers")
    return meta


class Follow:
    """A camera that follows the robot from behind-above, smoothly, through the whole room."""

    def __init__(self, env, size=(720, 1280), distance=2.6, elevation=-28, azimuth_offset=150):
        self.env, self.r = env, mujoco.Renderer(env.model, *size)
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.elevation = distance, elevation
        self.off, self.look, self.az = azimuth_offset, None, None

    def render(self):
        x, y, th = self.env.base_pose()
        p = np.array([x, y, 0.45])
        self.look = p if self.look is None else 0.9 * self.look + 0.1 * p
        az = np.degrees(th) + self.off
        self.az = az if self.az is None else self.az + 0.05 * ((az - self.az + 180) % 360 - 180)
        self.cam.lookat[:], self.cam.azimuth = self.look, self.az
        self.r.update_scene(self.env.data, self.cam)
        return self.r.render().copy()
