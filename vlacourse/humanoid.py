"""Capstone C4: a Unitree G1 humanoid (with Dex3 hands, MuJoCo Menagerie) doing language-conditioned tabletop tasks.

Scope, stated honestly:
  * the pelvis is FIXED (no balance control: whole-body loco-manipulation is S16 151 / an extension);
  * grasping uses a contact-triggered WELD: when a closing hand's grasp point is at a cube, the cube is attached to the
    hand; opening the hand releases it. The Dex3 fingers still close visibly. Dexterous grasp synthesis with a 3-finger
    hand is a research problem of its own (an extension). Habitat and BEHAVIOR use the same abstraction.

    G1Table             env: both arms (7-DoF each) + hands, 3 cubes, a bowl, head camera; 8-D action
                        [d_right_xyz(3), d_left_xyz(3), right_grip, left_grip], IK inside (like TableTop's EE deltas)
    G1Expert            scripted demonstrator (picks with the hand on the cube's side)
    collect_g1(_parallel)  demonstrations in the vlacourse dataset format (+ 'head' images)
    G1Film              1280x720 cinematic camera
"""
import json
import os
import time

import mujoco
import mujoco_menagerie as mm
import numpy as np

COLORS = {"red": [0.9, 0.18, 0.18, 1], "green": [0.2, 0.75, 0.3, 1], "blue": [0.2, 0.4, 0.95, 1]}
TABLE_Z, HALF = 0.80, 0.02
BOWL = 0.07
ARMS = {"right": ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]}
ARMS["left"] = ARMS["right"]
HAND_OPEN = dict(thumb_0=0.0, thumb_1=0.0, thumb_2=0.0, index_0=0.0, index_1=0.0, middle_0=0.0, middle_1=0.0)
HAND_CLOSED = {"right": dict(thumb_0=0.0, thumb_1=-0.5, thumb_2=-0.8, index_0=1.0, index_1=1.0, middle_0=1.0, middle_1=1.0),
               "left": dict(thumb_0=0.0, thumb_1=0.5, thumb_2=0.8, index_0=-1.0, index_1=-1.0, middle_0=-1.0, middle_1=-1.0)}
GRASP_DZ = 0.035                                      # grasp point height above the cube centre
MAX_D = 0.04                                          # max hand displacement per control step (m), 10 Hz control
WS = np.array([[0.30, 0.38], [-0.25, 0.25]])          # cube placement area on the table (x, y)
HOME = {"right": np.array([0.18, -0.22, TABLE_Z + 0.16]), "left": np.array([0.18, 0.22, TABLE_Z + 0.16])}


def _build(image_size):
    spec = mm.get("unitree_g1").spec("scene_with_hands")
    key = spec.compile().key_qpos[0].copy()
    spec.delete(spec.joint("floating_base_joint"))
    spec.body("pelvis").pos = key[:3]
    for k in list(spec.keys):
        spec.delete(k)
    for b in spec.bodies:
        if b.name not in ("world", ""):
            b.gravcomp = 1.0                          # position servos + gravity compensation, like a real arm controller
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    spec.visual.quality.shadowsize = 4096
    wb = spec.worldbody
    wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.28, 0.5, TABLE_Z / 2], pos=[0.46, 0, TABLE_Z / 2], rgba=[0.62, 0.47, 0.34, 1])
    for c, rgba in COLORS.items():
        b = wb.add_body(name=f"cube_{c}", pos=[0.3, 0, TABLE_Z + HALF], gravcomp=0.0)
        b.add_freejoint(name=f"cube_{c}")
        b.add_geom(name=f"cube_{c}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[HALF] * 3, rgba=rgba, mass=0.04, friction=[1.5, 0.02, 0.002])
    bowl = wb.add_body(name="bowl", pos=[0.4, 0, TABLE_Z])
    t, h = 0.004, 0.025
    bowl.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[BOWL, t / 2], pos=[0, 0, t / 2], rgba=[0.95, 0.95, 0.92, 1])
    for k in range(16):                                # a round rim from 16 short boxes
        a = 2 * np.pi * k / 16
        bowl.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[t, BOWL * np.pi / 16 + 0.002, h / 2], pos=[BOWL * np.cos(a), BOWL * np.sin(a), h / 2],
                      quat=[np.cos(a / 2), 0, 0, np.sin(a / 2)], rgba=[0.95, 0.95, 0.92, 1])
    for side in ("right", "left"):
        y = 0.017 if side == "right" else -0.017
        spec.body(f"{side}_wrist_yaw_link").add_site(name=f"{side}_grasp", pos=[0.15, y, 0.0], size=[0.01] * 3, rgba=[0, 0, 0, 0])
    spec.body("torso_link").add_camera(name="head", pos=[0.06, 0, 0.42], xyaxes=[0, -1, 0, 0.62, 0, 0.78], fovy=70)
    for side, short in (("right", "rwrist"), ("left", "lwrist")):  # wrist cameras looking along the fingers (GR00T/Helix-style)
        spec.body(f"{side}_wrist_yaw_link").add_camera(name=short, pos=[0.0, 0, 0.06], xyaxes=[0, -1, 0, 0, 0, 1], fovy=85)
    for c in COLORS:                                   # one weld per (hand, cube), activated by the grasp logic
        for side in ("right", "left"):
            spec.add_equality(name=f"grasp_{side}_{c}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                              name1=f"{side}_wrist_yaw_link", name2=f"cube_{c}", active=False, solref=[0.01, 1])
    return spec, key


class G1Table:
    def __init__(self, image_size=(128, 128), cameras=("head",), seed=0, max_steps=160, control_hz=10):
        self.spec, key = _build(image_size)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.key = key[7:7 + self.model.nu]
        m = self.model
        self.act = {m.actuator(i).name: i for i in range(m.nu)}
        self.arm = {s: [m.joint(f"{s}_{n}_joint").id for n in ARMS[s]] for s in ("right", "left")}
        self.qadr = {s: [m.jnt_qposadr[j] for j in self.arm[s]] for s in self.arm}
        self.dadr = {s: [m.jnt_dofadr[j] for j in self.arm[s]] for s in self.arm}
        self.site = {s: m.site(f"{s}_grasp").id for s in ("right", "left")}
        self.n_sub = int(round(1 / (control_hz * m.opt.timestep)))
        self.cameras, self.max_steps = tuple(cameras), max_steps
        self.renderer = mujoco.Renderer(m, *image_size) if cameras else None
        self.scratch = mujoco.MjData(m)
        self.rng = np.random.default_rng(seed)
        self._vid = None
        self.reset(seed)

    # ------------------------------------------------------------------ episode
    def reset(self, seed=None, task=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        d.qpos[:len(self.key)] = self.key
        d.ctrl[:] = self.key
        d.eq_active[:] = 0
        for _ in range(200):                           # resample the whole layout until every rule holds
            side = self.rng.choice([-1, 1])              # the bowl is on the robot's left or right
            bowl_xy = np.array([self.rng.uniform(0.29, 0.35), side * self.rng.uniform(0.16, 0.22)])
            placed, cubes = [bowl_xy], {}
            for c in COLORS:
                for _ in range(100):
                    xy = self.rng.uniform(WS[:, 0], WS[:, 1])
                    if abs(xy[1]) > 0.12 and all(np.linalg.norm(xy - p) > (0.10 if k == 0 else 0.08) for k, p in enumerate(placed)):
                        break
                else:
                    break
                placed.append(xy); cubes[c] = xy
            if len(cubes) == len(COLORS) and any(np.sign(xy[1]) == side for xy in cubes.values()):   # >= 1 cube on the bowl's side
                break
        m.body_pos[m.body("bowl").id][:2] = bowl_xy
        for c, xy in cubes.items():
            adr = m.jnt_qposadr[m.joint(f"cube_{c}").id]
            yaw = self.rng.uniform(-0.4, 0.4)
            d.qpos[adr:adr + 7] = [xy[0], xy[1], TABLE_Z + HALF, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(m, d)
        self.target = {s: d.site_xpos[self.site[s]].copy() for s in ("right", "left")}
        self.grip = {"right": 0.0, "left": 0.0}
        self.holding = {"right": None, "left": None}
        for s in ("right", "left"):                    # move both hands to their home pose (above the table)
            q = self._ik(s, HOME[s], iters=400)
            self._set_arm(s, q)
            for a, v in zip(self.qadr[s], q):
                d.qpos[a] = v
        for _ in range(400):
            mujoco.mj_step(m, d)
        self.target = {s: d.site_xpos[self.site[s]].copy() for s in ("right", "left")}
        self.t = 0
        self.task = task or self.sample_task()
        return self.observe()

    def sample_task(self):
        """pick: any cube. place: a cube on the bowl's side (one hand does it; no hand-over)."""
        kind = str(self.rng.choice(["place", "pick"]))
        same = [c for c in COLORS if np.sign(self.cube_pos(c)[1]) == np.sign(self.bowl_pos()[1])]
        if kind == "place" and not same:
            kind = "pick"
        c = str(self.rng.choice(same if kind == "place" else list(COLORS)))
        if kind == "pick":
            return dict(type="pick", obj=c, instruction=f"pick up the {c} cube")
        return dict(type="place", obj=c, target="bowl", instruction=f"put the {c} cube in the bowl")

    def side_of(self, color):
        return "right" if self.cube_pos(color)[1] < 0 else "left"

    # ------------------------------------------------------------------ control
    def _ik(self, side, target, iters=60, tol=1e-4):
        m, dk = self.model, self.scratch
        dk.qpos[:] = self.data.qpos
        sid, qadr, dadr = self.site[side], self.qadr[side], self.dadr[side]
        xdir, ydir = np.array([0.0, 0, -1.0]), np.array([1.0, 0, 0]) * (1 if side == "right" else -1)
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(iters):
            mujoco.mj_kinematics(m, dk); mujoco.mj_comPos(m, dk)
            R = dk.site_xmat[sid].reshape(3, 3)
            e = np.r_[target - dk.site_xpos[sid], 0.25 * (np.cross(R[:, 0], xdir) + np.cross(R[:, 1], ydir))]
            if np.linalg.norm(e[:3]) < 1e-4:
                break
            mujoco.mj_jacSite(m, dk, jp, jr, sid)
            J = np.vstack([jp[:, dadr], 0.25 * jr[:, dadr]])
            dq = J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e)
            for a, q, j in zip(qadr, dq, self.arm[side]):
                dk.qpos[a] = np.clip(dk.qpos[a] + q, *m.jnt_range[j])
        return np.array([dk.qpos[a] for a in qadr])

    def _set_arm(self, side, q):
        for j, v in zip(self.arm[side], q):
            self.data.ctrl[self.act[self.model.joint(j).name]] = v

    def _set_hand(self, side, g):
        for k, v in HAND_OPEN.items():
            c = HAND_CLOSED[side][k]
            self.data.ctrl[self.act[f"{side}_hand_{k}_joint"]] = v + (c - v) * g

    def _update_grasp(self):
        m, d = self.model, self.data
        for s in ("right", "left"):
            if self.grip[s] > 0.5 and self.holding[s] is None:
                p = d.site_xpos[self.site[s]]
                for c in COLORS:
                    if np.linalg.norm(self.cube_pos(c) - p) < 0.05 and all(self.holding[o] != c for o in ("right", "left")):
                        eq = m.equality(f"grasp_{s}_{c}").id
                        # weld at the CURRENT relative pose: the cube stays where the fingers are
                        hb, cb = m.body(f"{s}_wrist_yaw_link").id, m.body(f"cube_{c}").id
                        rel = np.zeros(3); q = np.zeros(4); hq = np.zeros(4)
                        mujoco.mju_negQuat(hq, d.xquat[hb]); mujoco.mju_rotVecQuat(rel, d.xpos[cb] - d.xpos[hb], hq)
                        mujoco.mju_mulQuat(q, hq, d.xquat[cb])
                        m.eq_data[eq, 3:6] = rel; m.eq_data[eq, 6:10] = q; m.eq_data[eq, 10] = 1.0
                        m.eq_data[eq, 0:3] = 0
                        d.eq_active[eq] = 1; self.holding[s] = c
                        break
            elif self.grip[s] < 0.5 and self.holding[s] is not None:
                d.eq_active[m.equality(f"grasp_{s}_{self.holding[s]}").id] = 0
                self.holding[s] = None

    def step(self, action):
        a = np.asarray(action, float)
        for s, sl, g in (("right", slice(0, 3), 6), ("left", slice(3, 6), 7)):
            dp = np.clip(a[sl], -MAX_D, MAX_D)
            self.target[s] = np.clip(self.target[s] + dp, [0.08, -0.45, TABLE_Z + 0.005], [0.5, 0.45, TABLE_Z + 0.35])
            self.grip[s] = float(np.clip(a[g], 0, 1))
            q = self._ik(s, self.target[s])
            q0 = np.array([self.data.ctrl[self.act[self.model.joint(j).name]] for j in self.arm[s]])
            self._q = getattr(self, "_q", {}); self._q[s] = (q0, q)
        for k in range(self.n_sub):
            f = (k + 1) / self.n_sub
            for s in ("right", "left"):
                q0, q = self._q[s]
                self._set_arm(s, q0 + (q - q0) * f)
                self._set_hand(s, self.grip[s])
            mujoco.mj_step(self.model, self.data)
            if self._vid is not None and k % self._vid["every"] == 0:
                self._vid["frames"].append(self._vid["film"].render())
        self._update_grasp()
        self.t += 1
        ok = self.success()
        return self.observe(), float(ok), ok or self.t >= self.max_steps, {"success": ok}

    # ------------------------------------------------------------------ sensing
    def cube_pos(self, c):
        return self.data.body(f"cube_{c}").xpos.copy()

    def bowl_pos(self):
        return self.data.body("bowl").xpos.copy()

    def hand_pos(self, s):
        return self.data.site_xpos[self.site[s]].copy()

    def state(self):
        return np.r_[self.hand_pos("right"), self.hand_pos("left"), self.grip["right"], self.grip["left"]].astype(np.float32)

    def render(self, cam="head"):
        self.renderer.update_scene(self.data, cam)
        return self.renderer.render().copy()

    def observe(self):
        return dict(images={c: self.render(c) for c in self.cameras}, state=self.state(), instruction=self.task["instruction"])

    def success(self):
        t, p = self.task, self.cube_pos(self.task["obj"])
        if t["type"] == "pick":
            return bool(p[2] > TABLE_Z + 0.08)
        b = self.bowl_pos()
        return bool(np.linalg.norm(p[:2] - b[:2]) < BOWL - 0.01 and p[2] < TABLE_Z + 0.06 and self.holding["right"] != t["obj"] and self.holding["left"] != t["obj"])

    # ------------------------------------------------------------------ video
    def start_video(self, size=(720, 1280), fps=50, **film_kw):
        self._vid = dict(film=G1Film(self, size=size, **film_kw), frames=[], every=max(1, int(round(1 / (fps * self.model.opt.timestep)))))

    def stop_video(self):
        f = self._vid["frames"] if self._vid else []
        self._vid = None
        return f


class G1Expert:
    """Scripted demonstrator: the hand on the cube's side goes above it, down, closes, lifts, (moves over the bowl,
    lowers, opens), and returns. Randomized speed and approach height, like the TableTop expert."""

    def __init__(self, env, seed=0):
        self.env, rng = env, np.random.default_rng(seed)
        self.speed, self.hover = rng.uniform(0.6, 1.0), rng.uniform(0.07, 0.12)
        self.side = env.side_of(env.task["obj"])
        self.i, self.wait = 0, 0
        self.plan = self._plan()

    def _plan(self):
        e, t, s = self.env, self.env.task, self.side
        c = e.cube_pos(t["obj"])
        g = c + [0, 0, GRASP_DZ]                        # the fingertips reach ~5 cm past the grasp point: stop above the table
        wps = [(c + [0, 0, self.hover + GRASP_DZ], 0.0, 0), (g, 0.0, 0), (g, 1.0, 4), (c + [0, 0, 0.16], 1.0, 0)]
        if t["type"] == "place":
            b = e.bowl_pos()
            off = np.zeros(2)
            wps += [(np.r_[b[:2] - off, TABLE_Z + 0.14], 1.0, 0), (np.r_[b[:2] - off, TABLE_Z + 0.08], 1.0, 0),
                    (np.r_[b[:2] - off, TABLE_Z + 0.08], 0.0, 4), (np.r_[b[:2] - off, TABLE_Z + 0.15], 0.0, 0)]
        return wps

    @property
    def done(self):
        return self.i >= len(self.plan)

    def act(self, obs=None):
        e, s = self.env, self.side
        a = np.zeros(8, np.float32)
        a[6], a[7] = e.grip["right"], e.grip["left"]
        if self.done:
            return a
        pos, grip, hold = self.plan[self.i]
        d = pos - e.target[s]
        n = np.linalg.norm(d)
        step = MAX_D * self.speed
        d = d / n * step if n > step else d
        sl = slice(0, 3) if s == "right" else slice(3, 6)
        a[sl] = d
        a[6 if s == "right" else 7] = grip
        if n < 2e-3 and np.linalg.norm(e.hand_pos(s) - pos) < 0.03:
            if self.wait >= hold:
                self.i, self.wait = self.i + 1, 0
            else:
                self.wait += 1
        return a


CAMS = ("head", "rwrist", "lwrist")


def collect_g1(out_dir, n, seed=0, image_size=(128, 128), verbose=True, cameras=CAMS):
    os.makedirs(out_dir, exist_ok=True)
    env = G1Table(image_size=image_size, cameras=cameras, seed=seed)
    saved, attempts, t0 = 0, 0, time.time()
    while saved < n:
        obs = env.reset(seed=seed * 100003 + attempts); attempts += 1
        ex = G1Expert(env, seed=attempts)
        rec = {**{c: [] for c in cameras}, "state": [], "action": []}
        done = False
        while not done:
            a = ex.act(obs)
            for c in cameras:
                rec[c].append(obs["images"][c])
            rec["state"].append(obs["state"]); rec["action"].append(a)
            obs, _, done, info = env.step(a)
            if ex.done and info["success"]:
                break
        if not info["success"]:
            continue
        np.savez_compressed(os.path.join(out_dir, f"episode_{saved:05d}.npz"), **{c: np.asarray(rec[c], np.uint8) for c in cameras},
                            state=np.asarray(rec["state"], np.float32), action=np.asarray(rec["action"], np.float32),
                            instruction=np.array(env.task["instruction"]), task=np.array(json.dumps(env.task)), success=np.array(True))
        saved += 1
        if verbose and saved % 25 == 0:
            print(f"  {saved}/{n} demos ({attempts} attempts, {time.time() - t0:.0f}s)")
    return dict(saved=saved, attempts=attempts)


def _part(args):
    out, n, seed = args
    if os.name != "nt":
        os.environ.setdefault("MUJOCO_GL", "egl")
    return collect_g1(out, n, seed=seed, verbose=False)


def collect_g1_parallel(out_dir, n, workers=16, seed=0):
    import glob
    import shutil
    from concurrent.futures import ProcessPoolExecutor
    per = [n // workers + (i < n % workers) for i in range(workers)]
    parts = [os.path.join(out_dir, f"_part_{i:02d}") for i in range(workers)]
    t0 = time.time()
    # spawn, not fork: a forked worker inherits the parent's GPU/EGL rendering context, which can crash the driver
    with ProcessPoolExecutor(workers, mp_context=__import__('multiprocessing').get_context('spawn')) as ex:
        infos = list(ex.map(_part, [(p, k, seed * 1000 + 7 * i + 1) for i, (p, k) in enumerate(zip(parts, per)) if k > 0]))
    k = 0
    for p in parts:
        for f in sorted(glob.glob(os.path.join(p, "episode_*.npz"))):
            shutil.move(f, os.path.join(out_dir, f"episode_{k:05d}.npz")); k += 1
        shutil.rmtree(p, ignore_errors=True)
    attempts = sum(i["attempts"] for i in infos)
    meta = dict(robot="unitree_g1_with_hands", n_episodes=k, attempts=attempts, expert_success=k / attempts, cameras=list(CAMS))
    json.dump(meta, open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
    print(f"{k} G1 demos from {attempts} attempts (expert success {k / attempts:.0%}) in {time.time() - t0:.0f}s with {workers} workers")
    return meta


class G1Film:
    """1280x720 view from the front-left, slightly above the table, looking at the robot working."""

    def __init__(self, env, size=(720, 1280), azimuth=205, elevation=-24, distance=1.0, lookat=(0.3, 0.0, 0.9)):
        self.env, self.r = env, mujoco.Renderer(env.model, *size)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = lookat
        self.cam.azimuth, self.cam.elevation, self.cam.distance = azimuth, elevation, distance

    def render(self):
        self.r.update_scene(self.env.data, self.cam)
        return self.r.render().copy()
