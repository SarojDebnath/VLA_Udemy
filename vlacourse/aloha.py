"""Bimanual ALOHA transfer-cube task (capstone C2): joint-space environment + scripted handover expert.

Action (14-D, joint targets like the original ACT/ALOHA setup):
    [left 6 joints, left gripper (0.002 closed .. 0.037 open), right 6 joints, right gripper]
"""
import mujoco
import mujoco_menagerie as mm
import numpy as np

OPEN, CLOSED = 0.037, 0.002
ARMS = {"left": (list(range(6)), 6), "right": (list(range(8, 14)), 13)}
CTRL = {"left": [0, 1, 2, 3, 4, 5], "right": [7, 8, 9, 10, 11, 12]}
GRIP = {"left": 6, "right": 13}


def ik(m, d, arm, target, approach, q0, w=0.5, iters=300):
    """Position + soft approach-direction IK for one 6-DoF ViperX arm (the jaw points along the gripper site's x axis)."""
    dofs, _ = ARMS[arm]
    sid = m.site(f"{arm}/gripper").id
    q, dk = q0.copy(), mujoco.MjData(m)
    dk.qpos[:] = d.qpos
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        dk.qpos[dofs] = q
        mujoco.mj_kinematics(m, dk); mujoco.mj_comPos(m, dk)
        a = dk.site_xmat[sid].reshape(3, 3)[:, 0]
        e = np.r_[target - dk.site_xpos[sid], w * np.cross(a, approach)]
        mujoco.mj_jacSite(m, dk, jp, jr, sid)
        J = np.vstack([jp[:, dofs], w * jr[:, dofs]])
        q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 1e-3 * np.eye(6), e), m.jnt_range[dofs, 0], m.jnt_range[dofs, 1])
    return q


class AlohaTransfer:
    def __init__(self, cameras=("overhead_cam", "wrist_cam_right"), image_size=(96, 96), control_hz=25, max_steps=300, seed=0):
        spec = mm.get("aloha").spec("scene")
        b = spec.worldbody.add_body(name="cube", pos=[-0.1, 0, 0.02])
        b.add_freejoint()
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.02] * 3, rgba=[0.9, 0.2, 0.2, 1], mass=0.05, friction=[1.5, 0.01, 0.001])
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.n_sub = int(round(1 / (control_hz * self.model.opt.timestep)))
        self.cameras, self.max_steps = cameras, max_steps
        self.renderer = mujoco.Renderer(self.model, *image_size) if cameras else None
        self.rng = np.random.default_rng(seed)
        self.reset(seed)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, 0)
        xy = self.rng.uniform([-0.15, -0.08], [-0.06, 0.08])
        d.qpos[16:19] = [*xy, 0.02]
        yaw = self.rng.uniform(-0.4, 0.4)
        d.qpos[19:23] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.ctrl[:] = 0
        d.ctrl[CTRL["left"]], d.ctrl[CTRL["right"]] = d.qpos[0:6], d.qpos[8:14]
        d.ctrl[6] = d.ctrl[13] = OPEN
        mujoco.mj_forward(m, d)
        self.t = 0
        return self.observe()

    def state(self):
        d = self.data
        return np.r_[d.qpos[0:6], d.qpos[6], d.qpos[8:14], d.qpos[14]].astype(np.float32)

    def observe(self):
        imgs = {}
        for c in self.cameras:
            self.renderer.update_scene(self.data, c)
            imgs[c] = self.renderer.render().copy()
        return {"images": imgs, "state": self.state()}

    def success(self):
        c = self.data.body("cube").xpos
        return bool(c[0] > 0.05 and c[2] > 0.1)

    def step(self, action):
        a = np.asarray(action, float)
        self.data.ctrl[CTRL["left"]], self.data.ctrl[6] = a[0:6], np.clip(a[6], CLOSED, OPEN)
        self.data.ctrl[CTRL["right"]], self.data.ctrl[13] = a[7:13], np.clip(a[13], CLOSED, OPEN)
        mujoco.mj_step(self.model, self.data, self.n_sub)
        self.t += 1
        ok = self.success()
        return self.observe(), float(ok), ok or self.t >= self.max_steps, {"success": ok}


class HandoverExpert:
    """Phased scripted expert; each phase re-measures the cube, solves IK, and interpolates joint targets (closed loop at phase level)."""

    def __init__(self, env, phase_steps=25, seed=0):
        self.env, self.n = env, phase_steps
        rng = np.random.default_rng(seed)
        self.mid = np.array([rng.uniform(-0.03, 0.03), rng.uniform(-0.03, 0.03), rng.uniform(0.17, 0.23)])
        self.phases = ["l_above", "l_down", "l_close", "l_mid", "r_approach", "r_in", "r_close", "l_open", "retreat"]
        self.k, self.i = 0, 0
        d = env.data
        self.q = {"left": d.qpos[0:6].copy(), "right": d.qpos[8:14].copy()}
        self.grip = {"left": OPEN, "right": OPEN}
        self._start_phase()

    def _start_phase(self):
        m, d = self.env.model, self.env.data
        down, side = np.array([0, 0, -1.0]), np.array([-1.0, 0, 0])
        cube = d.body("cube").xpos.copy()
        name = self.phases[self.k] if self.k < len(self.phases) else "hold"
        tg = {"l_above": {"left": (cube + [0, 0, 0.08], down)}, "l_down": {"left": (cube, down)},
              "l_mid": {"left": (self.mid + [-0.03, 0, 0], down)}, "r_approach": {"right": (cube + [0.10, 0, 0], side)},
              "r_in": {"right": (cube + [0.01, 0, 0], side)},
              "retreat": {"right": (self.mid + [0.15, 0, 0.05], side), "left": (self.mid + [-0.2, 0, 0.05], down)}}.get(name, {})
        if name == "l_close":
            self.grip["left"] = CLOSED
        if name == "r_close":
            self.grip["right"] = CLOSED
        if name == "l_open":
            self.grip["left"] = OPEN
        self.start = {a: v.copy() for a, v in self.q.items()}
        self.goal = {a: ik(m, d, a, np.array(t), app, self.q[a]) for a, (t, app) in tg.items()}
        self.i = 0

    def act(self, obs=None):
        s = min(1.0, (self.i + 1) / (0.8 * self.n)); s = 3 * s * s - 2 * s ** 3
        for a in self.q:
            if a in self.goal:
                self.q[a] = self.start[a] + (self.goal[a] - self.start[a]) * s
        self.i += 1
        if self.i >= self.n and self.k < len(self.phases):
            self.k += 1
            self._start_phase()
        return np.r_[self.q["left"], self.grip["left"], self.q["right"], self.grip["right"]].astype(np.float32)


# ----------------------------------------------------------------------------- share-ready HD video
class AlohaFilm:
    """Cinematic HD recording of an AlohaTransfer episode, independent of the policy's low-res cameras.

    A free camera slowly orbits the table at 1280x720 with 8x MSAA; frames are rendered at `sub` x the control rate by
    interpolating the simulator state between control steps (smooth 50 fps at sub=2). The policy's actual camera
    inputs are shown, honestly at their real resolution, as a picture-in-picture strip.
        film = AlohaFilm(env); film.begin(); ...each control step: film.add(env, obs, phase="...") ...; frames = film.end(ok)
    """

    def __init__(self, env, size=(720, 1280), sub=2, azimuth=(70, 110), distance=1.05, elevation=-24, lookat=(0, 0, 0.12),
                 title="ACT · bimanual cube handover", pip=True):
        m = env.model
        H, W = size
        m.vis.global_.offwidth, m.vis.global_.offheight = max(W, m.vis.global_.offwidth), max(H, m.vis.global_.offheight)
        m.vis.quality.offsamples = 8
        m.vis.quality.shadowsize = 8192
        self.env, self.size, self.sub, self.az, self.title, self.pip = env, size, sub, azimuth, title, pip
        self.r = mujoco.Renderer(m, H, W)
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:], self.cam.distance, self.cam.elevation = lookat, distance, elevation
        self.opt = mujoco.MjvOption()
        # the aluminium frame + camera mounts clutter the shot: hidden in the film view only
        self.hide = np.where(m.geom_group == 1)[0]
        self.scratch = mujoco.MjData(m)

    def begin(self, max_steps=None):
        self.frames, self.prev, self.k = [], self.env.data.qpos.copy(), 0
        self.T = (max_steps or self.env.max_steps) * self.sub
        return self

    def _render(self, qpos):
        m, d = self.env.model, self.scratch
        pos = m.geom_pos[self.hide].copy()
        m.geom_pos[self.hide, 2] -= 50.0                     # frame out of the shot (and its shadows), film data only
        cast = m.light_castshadow.copy()
        m.light_castshadow[:] = 0; m.light_castshadow[np.argmax(-m.light_dir[:, 2])] = 1   # one clean shadow from the top light
        d.qpos[:] = qpos
        mujoco.mj_forward(m, d)
        frac = min(1.0, self.k / max(1, self.T * 0.75))
        self.cam.azimuth = self.az[0] + (self.az[1] - self.az[0]) * (0.5 - 0.5 * np.cos(np.pi * frac))   # ease in/out
        self.r.update_scene(d, self.cam, self.opt)
        m.geom_pos[self.hide] = pos                          # restored before the policy's cameras render again
        m.light_castshadow[:] = cast
        self.k += 1
        return self.r.render().copy()

    def add(self, env, obs=None, phase=""):
        """Call once after every env.step: renders `sub` frames interpolated from the previous control step."""
        q1 = env.data.qpos.copy()
        for i in range(1, self.sub + 1):
            a = i / self.sub
            q = (1 - a) * self.prev + a * q1
            q[19:23] /= np.linalg.norm(q[19:23])                        # cube quaternion
            f = self._render(q)
            if self.pip and obs is not None:
                f = self._pip(f, obs)
            self.frames.append((f, phase, env.t / 25))
        self.prev = q1

    def _overlay(self, f, phase, t, color=None):
        from .showcase import caption, FG
        f = caption(f, f"{self.title}      t = {t:4.1f} s", where="top", size=max(18, self.size[0] // 28))
        return caption(f, phase, where="bottom", size=max(16, self.size[0] // 30), color=color or FG) if phase else f

    def _pip(self, f, obs):
        from PIL import Image, ImageDraw
        from .showcase import font, FG, BG
        im = Image.fromarray(f)
        s = self.size[0] // 5                                               # 144 px tiles at 720p
        x0, y0 = self.size[1] - 2 * s - 36, self.size[0] - s - 70
        d = ImageDraw.Draw(im, "RGBA")
        d.rounded_rectangle([x0 - 10, y0 - 34, x0 + 2 * s + 18, y0 + s + 10], radius=10, fill=(*BG, 200))
        d.text((x0, y0 - 28), "what the policy sees (96×96)", font=font(max(12, s // 10)), fill=FG)
        for j, c in enumerate([c for c in ("overhead_cam", "wrist_cam_right") if c in obs["images"]]):
            tile = Image.fromarray(obs["images"][c]).resize((s, s), Image.NEAREST)
            im.paste(tile, (x0 + j * (s + 8), y0))
        return np.asarray(im)

    def end(self, ok, hold=40):
        """Finished frames: overlays on every frame, then the last frame held with the outcome."""
        from .showcase import GOOD, BAD
        out = [self._overlay(f, ph, t) for f, ph, t in self.frames]
        f, _, t = self.frames[-1]
        out += [self._overlay(f, "✓ handover complete" if ok else "✗ failed", t, GOOD if ok else BAD)] * hold
        return out
