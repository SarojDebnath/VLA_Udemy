"""S04 project 42b: model predictive control by predictive sampling, on a mobile robot crossing a crowded room.

    CrowdRoom          Stanford TidyBot (Menagerie) + walking "people" (moving cylinders). The base takes velocity
                       commands with an acceleration limit, like a real mobile base: it cannot stop instantly.
    ReactiveController PID to the goal + repulsion from where the people are NOW (potential field)
    PredictiveSampling the sampling-based MPC used by MuJoCo MPC (Howell et al. 2022): perturb the current plan,
                       roll every candidate forward with a model (including where the people WILL be), keep the
                       cheapest, execute its first action, repeat at 10 Hz
    Film               1280x720 3/4 view; draws the sampled futures (faint) and the chosen plan (bright)
"""
import mujoco
import mujoco_menagerie as mm
import numpy as np

R_BASE, R_PERSON = 0.30, 0.22
V_MAX, A_MAX, DT = 1.0, 1.5, 0.1


class CrowdRoom:
    def __init__(self, n_people=5, seed=0, start=(-3.0, 0.0), goal=(3.0, 0.0)):
        spec = mm.get("stanford_tidybot").spec("scene")
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
        spec.visual.quality.shadowsize = 4096
        self.n = n_people
        cols = [[0.95, 0.35, 0.3, 1], [0.95, 0.75, 0.2, 1], [0.4, 0.7, 0.95, 1], [0.7, 0.45, 0.9, 1], [0.35, 0.85, 0.5, 1], [0.9, 0.5, 0.75, 1]]
        for i in range(n_people):
            b = spec.worldbody.add_body(name=f"person{i}", mocap=True, pos=[0, 0, 0.8])
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[R_PERSON, 0.55], rgba=cols[i % len(cols)], contype=0, conaffinity=0)
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.14], pos=[0, 0, 0.85], rgba=[0.95, 0.85, 0.75, 1], contype=0, conaffinity=0)
        g = spec.worldbody.add_body(name="goal", mocap=True, pos=[*goal, 0.005])
        g.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.35, 0.005], rgba=[0.2, 0.9, 0.4, 0.6], contype=0, conaffinity=0)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.n_sub = int(round(DT / self.model.opt.timestep))
        self.start, self.goal = np.array(start, float), np.array(goal, float)
        self.reset(seed)

    def reset(self, seed=0):
        rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, m.key("retract").id)
        d.qpos[:2] = self.start
        d.ctrl[:] = m.key_ctrl[m.key("retract").id] if m.nkey else d.ctrl
        d.ctrl[:2] = self.start
        # people walk back and forth ACROSS the robot's path (along y), each with its own speed and phase
        xs = np.linspace(-1.8, 1.8, self.n) + rng.uniform(-0.25, 0.25, self.n)
        self.people = [dict(x=x, amp=rng.uniform(1.4, 2.2), w=rng.uniform(0.35, 0.7), ph=rng.uniform(0, 2 * np.pi)) for x in xs]
        self.t, self.v = 0.0, np.zeros(2)
        self.collisions, self.min_clear = 0, np.inf
        self._move_people()
        mujoco.mj_forward(m, d)
        return self.obs()

    def people_at(self, t):
        """(n, 2) positions at time t: the 'ground truth' motion (a controller only ever sees the current state)."""
        return np.array([[p["x"], p["amp"] * np.sin(p["w"] * t + p["ph"])] for p in self.people])

    def _move_people(self):
        for i, xy in enumerate(self.people_at(self.t)):
            self.data.mocap_pos[i] = [xy[0], xy[1], 0.8]

    def obs(self):
        P = self.people_at(self.t)
        Vp = (self.people_at(self.t + 1e-3) - P) / 1e-3            # a tracker's velocity estimate
        return dict(pos=self.data.qpos[:2].copy(), vel=self.v.copy(), people=P, people_vel=Vp, goal=self.goal.copy(), t=self.t)

    def step(self, v_cmd):
        """Velocity command (m/s), limited to V_MAX and to A_MAX acceleration: the base needs time to stop."""
        v_cmd = np.asarray(v_cmd, float)
        n = np.linalg.norm(v_cmd)
        if n > V_MAX:
            v_cmd = v_cmd / n * V_MAX
        dv = v_cmd - self.v
        nd = np.linalg.norm(dv)
        if nd > A_MAX * DT:
            dv = dv / nd * A_MAX * DT
        self.v = self.v + dv
        target = self.data.qpos[:2] + self.v * DT
        for k in range(self.n_sub):
            self.data.ctrl[:2] = target
            mujoco.mj_step(self.model, self.data)
        self.t += DT
        self._move_people()
        mujoco.mj_forward(self.model, self.data)
        o = self.obs()
        clear = np.min(np.linalg.norm(o["people"] - o["pos"], axis=1)) - R_BASE - R_PERSON
        self.min_clear = min(self.min_clear, clear)
        self.collisions += int(clear < 0)
        return o, bool(np.linalg.norm(o["pos"] - self.goal) < 0.15 and np.linalg.norm(self.v) < 0.2)


class ReactiveController:
    """Classic potential field: attraction = P(D) control toward the goal, repulsion from people within `d0`,
    using where they are NOW. No prediction."""

    def __init__(self, kp=1.2, k_rep=0.6, d0=0.9):
        self.kp, self.k_rep, self.d0 = kp, k_rep, d0

    def reset(self):
        pass

    def __call__(self, o):
        v = self.kp * (o["goal"] - o["pos"])
        for p in o["people"]:
            r = o["pos"] - p
            d = np.linalg.norm(r) - R_BASE - R_PERSON
            if d < self.d0:
                v += self.k_rep * (1 / max(d, 0.05) - 1 / self.d0) * r / (np.linalg.norm(r) + 1e-9)
        return v, None


def constant_velocity(o, times):
    """Predict each person keeps walking straight at their current velocity (what a simple tracker gives you)."""
    return o["people"][None] + o["people_vel"][None] * (np.asarray(times) - o["t"])[:, None, None]


def oracle(room):
    """The TRUE future motion of the people (an upper bound: no real robot has this)."""
    return lambda o, times: np.stack([room.people_at(t) for t in times])


class PredictiveSampling:
    """Sampling MPC. The plan is a sequence of H velocity commands (piecewise-linear between K knots).
    Each control step: candidates = current plan + Gaussian noise (and 'stop' + the unperturbed plan); roll each out
    with the base model (velocity/acceleration limits) while the people move with their CURRENT velocity (constant-
    velocity prediction); cost = distance to goal + big penalty inside a safety margin + effort; keep the best."""

    def __init__(self, n_samples=128, horizon=20, knots=5, sigma=0.5, margin=0.25, seed=0, w_goal=1.0, w_coll=200.0, w_u=0.02,
                 predictor=None):
        """predictor(o, times) -> (len(times), n_people, 2) predicted positions. Default: constant velocity."""
        self.N, self.H, self.K, self.sigma, self.margin = n_samples, horizon, knots, sigma, margin
        self.predictor = predictor or constant_velocity
        self.w_goal, self.w_coll, self.w_u = w_goal, w_coll, w_u
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.plan = np.zeros((self.K, 2))

    def _expand(self, knots):                                       # (N, K, 2) knots -> (N, H, 2) commands
        tk = np.linspace(0, self.H - 1, self.K)
        t = np.arange(self.H)
        out = np.empty(knots.shape[:-2] + (self.H, 2))
        for j in range(2):
            out[..., j] = np.apply_along_axis(lambda k: np.interp(t, tk, k), -1, knots[..., j])
        return out

    def rollout(self, o, cmds):
        """Simulate the base for every candidate: (N, H, 2) commands -> (N, H, 2) positions. Vectorized."""
        N = cmds.shape[0]
        p, v = np.repeat(o["pos"][None], N, 0), np.repeat(o["vel"][None], N, 0)
        P = np.empty((N, self.H, 2))
        for h in range(self.H):
            c = cmds[:, h]
            n = np.linalg.norm(c, axis=1, keepdims=True)
            c = np.where(n > V_MAX, c / np.maximum(n, 1e-9) * V_MAX, c)
            dv = c - v
            nd = np.linalg.norm(dv, axis=1, keepdims=True)
            v = v + np.where(nd > A_MAX * DT, dv / np.maximum(nd, 1e-9) * A_MAX * DT, dv)
            p = p + v * DT
            P[:, h] = p
        return P

    def cost(self, o, P, cmds):
        people = self.predictor(o, o["t"] + (np.arange(self.H) + 1) * DT)                         # (H, n, 2)
        d = np.linalg.norm(P[:, :, None, :] - people[None], axis=-1) - R_BASE - R_PERSON           # (N, H, n)
        coll = np.sum(np.maximum(0, self.margin - d) ** 2, axis=(1, 2))
        goal = np.sum(np.linalg.norm(P - o["goal"], axis=-1), axis=1) * DT + 2.0 * np.linalg.norm(P[:, -1] - o["goal"], axis=-1)
        return self.w_goal * goal + self.w_coll * coll + self.w_u * np.sum(cmds ** 2, axis=(1, 2)) * DT

    def __call__(self, o):
        shift = np.r_[self.plan[1:], self.plan[-1:]]                   # warm start: the old plan, one knot later
        to_goal = o["goal"] - o["pos"]
        greedy = np.tile(to_goal / max(np.linalg.norm(to_goal), 1e-6) * V_MAX, (self.K, 1))
        noise = self.rng.normal(0, self.sigma, (self.N - 3, self.K, 2))
        knots = np.concatenate([shift[None], np.zeros((1, self.K, 2)), greedy[None], shift[None] + noise], 0)
        cmds = self._expand(knots)
        P = self.rollout(o, cmds)
        J = self.cost(o, P, cmds)
        best = int(np.argmin(J))
        self.plan = knots[best]
        return cmds[best, 0], dict(paths=P, best=best, cost=J)


def run(room, ctrl, seed, max_t=25.0, on_step=None):
    o = room.reset(seed); ctrl.reset(); done, t = False, 0.0
    while not done and room.t < max_t:
        v, info = ctrl(o)
        o, done = room.step(v)
        if on_step is not None:
            on_step(room, o, info)
    return dict(reached=done, time=room.t, collisions=room.collisions, min_clear=room.min_clear)


class Film:
    """1280x720 3/4 view of the room, with the MPC's sampled futures (thin, faint) and chosen plan (thick, green)."""

    def __init__(self, room, size=(720, 1280), azimuth=-60, elevation=-38, distance=7.5):
        self.room, self.r = room, mujoco.Renderer(room.model, *size)
        self.cam = mujoco.MjvCamera()
        self.cam.lookat[:] = [0, 0, 0.3]
        self.cam.azimuth, self.cam.elevation, self.cam.distance = azimuth, elevation, distance

    def _line(self, sc, a, b, width, rgba):
        if sc.ngeom >= sc.maxgeom:
            return
        g = sc.geoms[sc.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.asarray(rgba, np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width, a, b)
        sc.ngeom += 1

    def frames(self, info=None, sub=3, n_show=24):
        """`sub` frames interpolated between the previous and the current control step (30 fps from a 10 Hz loop)."""
        m, d = self.room.model, self.room.data
        if not hasattr(self, "scratch"):
            self.scratch, self.prev = mujoco.MjData(m), (d.qpos.copy(), d.mocap_pos.copy())
        out = []
        for i in range(1, sub + 1):
            a = i / sub
            self.scratch.qpos[:] = (1 - a) * self.prev[0] + a * d.qpos
            self.scratch.mocap_pos[:] = (1 - a) * self.prev[1] + a * d.mocap_pos
            self.scratch.mocap_quat[:] = d.mocap_quat
            mujoco.mj_forward(m, self.scratch)
            out.append(self._render(self.scratch, info, n_show))
        self.prev = (d.qpos.copy(), d.mocap_pos.copy())
        return out

    def frame(self, info=None, n_show=24):
        return self._render(self.room.data, info, n_show)

    def _render(self, data, info, n_show):
        self.r.update_scene(data, self.cam)
        sc = self.r.scene
        if info is not None and info.get("paths") is not None:
            P, best = info["paths"], info["best"]
            idx = np.r_[best, np.linspace(0, len(P) - 1, n_show).astype(int)]
            for j, i in enumerate(idx):
                pts = np.c_[P[i], np.full(len(P[i]), 0.02)]
                for k in range(1, len(pts)):
                    if j == 0:
                        self._line(sc, pts[k - 1], pts[k], 0.025, [0.2, 1.0, 0.4, 1.0])
                    else:
                        self._line(sc, pts[k - 1], pts[k], 0.008, [0.6, 0.8, 1.0, 0.35])
        return self.r.render().copy()
