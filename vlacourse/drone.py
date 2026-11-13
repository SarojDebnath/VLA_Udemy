"""S04 project: a quadrotor (Skydio X2 from MuJoCo Menagerie) flown with cascaded PID control.

    DroneRace(gates=...)      the X2 in a hall with square racing gates, wind gusts (xfrc_applied) and a payload option
    mixer(T, tau)             total thrust + body torques -> 4 motor thrusts (the "allocation matrix")
    PID                       a textbook PID with output limits and anti-windup
    CascadedController        position PID -> desired tilt + thrust -> attitude PD -> torques -> mixer
    Film                      a 1280x720 chase camera with a fading flight trail, gate highlights and captions
"""
import mujoco
import mujoco_menagerie as mm
import numpy as np

G = 9.81


def quat_to_rpy(q):
    w, x, y, z = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.array([roll, pitch, yaw])


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class DroneRace:
    """Skydio X2 + square gates. State: position, velocity, roll/pitch/yaw, body rates. Control: 4 motor thrusts (N)."""

    def __init__(self, gates=(), gate_size=0.9, payload=0.0, seed=0):
        spec = mm.get("skydio_x2").spec("scene")
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
        spec.visual.quality.shadowsize = 4096
        self.gates = [np.asarray(g, float) for g in gates]               # (x, y, z, yaw)
        for i, (x, y, z, yaw) in enumerate(self.gates):
            b = spec.worldbody.add_body(name=f"gate{i}", pos=[x, y, z], quat=[np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            h, t = gate_size / 2, 0.04                                     # a square frame in the gate's y-z plane
            bars = [([0.03, h + t, t], [0, 0, h]), ([0.03, h + t, t], [0, 0, -h]), ([0.03, t, h], [0, h, 0]), ([0.03, t, h], [0, -h, 0])]
            for k, (size, pos) in enumerate(bars):
                b.add_geom(name=f"gate{i}_{k}", type=mujoco.mjtGeom.mjGEOM_BOX, size=size, pos=pos, rgba=[1.0, 0.45, 0.1, 1],
                           contype=0, conaffinity=0)
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[0.025, (z - h) / 2], pos=[0, 0, -h - (z - h) / 2], rgba=[0.3, 0.3, 0.35, 1],
                       contype=0, conaffinity=0)
        if payload > 0:                                                     # an unknown extra mass (the integral term's job)
            spec.body("x2").add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.05, 0.05, 0.03], pos=[0, 0, -0.06], mass=payload,
                                     rgba=[0.2, 0.6, 0.9, 1], contype=0, conaffinity=0)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.mass = float(self.model.body_subtreemass[self.model.body("x2").id]) - payload   # the NOMINAL mass (spec sheet)
        self.payload = payload
        self.dt = self.model.opt.timestep
        m = self.model
        self.motor_xy = np.array([m.site_pos[m.actuator_trnid[i, 0]][:2] for i in range(4)])
        self.motor_yaw = np.array([m.actuator_gear[i, 5] for i in range(4)])
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self, pos=(0, 0, 0.3), yaw=0.0):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:3] = pos
        self.data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)
        self.t, self.wind = 0.0, np.zeros(3)
        return self.state()

    def state(self):
        d = self.data
        R = d.xmat[self.model.body("x2").id].reshape(3, 3)
        return dict(pos=d.qpos[:3].copy(), vel=d.qvel[:3].copy(), rpy=quat_to_rpy(d.qpos[3:7]), omega=d.qvel[3:6].copy(), R=R.copy())

    def mixer(self, T, tau):
        """[T, tau_x, tau_y, tau_z] = A @ f  ->  f = A^-1 [...]: the allocation matrix of this X-configuration."""
        x, y = self.motor_xy[:, 0], self.motor_xy[:, 1]
        A = np.vstack([np.ones(4), y, -x, self.motor_yaw])
        f = np.linalg.solve(A, np.r_[T, tau])
        return np.clip(f, 0, 13.0)

    def step(self, f, wind=None):
        d = self.data
        d.ctrl[:] = f
        d.xfrc_applied[self.model.body("x2").id, :3] = self.wind if wind is None else wind
        mujoco.mj_step(self.model, d)
        self.t += self.dt
        return self.state()


class PID:
    """kp*e + ki*integral(e) + kd*de/dt, with two anti-windup measures: the integrator is clamped (i_limit) and only
    accumulates while the error is small (i_zone, "conditional integration"): during a big step it would only wind up."""

    def __init__(self, kp, ki=0.0, kd=0.0, i_limit=np.inf, out_limit=np.inf, i_zone=np.inf):
        self.kp, self.ki, self.kd, self.i_limit, self.out_limit, self.i_zone = kp, ki, kd, i_limit, out_limit, i_zone
        self.i = 0.0

    def reset(self):
        self.i = 0.0

    def __call__(self, err, derr, dt):
        if abs(err) < self.i_zone:
            self.i = np.clip(self.i + err * dt, -self.i_limit, self.i_limit)
        return np.clip(self.kp * err + self.ki * self.i + self.kd * derr, -self.out_limit, self.out_limit)


class CascadedController:
    """Outer loop (slow): position error -> desired acceleration (PID per axis, + feed-forward).
    Middle: desired acceleration -> total thrust + desired roll/pitch (rotate into the yaw frame, small-angle tilt).
    Inner loop (fast): attitude PD on roll/pitch/yaw -> body torques -> mixer -> motor thrusts."""

    def __init__(self, drone, pos_gains=(4.0, 1.5, 3.2), z_gains=(8.0, 4.0, 5.0), att_gains=(6.0, 0.6), yaw_gains=(1.5, 0.3),
                 max_tilt=0.5, use_integral=True):
        kp, ki, kd = pos_gains
        zp, zi, zd = z_gains
        ki, zi = (ki, zi) if use_integral else (0.0, 0.0)
        self.drone, self.max_tilt = drone, max_tilt
        self.pid = [PID(kp, ki, kd, i_limit=2.0, out_limit=8.0, i_zone=0.3), PID(kp, ki, kd, i_limit=2.0, out_limit=8.0, i_zone=0.3),
                    PID(zp, zi, zd, i_limit=3.0, out_limit=15.0, i_zone=0.3)]
        self.att, self.yaw = att_gains, yaw_gains
        self.mass = drone.mass                       # the controller's belief (does NOT include an unknown payload)

    def reset(self):
        for p in self.pid:
            p.reset()

    def __call__(self, s, target, vel_ff=np.zeros(3), acc_ff=np.zeros(3), yaw_target=0.0):
        dt = self.drone.dt
        e, de = target - s["pos"], vel_ff - s["vel"]
        acc = np.array([self.pid[i](e[i], de[i], dt) for i in range(3)]) + acc_ff
        acc[2] += G
        T = self.mass * acc[2] / max(0.3, np.cos(s["rpy"][0]) * np.cos(s["rpy"][1]))
        yaw = s["rpy"][2]
        ax = np.cos(yaw) * acc[0] + np.sin(yaw) * acc[1]                   # into the yaw-aligned frame
        ay = -np.sin(yaw) * acc[0] + np.cos(yaw) * acc[1]
        pitch_d = np.clip(np.arctan2(ax, acc[2]), -self.max_tilt, self.max_tilt)
        roll_d = np.clip(-np.arctan2(ay, acc[2]), -self.max_tilt, self.max_tilt)
        kp, kd = self.att
        tau = np.array([kp * (roll_d - s["rpy"][0]) - kd * s["omega"][0],
                        kp * (pitch_d - s["rpy"][1]) - kd * s["omega"][1],
                        self.yaw[0] * wrap(yaw_target - yaw) - self.yaw[1] * s["omega"][2]])
        return self.drone.mixer(T, tau), dict(roll_d=roll_d, pitch_d=pitch_d, T=T)


def catmull_rom(points, n_per=200, closed=False):
    """A smooth path through the gate centres (C1 continuous): returns positions and unit tangents."""
    P = [np.asarray(p, float) for p in points]
    if closed:
        P = [P[-1]] + P + [P[0], P[1]]
    else:
        P = [2 * P[0] - P[1]] + P + [2 * P[-1] - P[-2]]
    pts = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for t in np.linspace(0, 1, n_per, endpoint=False):
            pts.append(0.5 * ((2 * p1) + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2 + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    pts.append(P[-2])
    pts = np.array(pts)
    seg = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return pts, seg


class PathFollower:
    """A carrot that moves along the path at `speed` (m/s); gives target position, velocity and acceleration feed-forward."""

    def __init__(self, pts, seg, speed):
        self.pts, self.seg, self.speed = pts, seg, speed

    def __call__(self, t):
        s = min(self.speed * t, self.seg[-1])
        i = min(np.searchsorted(self.seg, s), len(self.pts) - 1)
        j = min(i + 1, len(self.pts) - 1)
        tang = self.pts[j] - self.pts[max(i - 1, 0)]
        tang = tang / (np.linalg.norm(tang) + 1e-9)
        k = min(i + 8, len(self.pts) - 1)
        tang2 = self.pts[min(k + 1, len(self.pts) - 1)] - self.pts[max(k - 1, 0)]
        tang2 = tang2 / (np.linalg.norm(tang2) + 1e-9)
        ds = max(self.seg[k] - self.seg[i], 1e-6)
        acc = (tang2 - tang) * self.speed ** 2 / ds                        # centripetal feed-forward
        moving = s < self.seg[-1]
        return self.pts[i], tang * self.speed * moving, acc * moving, s >= self.seg[-1]


def gate_passed(prev, cur, gate, size):
    """Did the segment prev->cur cross the gate's plane inside its square opening?"""
    x, y, z, yaw = gate
    n = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    c = np.array([x, y, z])
    a, b = np.dot(prev - c, n), np.dot(cur - c, n)
    if a * b > 0 or a == b:
        return False
    p = prev + (cur - prev) * (a / (a - b))
    lateral = np.array([-np.sin(yaw), np.cos(yaw), 0.0])
    return abs(np.dot(p - c, lateral)) < size / 2 and abs(p[2] - z) < size / 2


class Film:
    """1280x720 chase camera: follows the drone from behind-above, draws a fading trail, highlights passed gates."""

    def __init__(self, race, size=(720, 1280), distance=2.6, elevation=-18, trail=160):
        self.race, self.r = race, mujoco.Renderer(race.model, *size)
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.elevation = distance, elevation
        self.trail, self.n_trail, self.size = [], trail, size
        self.az = None

    def frame(self, passed=()):
        d, m = self.race.data, self.race.model
        p = d.qpos[:3].copy()
        v = d.qvel[:3]
        heading = np.degrees(np.arctan2(v[1], v[0])) if np.linalg.norm(v[:2]) > 0.3 else (self.az if self.az is not None else 0.0)
        self.az = heading if self.az is None else self.az + 0.06 * ((heading - self.az + 180) % 360 - 180)   # smooth turn
        self.cam.lookat[:] = 0.85 * (self.cam.lookat if np.any(self.cam.lookat) else p) + 0.15 * p
        self.cam.azimuth = self.az
        for g in range(len(self.race.gates)):
            col = [0.2, 0.9, 0.4, 1] if g in passed else [1.0, 0.45, 0.1, 1]
            for k in range(4):
                m.geom_rgba[m.geom(f"gate{g}_{k}").id] = col
        self.r.update_scene(d, self.cam)
        sc = self.r.scene
        self.trail.append(p)
        self.trail = self.trail[-self.n_trail:]
        for i in range(1, len(self.trail)):
            if sc.ngeom >= sc.maxgeom:
                break
            a = i / len(self.trail)
            g = sc.geoms[sc.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3), np.zeros(9), np.array([0.3, 0.8, 1.0, 0.8 * a], np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.012, self.trail[i - 1], self.trail[i])
            sc.ngeom += 1
        return self.r.render().copy()
