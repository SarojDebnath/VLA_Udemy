"""S04 project 42d: a Unitree Go2 trotting with a classical controller (no learning).

    leg_ik(p, side)          analytic inverse kinematics of one 3-DoF leg (hip roll, thigh, calf)
    Go2Trot                  gait scheduler (trot: diagonal pairs) + swing/stance foot trajectories
                             + Raibert foot placement (step where you are falling) + body levelling
                             + joint PD at 500 Hz -> motor torques
    Go2World                 the Go2 (Menagerie) with optional pushes, and a follow camera for HD video
"""
import mujoco
import mujoco_menagerie as mm
import numpy as np

L1, L2, L3 = 0.0955, 0.213, 0.213            # hip offset, thigh, calf (m)
HIPS = {"FL": (0.1934, 0.0465), "FR": (0.1934, -0.0465), "RL": (-0.1934, 0.0465), "RR": (-0.1934, -0.0465)}
SIDE = {"FL": 1, "FR": -1, "RL": 1, "RR": -1}
LEGS = ["FL", "FR", "RL", "RR"]
PAIR = {"FL": 0, "RR": 0, "FR": 1, "RL": 1}  # trot: diagonal pairs swing together
FOOT_R = 0.022


def leg_ik(p, s):
    """Foot position p = (x, y, z) relative to the hip-roll joint (trunk frame) -> (hip, thigh, calf) angles.
    s = +1 left leg, -1 right leg. Derived for this leg's axes: roll about x, then thigh and calf about y."""
    x, y, z = p
    r2 = max(y * y + z * z - L1 * L1, 1e-6)
    r = np.sqrt(r2)
    q1 = np.arctan2(z, y) - np.arctan2(-r, s * L1)
    D = np.clip((x * x + r2 - L2 * L2 - L3 * L3) / (2 * L2 * L3), -1, 1)
    q3 = -np.arccos(D)                                              # knee bends backwards (calf range is negative)
    q2 = np.arctan2(-x, r) - np.arctan2(L3 * np.sin(q3), L2 + L3 * np.cos(q3))
    return np.array([q1, q2, q3])


def quat_to_rpy(q):
    w, x, y, z = q
    return np.array([np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)), np.arcsin(np.clip(2 * (w * y - z * x), -1, 1)),
                     np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))])


class Go2World:
    def __init__(self):
        spec = mm.get("unitree_go2").spec("scene")
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
        spec.visual.quality.shadowsize = 4096
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        self.dt = self.m.opt.timestep
        self.reset()

    def reset(self):
        mujoco.mj_resetDataKeyframe(self.m, self.d, self.m.key("home").id)
        mujoco.mj_forward(self.m, self.d)
        self.t = 0.0

    def base(self):
        d = self.d
        R = d.xmat[self.m.body("base").id].reshape(3, 3)
        return dict(pos=d.qpos[:3].copy(), rpy=quat_to_rpy(d.qpos[3:7]), vel_body=R.T @ d.qvel[:3], omega=d.qvel[3:6].copy(), R=R)

    def fallen(self):
        b = self.base()
        return b["pos"][2] < 0.15 or abs(b["rpy"][0]) > 0.8 or abs(b["rpy"][1]) > 0.8

    def step(self, tau, push=None):
        self.d.ctrl[:] = np.clip(tau, self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1])
        self.d.xfrc_applied[self.m.body("base").id, :3] = 0 if push is None else push
        mujoco.mj_step(self.m, self.d)
        self.t += self.dt


class Go2Trot:
    """Classical trot controller.
    command: (vx, vy, yaw_rate) in the body frame. Each control tick:
      phase -> which diagonal pair is in swing;
      stance feet slide backwards under the body at the commanded velocity (that's what pushes the body forward);
      swing feet lift (sine profile) and land at the Raibert point:  p_land = p_hip + v T_st/2 + k (v - v_cmd);
      body levelling: lengthen/shorten legs to cancel roll and pitch;
      leg IK -> joint targets -> joint PD at the physics rate."""

    def __init__(self, period=0.4, height=0.27, step_height=0.07, kp=120.0, kd=2.0, k_raibert=0.2, k_level=0.0, raibert=True,
                 gravity_ff=True, k_vel=1.0):
        self.T, self.h, self.sh, self.kp, self.kd = period, height, step_height, kp, kd
        self.gravity_ff, self.k_vel = gravity_ff, k_vel
        self.stance = set()
        self.k_r, self.k_level, self.raibert = k_raibert, k_level, raibert
        self.cmd = np.zeros(3)
        self.dt_ctrl = 0.002
        self.swing_start = {leg: None for leg in LEGS}
        self.foot = {leg: np.array([HIPS[leg][0], HIPS[leg][1] + SIDE[leg] * L1, -self.h]) for leg in LEGS}

    def nominal(self, leg):
        return np.array([HIPS[leg][0], HIPS[leg][1] + SIDE[leg] * L1, -self.h])

    def targets(self, world):
        b, t = world.base(), world.t
        phase = (t % self.T) / self.T                                   # 0..1
        swing_pair = 0 if phase < 0.5 else 1
        s = (phase % 0.5) / 0.5                                         # progress through this half-cycle
        v = b["vel_body"]
        v_cmd = np.r_[self.cmd[:2] + self.k_vel * (self.cmd[:2] - v[:2]), 0.0]     # optional velocity feedback
        self.stance = {leg for leg in LEGS if PAIR[leg] != swing_pair}
        q_des = np.zeros(12)
        for i, leg in enumerate(LEGS):
            nom = self.nominal(leg)
            yaw_v = self.cmd[2] * np.array([-nom[1], nom[0], 0.0])      # foot velocity needed to turn
            if PAIR[leg] == swing_pair:                                 # swing: lift and move to the landing point
                land = nom + (v_cmd + yaw_v) * (self.T / 4)
                if self.raibert:
                    land[:2] += self.k_r * (v[:2] - v_cmd[:2])          # step towards where you are drifting
                if s < 1e-9 or self.swing_start[leg] is None:
                    self.swing_start[leg] = self.foot[leg].copy()
                a = 3 * s * s - 2 * s ** 3
                p = self.swing_start[leg] + (land - self.swing_start[leg]) * a
                p[2] = nom[2] + self.sh * np.sin(np.pi * s)
            else:                                                       # stance: slide back under the body
                self.swing_start[leg] = None
                p = self.foot[leg] - (v_cmd + yaw_v) * self.dt_ctrl
                p[2] = nom[2]
            # body levelling: a leg on the low side of a roll/pitch extends, on the high side shortens
            roll, pitch = b["rpy"][0], b["rpy"][1]
            p[2] += self.k_level * (np.sign(nom[1]) * np.sin(roll) * abs(nom[1]) - nom[0] * np.sin(pitch))
            self.foot[leg] = p
            q_des[3 * i:3 * i + 3] = leg_ik(p - np.r_[HIPS[leg], 0.0], SIDE[leg])
        return q_des

    def torques(self, world, q_des):
        q, qd = world.d.qpos[7:], world.d.qvel[6:]
        tau = self.kp * (q_des - q) - self.kd * qd
        if self.gravity_ff and self.stance:
            # stance feet share the weight: tau = J^T F with F = (0, 0, -m g / n_stance) pushed into the ground
            m, d = world.m, world.d
            weight = float(m.body_subtreemass[m.body("base").id]) * 9.81
            jac = np.zeros((3, m.nv))
            for i, leg in enumerate(LEGS):
                if leg in self.stance:
                    mujoco.mj_jac(m, d, jac, None, d.geom(leg).xpos, m.body(f"{leg}_calf").id)
                    J = jac[:, 6 + 3 * i: 9 + 3 * i]
                    tau[3 * i:3 * i + 3] += J.T @ np.array([0.0, 0.0, -weight / len(self.stance)])
        return tau

    def run(self, world, commands, push_fn=None, on_tick=None, ctrl_every=1):
        """commands(t) -> (vx, vy, yaw_rate). Runs until commands returns None or the robot falls."""
        q_des = world.d.qpos[7:].copy()
        self.dt_ctrl = world.dt * ctrl_every
        k = 0
        while True:
            c = commands(world.t)
            if c is None:
                return True
            self.cmd = np.asarray(c, float)
            if k % ctrl_every == 0:
                q_des = self.targets(world)
            world.step(self.torques(world, q_des), None if push_fn is None else push_fn(world.t))
            if on_tick is not None:
                on_tick(world, k)
            if world.fallen():
                return False
            k += 1


class Follow:
    def __init__(self, world, size=(720, 1280), distance=1.6, elevation=-18, azimuth_offset=135):
        self.w, self.r = world, mujoco.Renderer(world.m, *size)
        self.cam = mujoco.MjvCamera()
        self.cam.distance, self.cam.elevation, self.off = distance, elevation, azimuth_offset
        self.look, self.az = None, None

    def render(self, arrow=None):
        b = self.w.base()
        p = b["pos"] * [1, 1, 0] + [0, 0, 0.22]
        self.look = p if self.look is None else 0.92 * self.look + 0.08 * p
        az = np.degrees(b["rpy"][2]) + self.off
        self.az = az if self.az is None else self.az + 0.04 * ((az - self.az + 180) % 360 - 180)
        self.cam.lookat[:], self.cam.azimuth = self.look, self.az
        self.r.update_scene(self.w.d, self.cam)
        if arrow is not None:
            sc = self.r.scene
            g = sc.geoms[sc.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.zeros(9), np.array([1, 0.2, 0.2, 1], np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_ARROW, 0.02, b["pos"] - arrow * 0.45, b["pos"] - arrow * 0.08)
            sc.ngeom += 1
        return self.r.render().copy()
