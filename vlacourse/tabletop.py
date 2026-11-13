"""TableTop: the course's running MuJoCo world (Franka Panda + colored cubes + trays).

Action (7-D, end-effector delta, like OpenVLA/LIBERO):
    [dx, dy, dz, droll, dpitch, dyaw, grip]   meters / radians per control step, grip in [0 open, 1 closed]
Observation dict:
    images: {camera_name: HxWx3 uint8}, state: ee_pos(3) + ee_quat(4) + gripper_width(1), qpos: 7 arm joints
"""
import mujoco
import mujoco_menagerie as mm
import numpy as np

from .panda import DOWN, GRIP_CLOSED, GRIP_OPEN, add_ee_site, ee_pose, solve_ik, yaw_quat

COLORS = {
    "red": [0.90, 0.15, 0.15, 1],
    "green": [0.20, 0.75, 0.25, 1],
    "blue": [0.20, 0.35, 0.90, 1],
    "yellow": [0.95, 0.85, 0.15, 1],
    "purple": [0.60, 0.30, 0.80, 1],
    "orange": [1.00, 0.55, 0.10, 1],
}
CUBE_HALF = 0.02
TRAY_HALF = 0.06
WORKSPACE = np.array([[0.35, 0.65], [-0.25, 0.25]])
TASK_TYPES = ("pick", "place", "stack")
MAX_DPOS, MAX_DROT = 0.05, 0.3                         # per-step limits of the 7-D EE-delta action
TARGET_LO, TARGET_HI = np.array([0.2, -0.45, 0.0]), np.array([0.8, 0.45, 0.6])   # commanded-target box

# Named scenes, from easy to hard. Pixel policies are taught on the easier tiers, where they demonstrably work;
# "full" is the hard benchmark (3 cubes + tray, pick/place/stack) that exposes their limits.
SCENES = {
    "single": dict(n_cubes=1, n_trays=1, task_types=("pick", "place")),   # one cube + one tray
    "colors": dict(n_cubes=2, n_trays=0, task_types=("pick",)),            # two cubes: the instruction picks the colour
    "sort": dict(n_cubes=2, n_trays=1, task_types=("pick", "place")),     # two cubes + tray: multi-step sorting (S11, C1)
    "full": dict(n_cubes=3, n_trays=1, task_types=TASK_TYPES),
}


def _panda_spec():
    spec = mm.get("franka_emika_panda").spec("scene")
    add_ee_site(spec)
    spec.body("hand").add_camera(name="wrist", pos=[0, -0.09, 0.0], quat=[0, 1, 0, 0], fovy=75)
    return spec


def _ur5e_spec():
    arm = mm.get("universal_robots_ur5e").spec("scene")
    gripper = mm.get("robotiq_2f85").spec("2f85")
    arm.option.cone, arm.option.impratio = gripper.option.cone, gripper.option.impratio
    arm.site("attachment_site").attach_body(gripper.worldbody.first_body(), "gripper_", "")
    arm.body("gripper_base").add_camera(name="wrist", pos=[0, -0.1, 0.02], quat=[0, 1, 0, 0], fovy=75)
    for b in arm.bodies:
        if b.name not in ("world", ""):
            b.gravcomp = 1.0
    return arm


ROBOTS = {
    "panda": dict(build=_panda_spec, n_arm=7, ee="ee", grip_open=GRIP_OPEN, grip_closed=GRIP_CLOSED,
                  width=lambda d: float(d.joint("finger_joint1").qpos[0] + d.joint("finger_joint2").qpos[0])),
    "ur5e": dict(build=_ur5e_spec, n_arm=6, ee="gripper_pinch", grip_open=0.0, grip_closed=255.0, start_pos=(0.45, 0.0, 0.4),
                 ready_q=(np.pi, -1.5708, 1.5708, -1.5708, -1.5708, 0.0),
                 width=lambda d: float(0.085 * (1 - d.joint("gripper_right_driver_joint").qpos[0] / 0.8))),
}


def lookat_xyaxes(pos, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    x = np.cross(f, up)
    x /= np.linalg.norm(x)
    y = np.cross(x, f)
    return np.concatenate([x, y]).tolist()


def yaw_of(quat):
    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, quat)
    R = R.reshape(3, 3)
    return float(np.arctan2(R[1, 0], R[0, 0]))


def rotvec_quat(rv):
    angle = np.linalg.norm(rv)
    q = np.array([1.0, 0, 0, 0])
    if angle > 1e-12:
        mujoco.mju_axisAngle2Quat(q, rv / angle, angle)
    return q


class TableTop:
    def __init__(self, n_cubes=3, n_trays=1, control_hz=10, image_size=(224, 224),
                 cameras=("front", "wrist"), seed=0, max_steps=200, task_types=TASK_TYPES, robot="panda",
                 workspace=None, visual_randomization=False, wrist_ft=False, scene=None):
        """scene: one of SCENES ('single', 'colors', 'full'); sets n_cubes, n_trays and task_types.
        workspace: [[xmin, xmax], [ymin, ymax]] for object placement (default WORKSPACE).
        visual_randomization: per-episode random front-camera pose, lighting and table color (visual domain randomization).
        wrist_ft: add a force/torque sensor at the wrist (used by safety shields in S15)."""
        if scene is not None:
            n_cubes, n_trays, task_types = (SCENES[scene][k] for k in ("n_cubes", "n_trays", "task_types"))
        self.scene = scene
        self.robot = robot
        self.workspace = np.array(workspace if workspace is not None else WORKSPACE, float)
        self.visual_randomization, self.wrist_ft = visual_randomization, wrist_ft
        self.cfg = ROBOTS[robot]
        self.n_arm, self.ee = self.cfg["n_arm"], self.cfg["ee"]
        assert n_cubes + n_trays <= len(COLORS)
        self.n_cubes, self.n_trays = n_cubes, n_trays
        self.cameras = tuple(cameras)
        self.task_types = tuple(task_types)
        self.max_steps = max_steps
        self.max_dpos, self.max_drot = MAX_DPOS, MAX_DROT
        self.spec = self.build_spec()
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.scratch = mujoco.MjData(self.model)
        self.n_substeps = int(round(1.0 / (control_hz * self.model.opt.timestep)))
        self.control_hz = control_hz
        self.h, self.w = image_size
        self.renderer = mujoco.Renderer(self.model, self.h, self.w) if self.cameras else None
        self._vid = None
        self.rng = np.random.default_rng(seed)
        self.reset(seed=seed)

    # ---------- model ----------
    def build_spec(self):
        spec = self.cfg["build"]()
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 1280, 960
        wb = spec.worldbody
        spec.geom("floor").pos = [0, 0, -0.75]
        wb.add_geom(name="table", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.5, 0.65, 0.375],
                    pos=[0.4, 0, -0.375], rgba=[0.55, 0.42, 0.3, 1])
        names = list(COLORS)
        self.cube_colors = names[: self.n_cubes]
        self.tray_colors = names[self.n_cubes: self.n_cubes + self.n_trays]
        for c in self.cube_colors:
            b = wb.add_body(name=f"cube_{c}", pos=[0.5, 0, CUBE_HALF])
            b.add_freejoint(name=f"cube_{c}")
            b.add_geom(name=f"cube_{c}", type=mujoco.mjtGeom.mjGEOM_BOX, size=[CUBE_HALF] * 3,
                       rgba=COLORS[c], mass=0.05, friction=[1.5, 0.01, 0.001])
        for c in self.tray_colors:
            b = wb.add_body(name=f"tray_{c}", pos=[0.5, 0.2, 0])
            t, h = 0.005, 0.02
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[TRAY_HALF, TRAY_HALF, t / 2], pos=[0, 0, t / 2], rgba=COLORS[c])
            for sx, sy, w, l in [(1, 0, t, TRAY_HALF), (-1, 0, t, TRAY_HALF), (0, 1, TRAY_HALF, t), (0, -1, TRAY_HALF, t)]:
                b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[w, l, h / 2],
                           pos=[sx * (TRAY_HALF - t), sy * (TRAY_HALF - t), h / 2], rgba=COLORS[c])
        wb.add_camera(name="front", pos=[1.0, 0, 0.9], xyaxes=lookat_xyaxes([1.0, 0, 0.9], [0.42, 0, 0.0]))
        wb.add_camera(name="top", pos=[0.5, 0, 1.3], xyaxes=[0, -1, 0, 1, 0, 0])
        wb.add_camera(name="side", pos=[0.5, -0.85, 0.6], xyaxes=lookat_xyaxes([0.5, -0.85, 0.6], [0.5, 0, 0.05]))
        if self.wrist_ft:
            body = "hand" if self.robot == "panda" else "gripper_base"
            spec.body(body).add_site(name="ft", size=[0.01] * 3)
            spec.add_sensor(name="wrist_force", type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="ft")
        return spec

    # ---------- episode ----------
    def sample_layout(self):
        items = [("tray", c, TRAY_HALF + 0.03) for c in self.tray_colors] + [("cube", c, 0.04) for c in self.cube_colors]
        placed = []
        for kind, c, r in items:
            for _ in range(1000):
                xy = self.rng.uniform(self.workspace[:, 0], self.workspace[:, 1])
                if all(np.linalg.norm(xy - p) > r + pr for _, _, p, pr in placed):
                    break
            placed.append((kind, c, xy, r))
        return placed

    def reset(self, seed=None, task=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        m, d = self.model, self.data
        mujoco.mj_resetDataKeyframe(m, d, m.key("home").id)
        for kind, c, xy, _ in self.sample_layout():
            if kind == "tray":
                m.body_pos[m.body(f"tray_{c}").id][:2] = xy
            else:
                adr = m.jnt_qposadr[m.joint(f"cube_{c}").id]
                yaw = self.rng.uniform(-np.pi / 4, np.pi / 4)
                d.qpos[adr: adr + 7] = [xy[0], xy[1], CUBE_HALF, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        if "ready_q" in self.cfg:
            d.qpos[:self.n_arm] = self.cfg["ready_q"]
        if self.visual_randomization:
            self.randomize_visuals()
        mujoco.mj_forward(m, d)
        pos, quat = ee_pose(m, d, self.ee)
        start = self.cfg.get("start_pos")
        self.target_pos = np.array(start, float) if start is not None else pos.copy()
        self.target_quat = yaw_quat(yaw_of(quat))
        n = self.n_arm
        self.q_cmd = d.qpos[:n].copy()
        self.grip = 0.0
        q, _ = solve_ik(m, d, self.target_pos, self.target_quat, self.q_cmd, site=self.ee, n_joints=n, iters=500, scratch=self.scratch)
        d.qpos[:n] = q
        d.ctrl[:n], d.ctrl[n] = q, self.cfg["grip_open"]
        self.q_cmd = q
        mujoco.mj_forward(m, d)
        self.t = 0
        self.unreachable = False
        self.task = task if task is not None else self.sample_task()
        return self.observe()

    def randomize_visuals(self):
        m = self.model
        if not hasattr(self, "_vis0"):
            cid, gid = m.camera("front").id, m.geom("table").id
            self._vis0 = (m.cam_pos[cid].copy(), m.cam_fovy[cid].copy(), m.light_diffuse.copy(), m.geom_rgba[gid].copy())
        cam_pos, fovy, light, table = self._vis0
        cid, gid = m.camera("front").id, m.geom("table").id
        m.cam_pos[cid] = cam_pos + self.rng.uniform(-0.05, 0.05, 3)
        m.cam_fovy[cid] = fovy + self.rng.uniform(-4, 4)
        m.light_diffuse[:] = np.clip(light * self.rng.uniform(0.4, 1.6), 0, 1)
        a, g = self.rng.uniform(0, 0.7), self.rng.uniform(0.3, 0.8)
        m.geom_rgba[gid][:3] = (1 - a) * table[:3] * self.rng.uniform(0.7, 1.15) + a * g

    def wrist_force(self):
        return self.data.sensor("wrist_force").data.copy() if self.wrist_ft else None

    def sample_task(self):
        kind = self.rng.choice(self.task_types)
        if kind == "place" and self.n_trays == 0:
            kind = "pick"
        if kind == "stack" and self.n_cubes < 2:
            kind = "pick"
        cubes = list(self.rng.permutation(self.cube_colors))
        if kind == "pick":
            return dict(type="pick", obj=cubes[0], instruction=f"pick up the {cubes[0]} cube")
        if kind == "place":
            tray = self.rng.choice(self.tray_colors)
            return dict(type="place", obj=cubes[0], target=tray, instruction=f"put the {cubes[0]} cube in the {tray} tray")
        return dict(type="stack", obj=cubes[0], target=cubes[1], instruction=f"stack the {cubes[0]} cube on the {cubes[1]} cube")

    # ---------- control ----------
    def step(self, action):
        a = np.asarray(action, dtype=float)
        dpos = np.clip(a[:3], -self.max_dpos, self.max_dpos)
        drot = np.clip(a[3:6], -self.max_drot, self.max_drot)
        self.grip = float(np.clip(a[6], 0, 1))
        prev_pos, prev_quat = self.target_pos, self.target_quat
        self.target_pos = np.clip(self.target_pos + dpos, TARGET_LO, TARGET_HI)
        q = np.zeros(4)
        mujoco.mju_mulQuat(q, rotvec_quat(drot), self.target_quat)
        mujoco.mju_normalize4(q)
        self.target_quat = q
        q_new, res = solve_ik(self.model, self.data, self.target_pos, self.target_quat, self.q_cmd,
                              site=self.ee, n_joints=self.n_arm, iters=50, scratch=self.scratch)
        self.unreachable = res > 5e-3
        if self.unreachable:
            self.target_pos, self.target_quat = prev_pos, prev_quat
            q_new = self.q_cmd
        q_old = self.q_cmd
        for i in range(self.n_substeps):
            s = (i + 1) / self.n_substeps
            self.data.ctrl[:self.n_arm] = q_old + (q_new - q_old) * s
            self.data.ctrl[self.n_arm] = self.cfg["grip_open"] + (self.cfg["grip_closed"] - self.cfg["grip_open"]) * self.grip
            mujoco.mj_step(self.model, self.data)
            if self._vid is not None:
                self._vid["k"] += 1
                if self._vid["k"] % self._vid["every"] == 0:
                    self._vid["frames"].append(self._render_vid())
        self.q_cmd = q_new
        self.t += 1
        success = self.success()
        done = success or self.t >= self.max_steps
        return self.observe(), float(success), done, {"success": success}

    # ---------- HD video recording (for humans, independent of the policy's low-res cameras) ----------
    def start_video(self, camera="front", size=(360, 480), fps=30, overlay=None):
        """Record smooth HD frames during the physics substeps (control is 10 Hz, video is `fps`)."""
        if self._vid is not None:
            self.stop_video()
        if camera == "showcase":                     # cinematic 3/4 view of the workspace (not a policy camera)
            cam = mujoco.MjvCamera()
            cam.lookat[:] = [0.5, 0.0, 0.08]; cam.distance, cam.azimuth, cam.elevation = 1.15, 145, -28
            camera = cam
        self._vid = dict(r=mujoco.Renderer(self.model, *size), cam=camera, k=0, frames=[], overlay=overlay,
                         every=max(1, int(round(1.0 / (fps * self.model.opt.timestep)))))
        self._vid["frames"].append(self._render_vid())

    def _render_vid(self):
        self._vid["r"].update_scene(self.data, self._vid["cam"])
        img = self._vid["r"].render()
        if self._vid["overlay"] is not None:            # live readout drawn on every frame, e.g. wrist force
            from .showcase import badge
            text, color = self._vid["overlay"](self)
            img = badge(img, text, "br", color=color)
        return img

    def stop_video(self):
        if self._vid is None:
            return []
        frames = self._vid["frames"]; self._vid["r"].close(); self._vid = None
        return frames

    # ---------- sensing ----------
    def render(self, camera="front", depth=False, segmentation=False):
        r = self.renderer or mujoco.Renderer(self.model, self.h, self.w)
        if depth:
            r.enable_depth_rendering()
        if segmentation:
            r.enable_segmentation_rendering()
        r.update_scene(self.data, camera)
        img = r.render().copy()
        r.disable_depth_rendering()
        r.disable_segmentation_rendering()
        return img

    def gripper_width(self):
        return self.cfg["width"](self.data)

    def state(self):
        pos, quat = ee_pose(self.model, self.data, self.ee)
        return np.concatenate([pos, quat, [self.gripper_width()]]).astype(np.float32)

    def observe(self):
        return {
            "images": {c: self.render(c) for c in self.cameras},
            "state": self.state(),
            "qpos": self.data.qpos[:self.n_arm].astype(np.float32).copy(),
            "instruction": self.task["instruction"],
        }

    def body_pos(self, name):
        return self.data.body(name).xpos.copy()

    def cube_pos(self, color):
        return self.body_pos(f"cube_{color}")

    def cube_yaw(self, color):
        return yaw_of(self.data.body(f"cube_{color}").xquat)

    def tray_pos(self, color):
        return self.body_pos(f"tray_{color}")

    def success(self):
        t = self.task
        p = self.cube_pos(t["obj"])
        if t["type"] == "pick":
            return bool(p[2] > 0.08)
        released = self.grip < 0.5 and self.gripper_width() > 0.045
        if t["type"] == "place":
            tp = self.tray_pos(t["target"])
            # z < 0.10: resting in the tray, possibly on top of a cube that is already there (multi-step sorting)
            return bool(np.all(np.abs(p[:2] - tp[:2]) < TRAY_HALF - 0.01) and p[2] < 0.10 and released)
        q = self.cube_pos(t["target"])
        return bool(np.linalg.norm(p[:2] - q[:2]) < 0.02 and 0.03 < p[2] - q[2] < 0.05 and released)
