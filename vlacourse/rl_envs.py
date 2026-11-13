"""Small MuJoCo environments for the RL section (Gymnasium API)."""
import gymnasium as gym
import mujoco
import mujoco_menagerie as mm
import numpy as np
from gymnasium import spaces

CARTPOLE_XML = """
<mujoco model="cartpole">
  <option timestep="0.01"/>
  <worldbody>
    <light pos="0 -2 3"/>
    <geom type="plane" size="4 1 .1" rgba=".85 .85 .85 1"/>
    <geom type="box" pos="0 0 .5" size="2.6 .02 .02" rgba=".4 .4 .4 1" contype="0" conaffinity="0"/>
    <body name="cart" pos="0 0 .5">
      <joint name="slider" type="slide" axis="1 0 0" range="-2.4 2.4" limited="false" damping="0"/>
      <geom type="box" size=".2 .1 .06" mass="1" rgba=".2 .4 .8 1"/>
      <body name="pole">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0"/>
        <geom type="capsule" fromto="0 0 0 0 0 1" size=".03" mass=".1" rgba=".9 .5 .2 1"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor joint="slider" gear="10" ctrlrange="-1 1"/></actuator>
</mujoco>"""

PENDULUM_XML = """
<mujoco model="pendulum">
  <option timestep="0.01" gravity="0 0 {g}"><flag contact="disable"/></option>
  <worldbody>
    <light pos="0 -2 3"/>
    <body name="pole" pos="0 0 1.5">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="0.05"/>
      <geom type="capsule" fromto="0 0 0 0 0 {L}" size=".04" mass="{m}" rgba=".9 .5 .2 1"/>
    </body>
  </worldbody>
  <actuator><motor joint="hinge" ctrlrange="-{tmax} {tmax}"/></actuator>
</mujoco>"""


class MjEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def render(self, size=(240, 320), camera=None):
        if not hasattr(self, "_renderer"):
            self._renderer = mujoco.Renderer(self.model, *size)
        if camera is None:
            camera = mujoco.MjvCamera()
            camera.lookat[:] = self.model.stat.center
            camera.distance, camera.azimuth, camera.elevation = 2.2 * self.model.stat.extent, 90, -10
        self._renderer.update_scene(self.data, camera)
        return self._renderer.render()


class CartPoleEnv(MjEnv):
    """Balance the pole. Discrete actions {push left, push right} (or continuous force in [-1, 1])."""

    def __init__(self, continuous=False, max_steps=500):
        self.model = mujoco.MjModel.from_xml_string(CARTPOLE_XML)
        self.data = mujoco.MjData(self.model)
        self.continuous, self.max_steps = continuous, max_steps
        self.action_space = spaces.Box(-1, 1, (1,), np.float32) if continuous else spaces.Discrete(2)
        self.observation_space = spaces.Box(-np.inf, np.inf, (4,), np.float32)

    def obs(self):
        return np.r_[self.data.qpos, self.data.qvel].astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.np_random.uniform(-0.05, 0.05, 2)
        self.data.qvel[:] = self.np_random.uniform(-0.05, 0.05, 2)
        mujoco.mj_forward(self.model, self.data)
        self.t = 0
        return self.obs(), {}

    def step(self, action):
        u = float(np.clip(action, -1, 1)[0]) if self.continuous else (1.0 if int(action) == 1 else -1.0)
        self.data.ctrl[0] = u
        mujoco.mj_step(self.model, self.data, 2)
        self.t += 1
        x, th = self.data.qpos
        terminated = bool(abs(x) > 2.4 or abs(th) > 0.21)
        truncated = self.t >= self.max_steps
        return self.obs(), 1.0, terminated, truncated, {}


class PendulumEnv(MjEnv):
    """Swing up and balance. Physical parameters are constructor args so they can be randomized."""

    def __init__(self, mass=1.0, length=1.0, gravity=9.81, max_torque=2.0, max_steps=200, randomize=None):
        self.params = dict(mass=mass, length=length, gravity=gravity, max_torque=max_torque)
        self.randomize = randomize
        self.max_steps = max_steps
        self.action_space = spaces.Box(-1, 1, (1,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (3,), np.float32)
        self.build(**self.params)

    def build(self, mass, length, gravity, max_torque):
        xml = PENDULUM_XML.format(m=mass, L=length, g=-gravity, tmax=max_torque)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.cur = dict(mass=mass, length=length, gravity=gravity, max_torque=max_torque)
        if hasattr(self, "_renderer"):
            del self._renderer

    def obs(self):
        th = self.data.qpos[0]
        return np.array([np.cos(th), np.sin(th), self.data.qvel[0]], np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.randomize:
            p = {k: v * self.np_random.uniform(1 - self.randomize.get(k, 0), 1 + self.randomize.get(k, 0)) for k, v in self.params.items()}
            self.build(**p)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0] = np.pi + self.np_random.uniform(-np.pi, np.pi)
        self.data.qvel[0] = self.np_random.uniform(-1, 1)
        mujoco.mj_forward(self.model, self.data)
        self.t = 0
        return self.obs(), {}

    def step(self, action):
        u = float(np.clip(action, -1, 1)[0]) * self.cur["max_torque"]
        self.data.ctrl[0] = u
        mujoco.mj_step(self.model, self.data, 5)
        self.t += 1
        th = ((self.data.qpos[0] + np.pi) % (2 * np.pi)) - np.pi
        cost = th ** 2 + 0.1 * self.data.qvel[0] ** 2 + 0.001 * u ** 2
        return self.obs(), -float(cost), False, self.t >= self.max_steps, {}


class PandaReachEnv(MjEnv):
    """Move the Panda fingertip to a random target. Action = joint-position deltas (7, scaled), state observation."""

    def __init__(self, max_steps=50, control_hz=20, delta=0.08, success_radius=0.03, sparse=False):
        spec = mm.get("franka_emika_panda").spec("scene")
        spec.body("hand").add_site(name="ee", pos=[0, 0, 0.1034], size=[0.005] * 3)
        tgt = spec.worldbody.add_body(name="target", pos=[0.5, 0, 0.3], mocap=True)
        tgt.add_geom(type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.03, 0, 0], rgba=[0, 1, 0, 0.5], contype=0, conaffinity=0)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)
        self.n_sub = int(1 / (control_hz * self.model.opt.timestep))
        self.max_steps, self.delta, self.radius, self.sparse = max_steps, delta, success_radius, sparse
        self.home = self.model.key("home").qpos[:7].copy()
        self.action_space = spaces.Box(-1, 1, (7,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (20,), np.float32)

    def obs(self):
        ee = self.data.site("ee").xpos
        return np.r_[self.data.qpos[:7], self.data.qvel[:7] * 0.1, self.target - ee, ee].astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        self.data.qpos[:7] = self.home + self.np_random.normal(0, 0.1, 7)
        self.data.ctrl[:7] = self.data.qpos[:7]
        self.target = self.np_random.uniform([0.3, -0.3, 0.1], [0.7, 0.3, 0.5])
        self.data.mocap_pos[0] = self.target
        mujoco.mj_forward(self.model, self.data)
        self.t = 0
        return self.obs(), {}

    def step(self, action):
        a = np.clip(action, -1, 1)
        lo, hi = self.model.jnt_range[:7, 0], self.model.jnt_range[:7, 1]
        self.data.ctrl[:7] = np.clip(self.data.qpos[:7] + self.delta * a, lo, hi)
        mujoco.mj_step(self.model, self.data, self.n_sub)
        self.t += 1
        dist = float(np.linalg.norm(self.target - self.data.site("ee").xpos))
        success = dist < self.radius
        reward = float(success) if self.sparse else -dist - 0.01 * float(np.sum(a ** 2)) + 0.5 * success
        return self.obs(), reward, False, self.t >= self.max_steps, {"success": success, "dist": dist}
