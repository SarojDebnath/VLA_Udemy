"""Small MuJoCo tasks with genuinely multimodal demonstrations (S08)."""
import mujoco
import numpy as np

AROUND_XML = """
<mujoco model="around">
  <option timestep="0.01" gravity="0 0 0"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="1 1 .01" rgba=".9 .9 .9 1" contype="0" conaffinity="0"/>
    <geom name="obstacle" type="cylinder" pos="0 0 .02" size=".18 .02" rgba=".8 .2 .2 1"/>
    <site name="goal" pos="0 .55 .02" size=".05" rgba="0 .8 0 .5"/>
    <body name="agent" pos="0 -.55 .03">
      <joint name="x" type="slide" axis="1 0 0" damping="5"/>
      <joint name="y" type="slide" axis="0 1 0" damping="5"/>
      <geom type="sphere" size=".03" rgba=".2 .4 .9 1" mass=".2"/>
    </body>
  </worldbody>
  <actuator>
    <velocity joint="x" kv="20" ctrlrange="-1 1"/>
    <velocity joint="y" kv="20" ctrlrange="-1 1"/>
  </actuator>
</mujoco>"""


class AroundObstacleEnv:
    """Reach the green goal by going around the red disc. Action = planar velocity command (2-D, |a| <= 1).
    Demonstrations go left OR right of the obstacle: averaging them drives straight into it."""

    def __init__(self, max_steps=120, image_size=(96, 96)):
        self.model = mujoco.MjModel.from_xml_string(AROUND_XML)
        self.data = mujoco.MjData(self.model)
        self.max_steps, self.size = max_steps, image_size
        self.goal = np.array([0.0, 0.55])
        self.rng = np.random.default_rng(0)
        self._r = None

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = [self.rng.uniform(-0.1, 0.1), self.rng.uniform(-0.05, 0.05)]
        mujoco.mj_forward(self.model, self.data)
        self.t, self.collided = 0, False
        return self.obs()

    def pos(self):
        return self.data.body("agent").xpos[:2].copy()

    def obs(self):
        return np.r_[self.pos(), self.data.qvel].astype(np.float32)

    def step(self, a):
        a = np.clip(np.asarray(a, float), -1, 1)
        self.data.ctrl[:] = a * 0.6
        mujoco.mj_step(self.model, self.data, 5)
        self.t += 1
        self.collided |= np.linalg.norm(self.pos()) < 0.18 + 0.03 + 1e-3
        reached = np.linalg.norm(self.pos() - self.goal) < 0.06
        success = reached and not self.collided
        done = reached or self.collided or self.t >= self.max_steps
        return self.obs(), float(success), done, {"success": bool(success), "collided": bool(self.collided)}

    def render(self):
        if self._r is None:
            self._r = mujoco.Renderer(self.model, *self.size)
        cam = mujoco.MjvCamera()
        cam.lookat[:] = [0, 0, 0]; cam.distance, cam.azimuth, cam.elevation = 1.9, 90, -90
        self._r.update_scene(self.data, cam)
        return self._r.render()


class AroundExpert:
    """Goes around the obstacle on a randomly chosen side via a waypoint, then to the goal."""

    def __init__(self, env, side=None, rng=None):
        rng = rng or np.random.default_rng()
        self.env = env
        self.side = side if side is not None else rng.choice([-1, 1])
        self.wp = [np.array([self.side * rng.uniform(0.34, 0.42), rng.uniform(-0.05, 0.05)]), env.goal]
        self.i = 0

    def act(self, obs=None):
        p = self.env.pos()
        if self.i == 0 and np.linalg.norm(p - self.wp[0]) < 0.08:
            self.i = 1
        d = self.wp[self.i] - p
        return np.clip(d / max(np.linalg.norm(d), 1e-6) * min(1.0, 4 * np.linalg.norm(d)), -1, 1).astype(np.float32)


def collect_around(n, seed=0, images=False):
    env, rng, eps = AroundObstacleEnv(), np.random.default_rng(seed), []
    for e in range(n):
        o = env.reset(seed=seed + e)
        ex = AroundExpert(env, rng=rng)
        O, A, I = [], [], []
        done = False
        while not done:
            a = ex.act()
            O.append(o); A.append(a)
            if images:
                I.append(env.render())
            o, _, done, info = env.step(a)
        eps.append(dict(obs=np.array(O), act=np.array(A), side=ex.side, success=info["success"], img=np.array(I) if images else None))
    return eps
