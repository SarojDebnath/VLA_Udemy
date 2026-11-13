"""S04 project 42c: LQR, from a cart-pole to a balancing Unitree G1 humanoid.

    linearize(m, d, ctrl0)        A, B of x' = A x + B u from MuJoCo itself (mjd_transitionFD, finite differences)
    dlqr(A, B, Q, R)              discrete-time LQR gain K (u = -K x) from the discrete algebraic Riccati equation
    cartpole()                    a cart-pole MJCF model (pole up = hinge angle 0)
    CartPole                      swing-up by energy pumping, then LQR catch-and-balance
    G1Balance                     whole-body LQR for a standing G1 (Menagerie), after the MuJoCo LQR tutorial:
                                  standing pose -> gravity-compensating set-point -> cost on the CoM over the feet
"""
import mujoco
import numpy as np
import scipy.linalg

CARTPOLE_XML = """
<mujoco model="cartpole">
  <option timestep="0.002" integrator="implicitfast"/>
  <visual><global offwidth="1920" offheight="1080"/><quality shadowsize="4096"/></visual>
  <asset>
    <texture name="sky" type="skybox" builtin="gradient" rgb1="0.35 0.5 0.7" rgb2="0.05 0.08 0.15" width="512" height="512"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.16 0.22 0.3" rgb2="0.2 0.27 0.36" width="512" height="512" mark="edge" markrgb="0.3 0.38 0.48"/>
    <material name="grid" texture="grid" texrepeat="8 8" reflectance="0.15"/>
  </asset>
  <worldbody>
    <light pos="0 -2 4" dir="0 0.4 -1" directional="true" castshadow="true"/>
    <geom name="floor" type="plane" size="6 6 0.1" material="grid" pos="0 0 -0.7"/>
    <geom name="rail" type="capsule" fromto="-2 0 0 2 0 0" size="0.02" rgba="0.55 0.55 0.6 1" contype="0" conaffinity="0"/>
    <geom type="box" pos="-2.05 0 -0.35" size="0.04 0.04 0.35" rgba="0.3 0.3 0.33 1" contype="0" conaffinity="0"/>
    <geom type="box" pos="2.05 0 -0.35" size="0.04 0.04 0.35" rgba="0.3 0.3 0.33 1" contype="0" conaffinity="0"/>
    <body name="cart">
      <joint name="slider" type="slide" axis="1 0 0" range="-1.9 1.9" limited="true" damping="0.2"/>
      <geom type="box" size="0.16 0.1 0.06" rgba="0.2 0.45 0.9 1" mass="1.0" contype="0" conaffinity="0"/>
      <body name="pole">
        <joint name="hinge" type="hinge" axis="0 1 0" damping="0.001"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.6" size="0.022" rgba="0.95 0.6 0.2 1" mass="0.1" contype="0" conaffinity="0"/>
        <geom type="sphere" pos="0 0 0.6" size="0.055" rgba="0.95 0.35 0.15 1" mass="0.2" contype="0" conaffinity="0"/>
      </body>
    </body>
  </worldbody>
  <actuator><motor name="push" joint="slider" ctrlrange="-30 30" ctrllimited="true"/></actuator>
</mujoco>
"""


def cartpole():
    return mujoco.MjModel.from_xml_string(CARTPOLE_XML)


def linearize(m, d, ctrl0, eps=1e-6):
    """Discrete-time A (2nv x 2nv), B (2nv x nu) of the dynamics around the current qpos/qvel and ctrl0, by finite
    differences of mj_step (state in tangent space: [dq, dqvel])."""
    d.ctrl[:] = ctrl0
    A = np.zeros((2 * m.nv, 2 * m.nv))
    B = np.zeros((2 * m.nv, m.nu))
    mujoco.mjd_transitionFD(m, d, eps, True, A, B, None, None)
    return A, B


def dlqr(A, B, Q, R):
    """K for u = -K x, minimizing sum x'Qx + u'Ru, from the discrete algebraic Riccati equation."""
    P = scipy.linalg.solve_discrete_are(A, B, Q, R)
    return np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class CartPole:
    """Cart-pole: angle theta = 0 is upright. LQR around upright; energy pumping to swing up from hanging."""

    def __init__(self, q_weights=(1.0, 10.0, 0.1, 0.1), r=0.01):
        self.m = cartpole()
        self.d = mujoco.MjData(self.m)
        self.set(0.0, 0.0)
        A, B = linearize(self.m, self.d, np.zeros(1))
        self.A, self.B = A, B
        self.Q, self.R = np.diag(q_weights), np.eye(1) * r
        self.K = dlqr(A, B, self.Q, self.R)
        # pendulum parameters (for the energy) read from the model
        m = self.m
        pole = m.body("pole").id
        self.mp = float(m.body_subtreemass[pole])
        mujoco.mj_forward(m, self.d)
        self.lc = float(self.d.subtree_com[pole][2] - self.d.xpos[pole][2])            # pivot -> CoM, pole upright
        self.I = self._pivot_inertia()

    def _pivot_inertia(self):
        # inertia of the pole about the hinge = sum over its geoms (rod + tip), parallel-axis
        rod, tip = 0.1, 0.2
        return rod * 0.6 ** 2 / 3 + tip * 0.6 ** 2

    def set(self, x, theta, xdot=0.0, thetadot=0.0):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = [x, theta]
        self.d.qvel[:] = [xdot, thetadot]
        mujoco.mj_forward(self.m, self.d)

    def state(self):
        x, th = self.d.qpos
        return np.array([x, wrap(th), *self.d.qvel])

    def energy(self):
        th, w = wrap(self.d.qpos[1]), self.d.qvel[1]
        return 0.5 * self.I * w ** 2 + self.mp * 9.81 * self.lc * (np.cos(th) - 1.0)    # 0 when upright and still

    def lqr_u(self):
        return float(np.clip(-self.K @ self.state(), -30, 30)[0])

    def swing_u(self, k=40.0):
        """Energy pumping (Astrom & Furuta): push the cart so the pendulum gains energy until it reaches the upright
        energy; plus a weak spring keeping the cart near the centre of the rail."""
        th, w = wrap(self.d.qpos[1]), self.d.qvel[1]
        x, v = self.d.qpos[0], self.d.qvel[0]
        e = self.energy()
        u = k * e * np.sign(w * np.cos(th) + 1e-9) - 8.0 * x - 3.0 * v
        return float(np.clip(u, -30, 30))

    def step(self, u, push=0.0):
        self.d.ctrl[0] = u
        self.d.xfrc_applied[self.m.body("pole").id, 0] = push
        mujoco.mj_step(self.m, self.d)


class G1Balance:
    """Whole-body LQR for the standing Unitree G1 (29 position-controlled joints, free-floating base).

    Following the MuJoCo LQR tutorial: (1) find the pelvis height where the feet carry the robot's weight exactly,
    (2) inverse dynamics gives the joint torques that hold that pose; the position servos' set-points that produce
    them become ctrl0, (3) linearize, (4) cost = keep the centre of mass over the feet + keep the joints near the pose,
    (5) u = ctrl0 - K dx."""

    def __init__(self, balance_cost=1000.0, balance_joint_cost=3.0, other_joint_cost=0.3, r=1.0):
        import mujoco_menagerie as mm
        spec = mm.get("unitree_g1").spec("scene")
        spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
        spec.visual.quality.shadowsize = 4096
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        m, d = self.m, self.d
        key = m.key("stand").id
        # (1) pelvis height at which the vertical inverse-dynamics force is ~zero (feet take exactly the weight)
        heights = np.linspace(-0.01, 0.01, 401)
        vert = []
        for h in heights:
            mujoco.mj_resetDataKeyframe(m, d, key)
            mujoco.mj_forward(m, d)
            d.qacc[:] = 0
            d.qpos[2] += h
            mujoco.mj_inverse(m, d)
            vert.append(d.qfrc_inverse[2])
        best = heights[int(np.argmin(np.abs(vert)))]
        mujoco.mj_resetDataKeyframe(m, d, key)
        d.qpos[2] += best
        mujoco.mj_forward(m, d)
        d.qacc[:] = 0
        mujoco.mj_inverse(m, d)
        self.qpos0 = d.qpos.copy()
        qfrc0 = d.qfrc_inverse.copy()
        # (2) servo set-points that produce the required joint torques: force = kp (ctrl - q) - kv qdot
        self.ctrl0 = np.zeros(m.nu)
        for i in range(m.nu):
            j = m.actuator_trnid[i, 0]
            q = d.qpos[m.jnt_qposadr[j]]
            tau = qfrc0[m.jnt_dofadr[j]]
            kp = m.actuator_gainprm[i, 0]
            self.ctrl0[i] = q + tau / kp
        # (3) linearize around (qpos0, ctrl0)
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self.qpos0
        self.A, self.B = linearize(m, d, self.ctrl0)
        # (4) cost: CoM over the left foot (both feet are symmetric about it at this pose) + joint deviations
        mujoco.mj_resetData(m, d)
        d.qpos[:] = self.qpos0
        mujoco.mj_forward(m, d)
        jac_com = np.zeros((3, m.nv))
        mujoco.mj_jacSubtreeCom(m, d, jac_com, m.body("torso_link").id)
        jac_foot = np.zeros((3, m.nv))
        mujoco.mj_jacBodyCom(m, d, jac_foot, None, m.body("left_ankle_roll_link").id)
        jac_r = np.zeros((3, m.nv))
        mujoco.mj_jacBodyCom(m, d, jac_r, None, m.body("right_ankle_roll_link").id)
        jac_diff = jac_com - 0.5 * (jac_foot + jac_r)
        Qbalance = jac_diff.T @ jac_diff
        joint_names = [m.joint(i).name for i in range(m.njnt)]
        root_dofs = range(6)
        body_dofs = range(6, m.nv)
        balance_dofs = [m.jnt_dofadr[m.joint(n).id] for n in joint_names if any(k in n for k in ("hip", "knee", "ankle", "waist"))]
        other_dofs = np.setdiff1d(body_dofs, balance_dofs)
        Qjoint = np.eye(m.nv)
        Qjoint[root_dofs, root_dofs] *= 0
        Qjoint[balance_dofs, balance_dofs] *= balance_joint_cost
        Qjoint[other_dofs, other_dofs] *= other_joint_cost
        Qpos = balance_cost * Qbalance + Qjoint
        self.Q = np.block([[Qpos, np.zeros((m.nv, m.nv))], [np.zeros((m.nv, 2 * m.nv))]])
        self.R = np.eye(m.nu) * r
        self.K = dlqr(self.A, self.B, self.Q, self.R)
        self.reset()

    def reset(self):
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[:] = self.qpos0
        self.d.ctrl[:] = self.ctrl0
        mujoco.mj_forward(self.m, self.d)

    def dx(self):
        dq = np.zeros(self.m.nv)
        mujoco.mj_differentiatePos(self.m, dq, 1, self.qpos0, self.d.qpos)
        return np.r_[dq, self.d.qvel]

    def control(self, mode="lqr"):
        """'lqr': ctrl0 - K dx.  'pd': hold the set-point ctrl0 (each joint's own position servo = joint PD)."""
        if mode == "pd":
            return self.ctrl0
        u = self.ctrl0 - self.K @ self.dx()
        return np.clip(u, self.m.actuator_ctrlrange[:, 0], self.m.actuator_ctrlrange[:, 1])

    def fallen(self):
        return self.d.qpos[2] < 0.55

    def push_test(self, force, mode, direction=(1, 0, 0), duration=0.1, T=3.0, frames=None, film=None):
        """Stand, push the torso with `force` N for `duration` s at t = 0.5 s, see whether it is still up after T s."""
        self.reset()
        m, d = self.m, self.d
        torso = m.body("torso_link").id
        n = int(T / m.opt.timestep)
        for k in range(n):
            t = k * m.opt.timestep
            d.ctrl[:] = self.control(mode)
            d.xfrc_applied[torso, :3] = np.asarray(direction, float) * force if 0.5 <= t < 0.5 + duration else 0.0
            mujoco.mj_step(m, d)
            if film is not None and k % 10 == 0:
                frames.append(film(self, t, force if 0.5 <= t < 0.5 + duration else 0.0))
            if self.fallen():
                if film is not None:
                    for _ in range(25):
                        mujoco.mj_step(m, d)
                        frames.append(film(self, t, 0.0))
                return False
        return True

    def max_push(self, mode, direction=(1, 0, 0), lo=0.0, hi=600.0, iters=9, **kw):
        """Largest torso push (N, 0.1 s) the controller survives: bisection."""
        if not self.push_test(lo, mode, direction, **kw):
            return 0.0
        for _ in range(iters):
            mid = 0.5 * (lo + hi)
            if self.push_test(mid, mode, direction, **kw):
                lo = mid
            else:
                hi = mid
        return lo
