import mujoco
import numpy as np

DOWN = np.array([0.0, 1.0, 0.0, 0.0])
GRIP_OPEN, GRIP_CLOSED = 255.0, 0.0


def add_ee_site(spec, body="hand", offset=0.1034):
    spec.body(body).add_site(name="ee", pos=[0, 0, offset], size=[0.005] * 3, rgba=[1, 0, 0, 0])
    return spec


def ee_pose(model, data, site="ee"):
    sid = model.site(site).id
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, data.site_xmat[sid])
    return data.site_xpos[sid].copy(), q


def yaw_quat(yaw):
    """Gripper pointing down, rotated by yaw about the world z axis."""
    qz = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
    out = np.zeros(4)
    mujoco.mju_mulQuat(out, qz, DOWN)
    return out


def pose_error(pos, quat, cur_pos, cur_quat):
    err = np.zeros(6)
    err[:3] = pos - cur_pos
    neg, e4 = np.zeros(4), np.zeros(4)
    mujoco.mju_negQuat(neg, cur_quat)
    mujoco.mju_mulQuat(e4, quat, neg)
    if e4[0] < 0:
        e4 = -e4
    mujoco.mju_quat2Vel(err[3:], e4, 1.0)
    return err


def solve_ik(model, data, pos, quat, q0=None, site="ee", n_joints=7, iters=200, damping=1e-4, tol=1e-5, scratch=None):
    """Damped-least-squares IK on a scratch MjData (kinematics only). Returns (q, residual)."""
    dk = scratch if scratch is not None else mujoco.MjData(model)
    dk.qpos[:] = data.qpos
    if q0 is not None:
        dk.qpos[:n_joints] = q0
    sid = model.site(site).id
    jacp, jacr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    lo, hi = model.jnt_range[:n_joints, 0], model.jnt_range[:n_joints, 1]
    err = np.ones(6)
    for _ in range(iters):
        mujoco.mj_kinematics(model, dk)
        mujoco.mj_comPos(model, dk)
        cq = np.zeros(4)
        mujoco.mju_mat2Quat(cq, dk.site_xmat[sid])
        err = pose_error(pos, quat, dk.site_xpos[sid], cq)
        if np.linalg.norm(err) < tol:
            break
        mujoco.mj_jacSite(model, dk, jacp, jacr, sid)
        J = np.vstack([jacp, jacr])[:, :n_joints]
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
        dk.qpos[:n_joints] = np.clip(dk.qpos[:n_joints] + dq, lo, hi)
    return dk.qpos[:n_joints].copy(), float(np.linalg.norm(err))


def torque_panda(scene=True):
    """Menagerie Panda with its position servos replaced by 7 joint torque motors (+ the original gripper actuator).
    Returns (model, data) reset to the 'home' keyframe."""
    import mujoco_menagerie as mm
    spec = mm.get("franka_emika_panda").spec("scene" if scene else "panda")
    add_ee_site(spec)
    limits = [87, 87, 87, 87, 12, 12, 12]
    g = spec.actuator("actuator8")
    grip = dict(target=g.target, trntype=g.trntype, gaintype=g.gaintype, biastype=g.biastype,
                gainprm=list(g.gainprm), biasprm=list(g.biasprm), ctrlrange=list(g.ctrlrange), forcerange=list(g.forcerange))
    for i in range(8):
        spec.delete(spec.actuator(f"actuator{i + 1}"))
    for i in range(7):
        a = spec.add_actuator(name=f"motor{i + 1}", target=f"joint{i + 1}", trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.ctrlrange = [-limits[i], limits[i]]
        a.ctrllimited = 1
    a = spec.add_actuator(name="gripper", **{k: v for k, v in grip.items() if k not in ("gainprm", "biasprm", "ctrlrange", "forcerange")})
    a.gainprm[:len(grip["gainprm"])] = grip["gainprm"]
    a.biasprm[:len(grip["biasprm"])] = grip["biasprm"]
    a.ctrlrange, a.forcerange = grip["ctrlrange"], grip["forcerange"]
    a.ctrllimited = a.forcelimited = 1
    for k in list(spec.keys):
        spec.delete(k)
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[:7] = [0, 0, 0, -1.5708, 0, 1.5708, -0.7853]
    data.qpos[7:9] = 0.04
    mujoco.mj_forward(model, data)
    return model, data
