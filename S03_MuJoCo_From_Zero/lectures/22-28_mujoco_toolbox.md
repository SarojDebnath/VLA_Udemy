# 22–28 · The MuJoCo Toolbox: Integrators, Actuators, Sensors, Cameras, MjSpec, Speed

**Section:** S03 · **Type:** Theory companion to labs 22–28 · **Duration:** ~35 min (one segment per lab)

---

## 22 · Integrators and timestep
| Integrator | Update | When to use |
|---|---|---|
| `Euler` (default) | semi-implicit: $v_{t+1} = v_t + h\,a_t$, $q_{t+1} = q_t + h\,v_{t+1}$; joint damping is treated implicitly | general purpose |
| `implicitfast` | implicit in velocity using the derivatives of actuator/passive forces (skips Coriolis terms) | **recommended for robots** with stiff servos |
| `implicit` | full implicit in velocity | rare; slower |
| `RK4` | 4th-order Runge–Kutta | smooth systems, accurate energy; ~3× slower |

Notebook results: at dt = 20 ms, Euler's energy drifts visibly while RK4 stays almost flat. **Stability lesson:** MuJoCo integrates damping it *knows about* (joint damping, velocity actuators) implicitly. Damping you compute yourself in Python (`ctrl = -kv * qvel`) is explicit and goes unstable once $k_v\,h/I > 2$. Put stiff feedback inside the model whenever possible.

Typical robot timestep: 1–2 ms physics, with a control loop at 10–50 Hz (`n_substeps = 1/(f_c h)`). TableTop uses 2 ms physics and 10 Hz control (50 substeps).

## 23 · Actuators
Every actuator computes
$$f = a\,u + b_0 + b_1\,l + b_2\,\dot l$$
($u$ = ctrl, $l$ = actuator length, e.g. the joint angle) and maps it through its transmission to generalized forces.
- `motor`: $a = 1$, $b = 0$, so ctrl is a torque.
- `position`: $a = k_p$, $b_1 = -k_p$, $b_2 = -k_v$, so ctrl is a target angle (a PD servo). `dampratio` picks $k_v$ for critical damping.
- `velocity`: $a = k_v$, $b_2 = -k_v$, so ctrl is a target speed.
- `general`: any $a, b$, which can reproduce all of the above exactly (shown in the notebook).
- Limits: `ctrlrange` clamps the command, `forcerange` saturates the force (visible as a slower, linear rise).
- The Menagerie **Panda** uses position servos ($k_p$ 2000–4500, force limits 12–87 N·m) plus a tendon-driven gripper with ctrl 0 (closed) to 255 (open).

## 24 · Sensors
Sensors write into `data.sensordata` (address `sensor_adr`, size `sensor_dim`) and never affect the physics.
- **Force/torque** at a site: the interaction force between the site's body and its parent. A wrist F/T sensor reads the weight of everything below it: the notebook sees about −9.4 N while holding the cube.
- **Touch**: normal force inside a site volume, i.e. fingertip pads. Both pads above 1 N means "grasped".
- **Rangefinder**: distance along the site's +z axis (set `intprm[0]`).
- **Frame / joint sensors**: poses and joint values; the `noise` attribute adds Gaussian noise, which helps train noise-robust policies.
- Sensors become **proprioception tokens** for VLAs (π0 consumes the robot state) and **monitors** for safety shields (S15).

## 25 · Cameras and rendering
`mujoco.Renderer(model, H, W)` renders offscreen:
- RGB (`uint8`), depth (`enable_depth_rendering`, metric `float32`), segmentation (`enable_segmentation_rendering`, `[object id, object type]` per pixel).
- Segmentation gives free ground-truth masks and boxes, useful for auxiliary losses, data filtering and evaluation.
- **Visual domain randomization:** jitter `cam_pos`, `cam_fovy`, `light_diffuse`, geom `rgba`/materials. VLAs are notoriously sensitive to camera pose; S12 measures it.

## 26 · MjSpec: models as code
`MjSpec` replaces string templating of XML:
```python
spec = mm.get("universal_robots_ur5e").spec("scene")
spec.site("attachment_site").attach_body(gripper.worldbody.first_body(), "gripper_", "")
body = spec.worldbody.add_body(name="obj", pos=[.5, 0, .05]); body.add_freejoint(); body.add_geom(...)
model = spec.compile();  xml = spec.to_xml()
```
Uses: procedural scenes, attaching end-effectors, randomizing physical parameters for domain randomization (S05), and building TableTop itself.

## 27 · Interaction and recording
- `mujoco.viewer.launch_passive(model, data, key_callback=...)` keeps your own loop in control; wrap modifications in `viewer.lock()` and call `viewer.sync()`. On macOS run with `mjpython`.
- **Mocap bodies** are kinematic targets you move by writing `data.mocap_pos`; a `weld` equality makes a dynamic body follow them softly. This is the basis of VR/teleop rigs.
- Record with the Renderer and save with `imageio`. Videos are the highest-bandwidth debugging tool in robot learning.

## 28 · Speed
| Method | Notebook throughput (Panda) |
|---|---|
| Python loop, `mj_step` per step | ≈ 90k steps/s |
| `mujoco.rollout`, 1 thread | ≈ 110k steps/s |
| `mujoco.rollout`, 16 threads | ≈ 660k steps/s |
| **MJX** (JAX, GPU) | millions of steps/s across thousands of parallel envs (S05) |

`rollout` takes a batch of initial states (`mj_getState`, `mjSTATE_FULLPHYSICS`) and control sequences, and returns states and sensor data. It's ideal for MPC/MPPI (S14) and for data generation. MJX trades single-env speed for massive parallelism and differentiability; MuJoCo Warp targets NVIDIA GPUs.

## Key takeaways
- Use `implicitfast` and keep stiff feedback inside the model.
- Actuators are gain/bias affine maps; position servos are PD controllers.
- Sensors and segmentation give free supervision; camera and lighting randomization build robustness.
- MjSpec builds models as code; `rollout`/MJX give scale.

## References
- MuJoCo docs: *Computation → Numerical integration, Actuation model*; *XML reference → sensor*; *Python → Rendering, MjSpec, rollout*; *MJX*
