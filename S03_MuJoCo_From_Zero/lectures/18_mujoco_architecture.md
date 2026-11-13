# 18 · MuJoCo Architecture: MJCF → MjSpec → MjModel / MjData, and the Simulation Pipeline

**Section:** S03 MuJoCo From Zero · **Type:** Theory · **Duration:** ~25 min
**Companion notebooks:** `19_mjcf_by_hand`, `26_mjspec_procedural`

## Learning objectives
- Explain the model/data split and why MuJoCo never allocates memory while stepping.
- Map MJCF elements to what they do physically.
- List the stages of `mj_step` and know what `mj_forward` computes.
- Avoid the "stale derived quantities" pitfall.

---

## 1. What MuJoCo is
**Mu**lti-**Jo**int dynamics with **Co**ntact (Todorov, Erez, Tassa 2012): a physics engine designed for model-based control and learning. It's open source (Apache-2.0) since 2022 and maintained by Google DeepMind. Its design goals:
- **Generalized coordinates.** Joints, not maximal coordinates, so there's no joint drift.
- **Convex, soft contact model.** Contacts are an optimization problem with a unique solution, so simulation is smooth and invertible.
- **Speed and determinism.** Millions of steps per second on a CPU; bitwise-reproducible on the same machine and version.

The course uses it for every lab: LIBERO, robosuite, D4RL, MetaWorld and MuJoCo Playground are all built on it.

## 2. The four representations
```
MJCF / URDF / MJZ file ──parse──▶ MjSpec (editable) ──compile──▶ MjModel (constant) + MjData (state)
```
| Object | Contains | Changes during simulation? |
|---|---|---|
| **MJCF** | XML text: the human-authored model | no |
| **MjSpec** | in-memory editable tree (add / delete / attach bodies) | no, it's edited *between* compiles |
| **MjModel** | compiled constants: masses, geometry, joint axes, actuator gains, options | no (but you can write to it, e.g. randomization) |
| **MjData** | state (`qpos`, `qvel`, `act`, `ctrl`, `time`) + all intermediate results (`xpos`, `qM`, `contact`, `sensordata`, …) | yes |

All `MjData` arrays are preallocated, so `mj_step` does no allocation. That's what makes it fast and safe to call 100k times per second.

## 3. MJCF building blocks
| Element | Purpose | Notes |
|---|---|---|
| `body` | rigid frame with mass/inertia, nested under `worldbody` | a body with no joint is welded to its parent |
| `joint` | degrees of freedom: `hinge`, `slide`, `ball`, `free` | several joints per body allowed |
| `geom` | shape for collision + visuals: plane, sphere, capsule, ellipsoid, cylinder, box, mesh, hfield | collisions are convex; meshes use their convex hull |
| `site` | massless reference frame | sensors, tendons, IK targets, cameras |
| `camera`, `light` | rendering | can be attached to moving bodies (wrist camera) |
| `actuator` | transmission × activation × gain/bias | `motor`, `position`, `velocity`, `general`, `muscle`, … |
| `sensor` | writes into `sensordata` | force, torque, touch, IMU, rangefinder, frame poses, … |
| `equality` | constraints: `weld`, `connect`, `joint` | closes kinematic loops, mocap following |
| `tendon` | fixed (joint combos) or spatial (site paths) | Panda's gripper coupling |
| `contact` | `pair` / `exclude` | explicit contact control |
| `default` | CSS-like attribute classes | keeps large models tidy |
| `keyframe` | named initial states | `mj_resetDataKeyframe` |

**`nq` vs `nv`:** a ball joint has 4 position numbers (a quaternion) but 3 velocities; a free joint has 7 and 6. So `nq ≥ nv`. Use `mj_differentiatePos` / `mj_integratePos` to move between them.

## 4. The pipeline inside `mj_step`
Condensed from the MuJoCo docs:
1. **Position stage** (`mj_fwdPosition`): forward kinematics → body poses; center-of-mass quantities; tendon lengths; joint-space inertia $M(q)$ via the composite rigid-body algorithm, plus its sparse factorization; **collision detection** → contacts; constraint Jacobians; actuator moment arms.
2. **Velocity stage** (`mj_fwdVelocity`): body velocities; passive forces (springs, dampers, fluid); bias forces $c(q,\dot q)$ via recursive Newton–Euler.
3. **Control callback**, then **actuation** (`mj_fwdActuation`): actuator forces from `ctrl`.
4. **Acceleration** (`mj_fwdAcceleration`, `mj_fwdConstraint`): unconstrained acceleration, then the constraint solver computes contact/limit/equality forces → `qacc`.
5. **Sensors** at the position, velocity and acceleration stages.
6. **Integration**: semi-implicit Euler, implicit(fast), or RK4 → new `qpos`, `qvel`, `time`.

`mj_forward` runs stages 1–5 *without* integrating. Use it after you set `qpos` by hand.

## 5. The stale-data pitfall
Setting `data.qpos[...] = ...` does **not** update `data.xpos`, sensors or Jacobians. Call `mj_forward` (or at least `mj_kinematics`) first. Also, after `mj_step`, derived quantities in `MjData` describe the state *before* integration. The notebook `19` shows `xpos` reading zeros until `mj_forward` runs.

## 6. Python bindings in one slide
```python
model = mujoco.MjModel.from_xml_path("scene.xml")    # or from_xml_string, or spec.compile()
data  = mujoco.MjData(model)
mujoco.mj_step(model, data, nstep=10)                # loops in C
data.body("hand").xpos; model.actuator("actuator1").gainprm  # named access
with mujoco.Renderer(model, 240, 320) as r: r.update_scene(data, "front"); img = r.render()
```
Arrays are zero-copy views into C memory. **`.copy()` before you store them in a log.**

## Key takeaways
- MjModel holds constants, MjData holds state plus derived quantities; nothing is allocated while stepping.
- MJCF is built from bodies, joints, geoms, sites, actuators and sensors; `nq ≠ nv` for ball and free joints.
- `mj_step` = position → velocity → actuation → constraint solve → integrate; `mj_forward` skips the integration.
- Call `mj_forward` after editing `qpos`, and copy arrays before logging them.

## References
- MuJoCo docs: *Overview*, *Computation → Simulation pipeline*, *Modeling*, *Python*
- Todorov, Erez & Tassa 2012, *MuJoCo: A physics engine for model-based control* (IROS)
