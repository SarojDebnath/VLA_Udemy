# 29 · Project 1 Brief: TableTop

**Section:** S03 · **Type:** Project brief · **Notebook:** `29_project_tabletop.ipynb` · **Library:** `vlacourse/tabletop.py`

## Goal
Build the world that every later model is trained and evaluated in: a Franka Panda on a table with colored cubes and trays, cameras, randomized layouts, language tasks and success checks.

## Specification
| Component | Choice | Why |
|---|---|---|
| Robot | Menagerie `franka_emika_panda` + `ee` site at the fingertip center | the most common arm in VLA datasets (DROID, LIBERO) |
| Objects | 3 free-joint cubes (4 cm), 1 static tray (12 cm); colors from {red, green, blue, yellow, purple, orange} | colors make language grounding testable |
| Cameras | `front` (third person), `wrist` (on the hand), `top`, `side` | OpenVLA uses one third-person view; π0/ACT use wrist + external |
| Randomization | non-overlapping xy positions in a 30 × 50 cm workspace, cube yaw ±45° | prevents memorization of a single layout |
| Tasks | `pick up the {c} cube`, `put the {c} cube in the {t} tray`, `stack the {c1} cube on the {c2} cube` | short-horizon, compositional |
| Action | 7-D: $\Delta x,\Delta y,\Delta z$ (m), $\Delta$roll/pitch/yaw (rad), grip ∈ [0, 1] at 10 Hz | same convention as OpenVLA/LIBERO end-effector deltas |
| Low-level control | DLS IK → joint targets → Panda position servos, 50 physics substeps per action | the "classical layer under the VLA" |
| Observation | images per camera, `state` = ee pos + quat + gripper width (8-D), `qpos` (7-D), instruction | the slots every VLA consumes |
| Success | pick: cube z > 8 cm; place: cube inside the tray footprint, resting, released; stack: cube centered on the other cube, released | privileged checks used only for evaluation |

## Steps in the notebook
1. Build a v0 scene with MjSpec from scratch (robot, table, cubes, tray, cameras).
2. Randomize layouts without overlap (rejection sampling).
3. Switch to the course version `vlacourse.TableTop`, inspect the observation and action spaces.
4. Check action semantics: commanded deltas vs measured end-effector displacement (servo lag of ~1 cm is expected).
5. Check that reachability is enforced: unreachable targets are rejected instead of letting IK diverge.
6. Check the success detectors by teleporting a cube into the tray.

## Engineering notes (lessons from building it)
- **Z-fighting:** a table top at the same height as the floor flickers in renders, so the floor was lowered under a full-height table.
- **Quaternion double cover:** orientation error must use the short way around ($w \ge 0$), or IK spins the wrong way.
- **Reachability:** the Panda's reach with the gripper pointing down shrinks with height; targets beyond it make IK diverge, so the env rejects them.

## Extensions (homework)
- Add a drawer (slide joint + handle) and an "open the drawer" task.
- Add distractor objects with meshes from another Menagerie asset.
- Swap the Panda for a UR5e + Robotiq using MjSpec attach (notebook 26) while keeping the same action space.
