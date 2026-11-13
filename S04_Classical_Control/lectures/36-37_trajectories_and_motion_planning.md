# 36–37 · Trajectory Generation and Sampling-Based Motion Planning

**Section:** S04 · **Type:** Theory + math · **Duration:** ~35 min
**Companion notebooks:** `36_trajectory_generation`, `37_rrt_motion_planning`

---

## Part 1: Trajectories (notebook 36)

### 1. Path vs trajectory
A **path** is a geometric curve; a **trajectory** is a path plus timing $s(t)$. Write $q(t) = q_0 + s(t)(q_1 - q_0)$, $s: [0,T]\to[0,1]$.

### 2. Time-scaling profiles
| Profile | $s(u)$, $u = t/T$ | Boundary conditions | Note |
|---|---|---|---|
| Cubic | $3u^2 - 2u^3$ | zero velocity | acceleration jumps → infinite jerk |
| Quintic (min-jerk) | $10u^3 - 15u^4 + 6u^5$ | zero velocity and acceleration | smooth; human-like reaching |
| Trapezoidal velocity | accelerate, cruise, decelerate | velocity/acceleration limits | industrial default |

Peak quintic speed is $\dot s_{\max} = 1.875/T$. The minimum feasible duration under joint speed limits is $T_{\min} = \max_i 1.875\,|\Delta q_i| / \dot q_{i,\max}$ (1.55 s for the notebook's move).

### 3. Joint-space vs Cartesian interpolation
- Joint-space: simple, always feasible between valid configurations, but the end effector traces a curve (4.8 cm off the straight line in the notebook).
- Cartesian: a straight line for the tool (needed for insertion and approaching objects), but every point needs IK and may cross singularities.

**Learned policies as trajectory generators:** action chunks (ACT, Diffusion Policy, π0) are short trajectories. Smoothness, temporal ensembling and chunk boundaries (S08) are trajectory-generation problems.

## Part 2: Motion planning (notebook 37)

### 4. Configuration space
Planning happens in C-space, $q \in \mathbb{R}^7$. An obstacle in the workspace becomes a complicated forbidden region in C-space, so we never build it explicitly. We **sample** and **check**: set `qpos`, run `mj_kinematics` + `mj_collision`, and look for robot–obstacle contacts.

### 5. RRT and RRT-Connect
- RRT (LaValle 1998): grow a tree from the start toward random samples, a fixed step at a time, keeping collision-free edges.
- **RRT-Connect** (Kuffner & LaValle 2000): grow trees from both start and goal and greedily try to connect them. The notebook solves the wall-and-shelf problem in ~250 iterations and about 0.07 s.
- Probabilistically complete (finds a path if one exists, given enough time), but **not optimal**: raw paths are jagged.

### 6. Post-processing
- **Shortcutting:** repeatedly try to replace a sub-path with a straight collision-free segment (40 → 5 waypoints, 11.2 → 7.0 rad).
- Then time-parameterize (Part 1) for execution.

### 7. The plan–execution gap
The plan is collision-free, but execution isn't the plan: servo lag of several degrees at speed let a finger clip the shelf in an early version of this notebook. Fixes: **clearance margins** in the checker (1 cm on obstacles) and slower or time-optimal execution. Learned policies face the same gap; S15 adds runtime monitors.

### 8. Where planners sit today
VLAs replace planners for short, contact-rich, semantically specified motions. Planners remain load-bearing for long collision-free transit, safety-certified motion, and as data generators (MimicGen-style pipelines use planners to create demonstrations).

## Key takeaways
- Use quintic/min-jerk or trapezoidal timing; check joint limits to pick $T$.
- RRT-Connect + shortcutting solves high-dimensional planning with only a collision checker.
- Always pad obstacles: execution deviates from the plan.

## References
- LaValle, *Planning Algorithms* (2006), Ch. 5
- Kuffner & LaValle 2000, *RRT-Connect*
- Lynch & Park, *Modern Robotics*, Ch. 9–10
- Kandapagari, *Action Models for Robot Learning*, §4.2
