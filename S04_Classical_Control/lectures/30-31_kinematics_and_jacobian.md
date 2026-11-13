# 30–31 · Forward Kinematics and the Geometric Jacobian

**Section:** S04 Classical Action Models · **Type:** Math · **Duration:** ~35 min
**Companion notebooks:** `30_forward_kinematics.ipynb`, `31_jacobian_singularities.ipynb`

## Learning objectives
- Write forward kinematics of a serial arm with (modified) Denavit–Hartenberg parameters.
- Derive the geometric Jacobian column by column and verify it against MuJoCo.
- Detect singularities with the SVD and handle them with damping.

---

## Part 1: Forward kinematics

### 1. The problem
Given joint angles $q \in \mathbb{R}^n$, find the pose of the end effector: $T_{0,ee}(q) \in SE(3)$. FK is always unique and cheap. Every observation a policy sees ("where is my gripper?") depends on it.

### 2. Modified DH convention (Craig)
Each link $i$ is described by four numbers $(a_{i-1}, \alpha_{i-1}, d_i, \theta_i)$:
$$T_{i-1,i} = R_x(\alpha_{i-1})\,\mathrm{Tr}_x(a_{i-1})\,R_z(\theta_i)\,\mathrm{Tr}_z(d_i)$$
$$= \begin{bmatrix} c\theta & -s\theta & 0 & a\\ s\theta c\alpha & c\theta c\alpha & -s\alpha & -d\,s\alpha\\ s\theta s\alpha & c\theta s\alpha & c\alpha & d\,c\alpha\\ 0&0&0&1\end{bmatrix}$$
For revolute joints $\theta_i = q_i$. Chain them: $T_{0,7} = \prod_i T_{i-1,i}$, then add the flange offset ($d = 0.107$ m for the Panda).

**Panda parameters**
| i | a [m] | d [m] | α |
|---|---|---|---|
| 1 | 0 | 0.333 | 0 |
| 2 | 0 | 0 | −π/2 |
| 3 | 0 | 0.316 | π/2 |
| 4 | 0.0825 | 0 | π/2 |
| 5 | −0.0825 | 0.384 | −π/2 |
| 6 | 0 | 0 | π/2 |
| 7 | 0.088 | 0 | π/2 |

The notebook matches MuJoCo to $10^{-15}$ m over 1000 random configurations. MuJoCo's `hand` frame is rotated −45° about z relative to the DH flange. **Frame conventions differ between libraries**, so always verify with a numeric check.

### 3. MuJoCo doesn't use DH
MJCF describes each body by its offset from the parent plus joint axes (the "product of offsets" form, lecture 10). It's more general (any axis, any offset) and easier to read from CAD. DH is still worth learning: datasheets and textbooks use it, and it explains how a 7-DoF arm is built.

### 4. Workspace
Sampling 20,000 random configurations shows the reachable set. Requiring the gripper to **point down** (what tabletop grasping needs) shrinks it a lot: an annulus around the base, limited in height. TableTop's 30 × 50 cm workspace sits inside it. Targets outside make IK fail, which is why `TableTop.step` rejects unreachable moves.

## Part 2: The Jacobian

### 5. Definition
$$\begin{bmatrix} v \\ \omega \end{bmatrix} = J(q)\,\dot q,\qquad J \in \mathbb{R}^{6\times n}$$
For revolute joint $i$ with axis $z_i$ through point $p_i$ (world frame):
$$J_i = \begin{bmatrix} z_i \times (p_{ee} - p_i) \\ z_i \end{bmatrix}$$
Intuition: rotating joint $i$ moves the end effector in a circle around the joint axis. In MuJoCo, $z_i$ = `data.xaxis[j]`, $p_i$ = `data.xanchor[j]`; `mj_jacSite` returns the same matrix (verified to $10^{-17}$).

First-order prediction: $\Delta x \approx J\,\Delta q$ (the notebook compares the predicted and actual displacement for a 0.01 rad perturbation).

### 6. Singularities
At a singularity $J$ loses rank: some Cartesian directions become impossible, and $J^{-1}$ requires infinite joint speed.
- Measure with the SVD: manipulability $w = \prod \sigma_i$ and condition number $\sigma_{\max}/\sigma_{\min}$.
- Panda example: straightening the elbow ($q_4 \to 0$) drives $w \to 0$ and the condition number up by orders of magnitude.

### 7. Damped least squares
Pseudoinverse: $\dot q = J^+ v$ blows up near singularities. DLS:
$$\dot q = J^\top (JJ^\top + \lambda^2 I)^{-1} v$$
It trades a small velocity error for bounded joint speeds. Notebook: at $q_4 = -0.5$ the pseudoinverse asks for 9.2 rad/s; DLS with λ = 0.1 asks for 0.12 rad/s with a 5 cm/s velocity error.

## Key takeaways
- FK = a chain of homogeneous transforms (DH or MuJoCo's offsets), always unique.
- Jacobian column = $[z_i \times (p_{ee}-p_i);\ z_i]$; verify against `mj_jacSite`.
- Singularities are rank loss; damping keeps the commands bounded.

## References
- Craig, *Introduction to Robotics*, Ch. 3, 5
- Franka Emika, *Robot and interface specifications* (DH table)
- Lynch & Park, *Modern Robotics*, Ch. 4–5
