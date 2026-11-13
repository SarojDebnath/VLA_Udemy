# 09 · Rotations: Matrices, Euler Angles, Axis-Angle, Quaternions

**Section:** S02 · **Type:** Math · **Duration:** ~30 min
**Companion notebook:** `09_rotations_quaternions.ipynb`

## Learning objectives
- Represent a 3D rotation four ways and convert between them.
- Explain gimbal lock and the double cover of quaternions.
- Know which representation VLA action spaces use, and why.

---

## 1. Rotation matrices, $SO(3)$
$$SO(3) = \{R \in \mathbb{R}^{3\times3} : R^\top R = I,\; \det R = +1\}$$
- The columns of $R$ are the rotated frame's axes expressed in the parent frame.
- Inverse = transpose. Composition = multiplication, **which is not commutative**: $R_zR_x \neq R_xR_z$.
- 9 numbers for 3 degrees of freedom, so they're redundant and must be re-orthonormalized after numerical drift.

Elementary rotations:
$$R_z(\psi)=\begin{bmatrix}\cos\psi&-\sin\psi&0\\ \sin\psi&\cos\psi&0\\0&0&1\end{bmatrix}$$

## 2. Euler angles
Three successive rotations about axes, e.g. ZYX (yaw $\psi$, pitch $\theta$, roll $\phi$):
$$R = R_z(\psi)\,R_y(\theta)\,R_x(\phi)$$
- They're intuitive, and the robot action spaces of RT-1 and OpenVLA (Δroll, Δpitch, Δyaw) use them.
- **Gimbal lock:** at $\theta = \pm 90°$, yaw and roll rotate about the same axis, so one degree of freedom is lost. The notebook shows yaw $=+0.3$ and roll $=-0.3$ giving the *same* matrix at pitch $=90°$.
- There are 12 conventions. Always check which one a dataset uses.

## 3. Axis-angle and the rotation vector
Any rotation is a single rotation by angle $\theta$ about a unit axis $\hat\omega$. Rodrigues' formula:
$$R = I + \sin\theta\,[\hat\omega]_\times + (1-\cos\theta)\,[\hat\omega]_\times^2$$
The **rotation vector** $r = \theta\hat\omega \in \mathbb{R}^3$ is minimal and works well as a *small delta* action. Our TableTop action uses it: $(\Delta r_x, \Delta r_y, \Delta r_z)$. It's the exponential-map coordinate: $R = \exp([r]_\times)$.

## 4. Unit quaternions
$$q = \big(\cos\tfrac{\theta}{2},\; \sin\tfrac{\theta}{2}\,\hat\omega\big) = (w, x, y, z),\qquad \|q\| = 1$$
- 4 numbers, no singularities, cheap composition (the Hamilton product), easy to renormalize.
- **Double cover:** $q$ and $-q$ are the same rotation. A network that regresses quaternions must handle this (e.g. force $w \ge 0$).
- **MuJoCo uses $(w,x,y,z)$; SciPy uses $(x,y,z,w)$.** This mismatch is a classic source of silent bugs.
- Composition: $q_{ab} = q_a \otimes q_b$; rotating a vector: $v' = q \otimes (0,v) \otimes q^*$.

## 5. Interpolation: SLERP
Linear interpolation of matrices or Euler angles leaves the rotation manifold or takes strange paths. Spherical linear interpolation follows the shortest arc at constant angular speed:
$$\mathrm{slerp}(q_0,q_1;s) = \frac{\sin((1-s)\Omega)}{\sin\Omega}q_0 + \frac{\sin(s\Omega)}{\sin\Omega}q_1,\quad \cos\Omega = q_0\cdot q_1$$
The notebook interpolates the Panda gripper orientation with SLERP and IK.

## 6. Orientation error for control
The error between a desired $q_d$ and current $q$ as a rotation vector:
$$e_{rot} = \log\!\big(q_d \otimes q^{-1}\big) \in \mathbb{R}^3$$
MuJoCo computes it with `mju_mulQuat` + `mju_negQuat` + `mju_quat2Vel`; our IK solver uses exactly this.

## 7. Which representation for learning?
| Use | Best choice |
|---|---|
| Delta actions (VLA outputs) | rotation vector / small Euler deltas |
| Absolute orientation regression | 6D representation (two columns of $R$; Zhou et al. 2019) |
| Simulation state, composition | quaternion |
| Human-readable logs | Euler angles (state the convention!) |

## Key takeaways
- $SO(3)$: orthonormal, det +1, inverse = transpose, order matters.
- Euler angles suffer gimbal lock; quaternions have a double cover; rotation vectors are ideal for small deltas.
- MuJoCo quaternions are $(w,x,y,z)$. Check conventions at every library boundary.

## References
- Lynch & Park, *Modern Robotics*, Ch. 3
- Zhou et al. 2019, *On the Continuity of Rotation Representations in Neural Networks*
- MuJoCo API: `mju_mulQuat`, `mju_quat2Mat`, `mju_quat2Vel`
