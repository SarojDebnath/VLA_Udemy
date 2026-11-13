# 08 · Linear Algebra for Robotics: SVD, Least Squares, Manipulability

**Section:** S02 Math Foundations · **Type:** Math · **Duration:** ~25 min
**Companion notebook:** `08_linear_algebra.ipynb`

## Learning objectives
- Read a matrix as a linear map and interpret its SVD geometrically.
- Solve over-determined systems with least squares (normal equations and the pseudoinverse).
- Use the SVD of a robot Jacobian to measure manipulability and detect singularities.

---

## 1. Matrices are maps
A matrix $A \in \mathbb{R}^{m\times n}$ maps $x \in \mathbb{R}^n$ to $Ax \in \mathbb{R}^m$. In robotics the important maps are:
- rotation matrices $R \in SO(3)$ (lecture 09),
- the **Jacobian** $J(q)$: joint velocities → end-effector velocity, $\dot x = J(q)\,\dot q$,
- the camera matrix $K$: 3D points → pixels (lecture 10).

## 2. Singular Value Decomposition
Every matrix factors as

$$A = U \Sigma V^\top,\qquad U^\top U = I,\; V^\top V = I,\; \Sigma = \mathrm{diag}(\sigma_1 \ge \sigma_2 \ge \dots \ge 0)$$

Geometric reading: rotate by $V^\top$, stretch the axes by $\sigma_i$, rotate by $U$. So $A$ maps the unit sphere to an **ellipsoid** with semi-axes $\sigma_i u_i$.
- Rank = number of nonzero $\sigma_i$.
- Condition number $\kappa = \sigma_{\max}/\sigma_{\min}$: how much the map distorts, and how much errors get amplified when you invert it.

## 3. Least squares
Given more equations than unknowns, $Ax \approx b$, minimize $\|Ax-b\|^2$. Setting the gradient to zero:

$$\nabla_x \|Ax-b\|^2 = 2A^\top(Ax-b) = 0 \;\Rightarrow\; A^\top A\,x = A^\top b \quad\text{(normal equations)}$$

$$x^* = (A^\top A)^{-1}A^\top b = A^{+} b,\qquad A^{+} = V\Sigma^{+}U^\top$$

The notebook calibrates a sensor's scale and offset from 40 noisy measurements. This is the same math behind hand-eye calibration, system identification, and the linear layer of every neural network.

**Damped least squares** adds a penalty $\lambda\|x\|^2$:

$$x^* = A^\top (AA^\top + \lambda I)^{-1} b$$

This stays bounded when $A$ is near-singular. It's the exact formula our IK solver uses (S04).

## 4. Manipulability
For the end-effector, $\dot x = J\dot q$. Unit-norm joint velocities $\|\dot q\| \le 1$ map to the **velocity ellipsoid** whose axes come from the SVD of $J$. Yoshikawa's manipulability index:

$$w(q) = \sqrt{\det(JJ^\top)} = \prod_i \sigma_i$$

- $w \to 0$ at a **singularity**: some Cartesian direction can't be produced, and IK needs huge joint speeds.
- In the notebook, the Panda's *stretched* configuration has a collapsed ellipse (an arm at full reach can't move further outward).

## 5. Why VLA engineers care
- A VLA outputs end-effector deltas, and IK turns them into joint commands through $J^{+}$. Near singularities, small action errors become violent joint motions.
- Normalizing actions per dimension (S10) is a conditioning problem in disguise.
- LoRA (S13) is a low-rank matrix factorization, $\Delta W = BA$, so rank and SVD intuition carries straight over.

## Key takeaways
- SVD = rotate, stretch, rotate. Singular values are the stretch factors.
- Least squares solves $A^\top A x = A^\top b$; damping keeps it stable.
- The Jacobian's singular values measure how well a robot can move in each direction.

## References
- Strang, *Introduction to Linear Algebra*, Ch. 4 and 7
- Lynch & Park, *Modern Robotics*, §5.4 (manipulability)
- Kandapagari, *Action Models for Robot Learning*, §3.1, Appendix A
