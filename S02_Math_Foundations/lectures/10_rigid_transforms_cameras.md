# 10 · Rigid Transforms, Frame Trees, and the Camera Model

**Section:** S02 · **Type:** Math · **Duration:** ~30 min
**Companion notebook:** `10_rigid_transforms_cameras.ipynb`

## Learning objectives
- Compose and invert homogeneous transforms in $SE(3)$.
- Rebuild MuJoCo's forward kinematics as a product of parent-to-child transforms.
- Project 3D points to pixels with the pinhole model, and back-project depth to a point cloud.

---

## 1. Homogeneous transforms
A pose = rotation + translation, packed in a $4\times4$ matrix:
$$T = \begin{bmatrix} R & p \\ 0 & 1\end{bmatrix} \in SE(3),\qquad
T^{-1} = \begin{bmatrix} R^\top & -R^\top p \\ 0 & 1\end{bmatrix}$$
Notation: $T_{AB}$ is the pose of frame $B$ expressed in frame $A$. Composition chains frames:
$$T_{AC} = T_{AB}\,T_{BC}$$
The subscripts cancel like units, so write them out in code (`T_w_hand @ T_hand_cube`) and most frame bugs disappear.

## 2. Frame trees and forward kinematics
A robot is a tree of bodies. Each body has a fixed offset from its parent (`body_pos`, `body_quat` in MuJoCo) and joints that add a motion:
$$T_{w,\text{hand}}(q) = \prod_{i \in \text{chain}} T_{\text{offset},i}\; T_{\text{joint},i}(q_i)$$
For a hinge about axis $\hat a$: $T_{\text{joint}} = \begin{bmatrix}\exp([\hat a]_\times q_i) & 0\\0&1\end{bmatrix}$.
The notebook rebuilds this product by hand and checks that it matches `data.xpos` / `data.xmat` to $10^{-9}$.

## 3. Relative poses
"Where is the cube relative to the gripper?" is the question every grasping policy answers:
$$T_{\text{hand},\text{cube}} = T_{w,\text{hand}}^{-1}\,T_{w,\text{cube}}$$
Wrist cameras make this relation directly visible to a policy, which is one reason wrist views help VLAs generalize.

## 4. The pinhole camera
A point $X_c = (x,y,z)$ in the camera frame projects to pixel $(u,v)$:
$$\begin{bmatrix}u\\v\\1\end{bmatrix} \sim K X_c,\qquad K = \begin{bmatrix} f & 0 & c_x\\ 0 & f & c_y \\ 0 & 0 & 1\end{bmatrix},\quad f = \frac{H/2}{\tan(\text{fovy}/2)}$$
Full projection from world points: $\tilde u \sim K\,[R_{cw}\,|\,t_{cw}]\,\tilde X_w$.

**MuJoCo convention:** cameras look along their local $-z$, with $+y$ up. Image convention is $+z$ forward and $+y$ down, so we flip: $R_{\text{cv}} = R_{\text{mj}}\,\mathrm{diag}(1,-1,-1)$.

## 5. Back-projection: depth to point cloud
With a depth image $z(u,v)$:
$$x = \frac{(u-c_x)\,z}{f},\quad y = \frac{(v-c_y)\,z}{f},\qquad X_w = T_{wc}\,X_c$$
MuJoCo's `Renderer` depth mode returns metric depth along the optical axis, which is exactly what this formula needs.

## 6. Where this shows up in VLAs
- **Action frames:** an end-effector delta in the *base* frame vs the *tool* frame are different actions. Datasets mix both, and Open X-Embodiment normalization has to reconcile them.
- **Camera placement:** a VLA trained with one camera pose often fails when the camera moves a few centimeters (S12 perturbation tests).
- **3D-aware policies** (point clouds, depth tokens) use back-projection directly.

## Key takeaways
- $T_{AC} = T_{AB}T_{BC}$; invert with $R^\top$ and $-R^\top p$.
- Forward kinematics is a product of offset and joint transforms down the tree.
- The pinhole model $K[R|t]$ maps 3D to pixels; depth plus $K^{-1}$ maps pixels back to 3D.

## References
- Hartley & Zisserman, *Multiple View Geometry*, Ch. 6
- Lynch & Park, *Modern Robotics*, Ch. 3–4
- MuJoCo docs: *Modeling → Cameras*, Python `mujoco.Renderer`
