# 54 · Why Imitation Is the Dominant Signal in Modern Robotics

**Section:** S06 Imitation Learning · **Type:** Theory · **Duration:** ~20 min

## Learning objectives
- Compare demonstrations, rewards and derivations as training signals for manipulation.
- Know the main ways of collecting demonstrations and their trade-offs.
- Set up the supervised-learning view of behavior cloning used in the rest of S06.

---

## 1. Three ways to get an action model
| Signal | Example | Bottleneck in manipulation |
|---|---|---|
| Derivation (classical) | IK + planner + scripted grasps (S04) | needs object models, brittle to novelty |
| Reward (RL) | PPO/SAC (S05) | exploration: random actions almost never complete a pick-and-place; reward design; sample cost on hardware |
| **Demonstration (imitation)** | a human or expert shows the task | collecting data takes human time; compounding error |

For manipulation, demonstrations win in practice: a person can show a task in a minute that RL would need millions of trials to discover. Every VLA in this course (RT-1, OpenVLA, Octo, π0, SmolVLA) is trained primarily by imitation.

## 2. Behavior cloning is supervised learning
Given $\mathcal D = \{(o_i, \ell_i, a_i)\}$:
$$\theta^* = \arg\min_\theta \sum_i \ell\big(\pi_\theta(o_i, \ell_i),\, a_i\big)$$
with ℓ = MSE (Gaussian MLE), cross-entropy (binned actions), or a denoising loss (S08). Everything from S02 applies: normalization, train/val split, the debugging checklist.

## 3. Where demonstrations come from
| Method | Pros | Cons | Used by |
|---|---|---|---|
| Kinesthetic teaching | intuitive | only works on backdrivable arms; human hand in the image | early LfD |
| Keyboard / SpaceMouse teleop | cheap | slow, unnatural for 6-DoF | robomimic, LIBERO |
| VR teleop | natural 6-DoF | latency, motion sickness | RT-1, DROID |
| **Leader-follower arms** | high quality, bimanual, intuitive force feel | hardware per robot | ALOHA, SO-100/101 (LeRobot) |
| Scripted / planner expert | free, unlimited, perfect labels | unnatural motion, needs privileged state | our data engine, MimicGen |
| Human video (no robot) | massive scale | no action labels, embodiment gap | V-JEPA 2, egocentric datasets (S16) |

## 4. What makes good demonstrations
- **Coverage beats volume:** vary object positions, lighting and approaches; 50 diverse demos beat 200 copies.
- **Consistent conventions:** same action frame, gripper sign and control rate everywhere.
- **Recovery behavior:** include corrections (DART, DAgger) or the policy never learns to recover.
- **Multimodality is real:** different people (or a careful scripted expert) solve tasks differently, and the policy class must represent that (S08).
- **Reset discipline:** re-randomize the scene between episodes.

## 5. The rest of S06
55 teleop and a simulated human operator → 56 the LeRobotDataset format → 57/58 BC from state and pixels → 59 covariate shift → 60 DAgger and DART → 61 IRL/GAIL → 62 project ablations.

## Key takeaways
- Demonstrations sidestep RL's exploration problem; that's why imitation dominates manipulation.
- BC = supervised learning on (observation, action) pairs.
- Data quality (coverage, conventions, recoveries) matters more than quantity.

## References
- Argall et al. 2009, *A survey of robot learning from demonstration*
- Zhao et al. 2023 (ALOHA/ACT); Khazatsky et al. 2024 (DROID)
