# C8 · A Car That Drives Itself Through a City: a Mini Driving VLA

**Section:** S17 · **Type:** Capstone (reference solution, theory + measured results) · **Duration:** ~45 min
**Notebook:** `C8_city_driver.ipynb` · **Videos:** `outputs/C8_city_driver.mp4` (2:24), `outputs/C8_city_driver_linkedin.mp4` (90 s, 1.6×)
**Companion:** S16 notebook 150 runs NVIDIA's Alpamayo 2 Edge in the same city.

---

## 1. What a driving VLA is
Same three slots as every VLA (lecture 07):

| Slot | NVIDIA Alpamayo 2 Edge | Our mini driving VLA |
|---|---|---|
| **Input** | 6 cameras × up to 5 frames over 2 s, 20 past poses, "Turn right in 30m" | front camera × 2 frames (now, −0.4 s), speed, curvature, "turn left at the next intersection" |
| **Output** | chain-of-causation text + 6.4 s trajectory (64 poses, 10 Hz) | 6 sampled 2 s plans of (acceleration, curvature) + 1 of 6 reasons |
| **Training signal** | 115,000 h of human driving + 3.7M reasoning traces | 787 simulated expert drives (~2 h), with the expert's reason as a label |
| **Size** | ~2.65B parameters | 14.6M parameters |

Both use the **same action head**: flow matching (S08) over a sequence of **acceleration $a$ and curvature $\kappa$**,
integrated into a trajectory with the unicycle model:
$$v_{t+1} = v_t + a_t\,\Delta t, \qquad \theta_{t+1} = \theta_t + v_{t+1}\kappa_t\,\Delta t, \qquad p_{t+1} = p_t + v_{t+1}(\cos\theta_{t+1}, \sin\theta_{t+1})\,\Delta t$$
Why $(a, \kappa)$ and not waypoints? They are what a car's controller can actually execute, and the integration
guarantees a kinematically feasible, smooth path.

## 2. The world (MuJoCo, built in code)
- **City:** 4 × 4 intersections, 60 m apart, two-lane roads (10 m), crosswalks, a traffic light per approach, buildings, parks, trees. 180 m × 180 m.
- **Rendering trick:** MuJoCo draws every geom separately. 3,236 small geoms → **156 ms/frame** on a laptop. Merging the static scenery into 19 meshes (one per material): **22 ms**. The traffic lights keep **one** lit lamp per head that moves between red / yellow / green.
- **Traffic lights:** fixed cycle per intersection, random phase: green 12 s, yellow 3 s, all-red 2 s per axis.
- **Cars (24)** follow random routes with the **Intelligent Driver Model** (Treiber 2000):
$$a = a_{max}\Big[1 - \big(\tfrac{v}{v_0}\big)^4 - \big(\tfrac{s^*}{s}\big)^2\Big], \qquad s^* = s_0 + vT + \frac{v\,\Delta v}{2\sqrt{a_{max}b}}$$
  $s$ = gap to the nearest obstacle in the lane corridor ahead: a car, a pedestrian, or a red light's stop line (a standing obstacle).
- **Pedestrians (40)** walk the sidewalks, cross at crosswalks on the walk signal, and don't walk into moving cars.
- **Jaywalker:** in 30% of random episodes (and every "jaywalker" episode) a pedestrian steps off the kerb mid-block when the car is ~3.5 s away.
- **Kinematic vehicles:** all agents are mocap bodies moved by our code, like most driving simulators' traffic.

## 3. The expert (privileged)
- **Lateral:** pure pursuit on the lane centre of the route: $\kappa = \frac{2\sin\alpha}{L_d}$, look-ahead $L_d = \text{clip}(0.6v + 3, 3.5, 9)$ m.
- **Longitudinal:** IDM against everything in its corridor, a curvature speed limit $v \le \sqrt{a_{lat}/\kappa}$ (with comfortable braking toward it), red lights, and on left turns it waits for oncoming traffic (time-to-arrival < 5 s).
- **It says why:** the binding constraint is the label: cruise · turn · car · red · ped · yield.
- **Measured (150 episodes, 5 scenario types):** 149/150 routes completed, **0 incidents in 19.8 km**.

## 4. Data: noise injection (DART)
- In 60% of episodes, slowly varying (Ornstein–Uhlenbeck) noise is added to the **executed** steering and acceleration; the recorded label stays the expert's **clean** action. The data contains recoveries from small mistakes (S06 lecture 59–60).
- Episodes the expert fails are discarded: **787/800 kept**, 131,361 camera frames (5 Hz; every 3rd frame while waiting at a light).
- Recording time: **110 s** on 30 CPU cores with GPU rendering.
- Reason balance in the data: red 38%, cruise 25%, turn 19%, car 11%, pedestrian 6%, yield 2%, so training samples are weighted by $1/\sqrt{\text{count}}$ of their reason.

## 5. The model
```
frame now + frame 0.4 s ago ─ ResNet-18 (ImageNet) ─┐
speed, curvature ─────────────────────── MLP ───────┼─ context (512) ─┬─ flow head: 20 × (a, κ), 10 Euler steps, 6 samples
instruction ─ CLIP text (frozen), 4 paraphrases ────┘                 └─ reasoning head: 6 classes
```
- **Two frames:** one image can't tell a pedestrian standing on the kerb from one walking into the road.
- **Loss:** flow matching + 0.2 × cross-entropy on the reason.
- **Training:** 30,000 steps, batch 128, AdamW + one-cycle, bf16, augmentation (±6 px shifts, brightness/contrast). **16 min** on one RTX PRO 6000.
- **Overfitting, visible:** held-out flow loss was lowest at step 12k (0.148) and drifted up to 0.167 by 30k while training loss kept falling. Closed loop is what decides; early stopping is an exercise.
- **Execution:** every 0.1 s sample 6 plans, execute the first step of the one closest to the previous plan (temporal consistency, S11 114).

## 6. Results
**Open loop:** the reasoning head names the expert's reason on **96%** of held-out frames.

**Closed loop (150 episodes, same seeds as the expert):**

| Scenario | Mini VLA | 95% CI | Expert |
|---|---|---|---|
| random | 90% | 74–97% | 97% |
| the light turns red | 93% | 79–98% | 100% |
| a pedestrian steps out | 90% | 74–97% | 100% |
| unprotected left turn | 90% | 74–97% | 100% |
| right turn past a crosswalk | 87% | 70–95% | 100% |
| **overall** | **90%** | | **99%** |

**Incidents:** 11 in 19.1 km (**1.7 km per incident**; the expert: none in 19.8 km). Failure types: car collisions (6), pedestrian collisions (4), off-road (1), plus wrong turns (3) and red lights run (2). A separate evaluation run of the same checkpoint gave 92.7% and 2.8 km per incident: with 30 episodes per scenario, the CIs are ±12 points.

For scale: Alpamayo 2 Edge reports **15.15 km per at-fault incident** in NVIDIA's AlpaSim closed-loop simulator, on reconstructions of real drives.

## 7. Lessons
1. **The recipe transfers down in scale:** camera + instruction → sampled trajectories + a stated reason works with 14.6M parameters and 2 hours of driving, in one simulated city.
2. **Closed loop tells the truth.** 96% reason accuracy and low validation loss still meant 1 incident every 1.7 km.
3. **Reasoning as a label ≠ reasoning as a cause.** Our reasoning head is trained to *name* the expert's reason; nothing forces the plan to follow from it (lecture 150: faithfulness).
4. **Renderer engineering matters:** merging geometry made data collection 7× cheaper on a laptop.
5. **Our world is easy:** one layout, clear weather, kinematic traffic, perfect rendering. Generalization is the next exercise, not a claim.

## Extensions
1. Ablate the noise injection (`noise_frac=0`) and retrain.
2. One frame vs two: which scenario suffers most?
3. Six cameras, like Alpamayo: does the unprotected left turn improve?
4. Test on a new city layout (`city_seed`), never seen in training.
5. Night: dim the sun, turn on headlights. What breaks first?
6. Early stopping on held-out loss vs on closed-loop success: do they pick the same checkpoint?

## References
- NVIDIA, *Alpamayo 2 Edge* model card and code (2026): huggingface.co/nvidia/Alpamayo2-Edge, github.com/NVlabs/alpamayo2-edge
- Treiber, Hennecke, Helbing 2000, *Congested traffic states in empirical observations and microscopic simulations* (IDM)
- Coulter 1992, *Implementation of the Pure Pursuit Path Tracking Algorithm*
- Laskey et al. 2017, *DART: Noise Injection for Robust Imitation Learning*
- Bansal et al. 2018, *ChauffeurNet*; Codevilla et al. 2018, *End-to-end Driving via Conditional Imitation Learning*
- Lipman et al. 2023, *Flow Matching for Generative Modeling*
