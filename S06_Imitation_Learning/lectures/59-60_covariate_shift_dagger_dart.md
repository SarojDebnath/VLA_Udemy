# 59–60 · Covariate Shift, DAgger, and DART

**Section:** S06 · **Type:** Math + lab · **Duration:** ~35 min
**Companion notebooks:** `57_bc_from_state`, `60_dagger_and_dart`

## Learning objectives
- Derive why behavior cloning's error grows as $O(\epsilon T^2)$.
- Explain DAgger as a reduction to online learning and why it achieves $O(\epsilon T)$.
- Compare DAgger with noise-injection (DART) in practice.

---

## 1. The offline/online gap (notebook 57)
State-based BC on TableTop (200 expert episodes, 8,442 pairs) reaches a validation MSE of 0.07, yet closed-loop success is only **50%** (CI 40–60%): pick 97%, stack 34%, place 15%. **Every** failure runs into the 200-step limit: a small error moves the gripper or cube into a configuration the expert never visited, and from there the policy wanders until time runs out. The longer the task, the worse it gets, exactly as the bound below predicts. This is lecture 05's covariate shift on a real task.

## 2. The bound (Ross & Bagnell 2010)
Let $\epsilon$ = probability the learner disagrees with the expert *under the expert's state distribution*, with cost bounded by 1 per step over horizon $T$:
- once a mistake is made, the learner may be off-distribution for the rest of the episode,
- summing over the time of the first mistake gives
$$J(\hat\pi) \le J(\pi^*) + T^2\epsilon$$
Supervised learning with i.i.d. test inputs would give $T\epsilon$. The extra factor of $T$ is the price of the learner choosing its own future inputs.

## 3. DAgger: Dataset Aggregation (Ross, Gordon & Bagnell 2011)
```
D ← expert demonstrations;  π₁ ← BC(D)
for i = 1..N:
    run πᵢ' = βᵢ·π* + (1−βᵢ)·πᵢ   in the environment, visiting states s
    label every visited s with the expert: a* = π*(s)
    D ← D ∪ {(s, a*)};  πᵢ₊₁ ← BC(D)
```
- Training data now comes from the **learner's own state distribution**, which removes the mismatch.
- Guarantee (no-regret online learning): $J(\hat\pi) \le J(\pi^*) + uT\epsilon_N + O(1)$, **linear in T**.
- β mixes in the expert early on (we use $\beta_i = 0.5^i$) to keep rollouts sane.

**Practical cost:** an expert must label arbitrary states, including weird ones. Humans find this hard, which motivated HG-DAgger, interventions and "corrections" interfaces. Scripted experts can find it hard too, and that bit us (next box).

**Box: an expert that can be queried anywhere.** Our data-collection expert is a *waypoint state machine*: it remembers which waypoint it's on. Ask it "what would you do here?" in a state the learner wandered into, and its answer depends on its internal counter, not on the state. DAgger labels from such an expert are wrong. For notebook 60 we wrote `ReactiveExpert`, a **stateless funnel controller**: its action is a function of the current simulator state only (descend in proportion to how well you're aligned; close when the gripper is at the cube; carry; release over the tray). Rule: **a DAgger expert must be a function of the state.** For a human, that means "show me what you would do from here", not "continue your demo".

## 4. DART: Disturbances for Augmenting Robot Trajectories (Laskey 2017)
Instead of querying the expert on learner states, **inject noise while the expert demonstrates**: execute $a^* + \eta$ but record the clean $a^*$. The expert's closed-loop corrections show up in the data as recovery examples. It's cheap and works with humans: the noise is added to the commands, and the human corrects naturally. `vlacourse.collect(dart_noise=…)` implements it.

## 5. What notebook 60 measures
Pick task, privileged-state MLP, success over 60 new layouts. Two lessons, both measured.

**Lesson 1: observe what the expert's decision depends on.** The expert decides from the *commanded* gripper pose (where the arm is being told to go); the learner only saw the *achieved* pose (where it is). On the expert's own path the two differ by a constant lag, so the labels look consistent; off that path they don't. Same 30 demos, same network:

| Learner input | BC success |
|---|---|
| achieved pose only (20-D) | 50% |
| + commanded-pose offset (23-D) | **93%** |

Before collecting more data, check that the learner *can* see what the labels depend on.

**Lesson 2: DAgger pays off when execution is imperfect.** We compare BC with more demonstrations against DAgger (3 demos, then 6 rounds of 4 learner-driven episodes relabelled by the expert, β = 0.5ⁱ), on a perfect arm and on a **sloppy arm** whose every commanded move lands with 12 mm random error:

| | Expert-labelled pairs | Perfect arm | Sloppy arm |
|---|---|---|---|
| BC, 15 demos | 248 | 88% | **20%** |
| BC, 30 demos | 500 | 93% | 68% |
| BC, 70 demos | 1,166 | 100% | 93% |
| **DAgger** (3 demos + 6 rounds) | 976 | 85% | **100%** |

- On a perfect arm, clean demonstrations are enough: BC with 70 demos is perfect, and DAgger is no better (its perfect-arm curve is noisy, 63–95% across rounds).
- On a sloppy arm, BC collapses unless it gets a lot more data (15 demos: 20%), because the arm keeps pushing it off the expert's path. DAgger's labels come *from* those off-path states, so it is unaffected: 100% with fewer labels than BC-70.
- Real robots are always somewhat sloppy (backlash, compliance, latency, objects that slide). That's where DAgger-style data earns its cost.

The video replays BC-70 and DAgger on the same episode with *identical* actuator errors.

**DART (exercise in the notebook).** Implement DART by adding noise to the expert's executed actions while recording its clean labels. In our runs it helped on the sloppy arm but *collapsed on the perfect arm*. The notebook asks you to find out why: the learner learned to rely on noise to nudge it into the grasp pose, and without noise it stops just short. Noise levels must match the deployment errors, not exceed them.

## 6. Connection to VLAs
- Large VLA datasets are mostly clean teleop, so recovery behavior is rare and fine-tuned VLAs drift.
- Modern remedies: intervention data (human takes over, the correction is recorded), RL fine-tuning on the policy's own rollouts (S16), and action chunking, which reduces the number of decisions $T$ (S08).

## Key takeaways
- BC's cost bound is $O(\epsilon T^2)$ because the learner's errors change its inputs.
- DAgger trains on the learner's own states → $O(\epsilon T)$, but needs an interactive expert.
- DART gets recovery data cheaply by perturbing the expert, but only if the noise matches what deployment will do.
- A DAgger expert must be a function of the state, and the learner must observe what the expert's decisions depend on.
- Measured: on a sloppy arm, DAgger reaches 100% where BC with 15 demos gets 20%.

## References
- Ross & Bagnell 2010; Ross, Gordon & Bagnell 2011 (DAgger)
- Laskey et al. 2017 (DART); Kelly et al. 2019 (HG-DAgger)
