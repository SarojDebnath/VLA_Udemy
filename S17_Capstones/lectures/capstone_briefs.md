# S17 · Capstone Projects

Pick at least one. Each capstone has a brief, a starter or reference notebook, and a grading rubric. Every claim must come with a Wilson interval (S12).

---

## C1 · Language-conditioned table sorter (🟡) · `C1_language_sorter.ipynb` (reference solution)
**Task:** "sort all the cubes into the yellow tray": a long-horizon instruction never seen in training.
**Pipeline:** System 2 planner (rule-based reference; swap in the S11 SmolVLM planner) → S09 FiLM language-conditioned chunk policy (System 1) → S15 safety shield (wrist F/T, workspace, velocity) → evaluation: cubes placed per episode, full-sort success with a CI, shield interventions.
**Reference result:** both cubes sorted in **17/20** episodes (95% CI 64–95%), 1.85 of 2 cubes placed on average, with the shield active. The showcase video (`outputs/C1_sorter_showcase.mp4`) is the format to submit.
**Extensions:** train the policy on the visually randomized dataset (S13 notebook 131) and report the robustness matrix (S12 notebook 123); replace the planner with the VLM and report the decomposition accuracy.
**Rubric:** pipeline runs end to end (30%), honest evaluation with intervals and failure analysis (30%), one measured improvement over the reference (30%), video and README (10%).

## C2 · Bimanual ALOHA transfer-cube with ACT (🟡) · `C2_aloha_transfer_act.ipynb` (reference solution)
**Task:** the left ViperX arm picks a randomly placed cube, carries it to a randomized handover point, the right arm grasps it from the side, and the left arm releases.
**Provided:** `vlacourse.aloha`: a joint-space env at 25 Hz with 14-D actions (6 joints + gripper per arm), a phased, closed-loop scripted handover expert (100% success), per-arm IK with an approach-direction term.
**Reference:** 100 demos → ACT (CVAE transformer, **chunk 50 = 2 s**, overhead + right-wrist cameras, temporal ensembling, 60 epochs) → **17/20** test episodes (Wilson CI 64–95%).
**What it took (measured, 20 test episodes each):**

| Chunk | Demos | Epochs | Success |
|---|---|---|---|
| 20 (0.8 s) | 50 | 20 | 0/20 |
| 20 | 50 | 100 | 5/20 |
| 20 | 200 | 40 | 4/20 |
| **50 (2 s)** | 50 | 100 | 14/20 |
| **50 (2 s)** | **100** | 60 | **17/20** |

With short chunks the policy grasped and lifted the cube but failed the handover. The expert works in timed phases (25 steps each, several of them pure waits for a gripper), and a 0.8 s chunk can't see where it is inside a 1 s wait. **Match the chunk length to the timescale of your demonstrations' pauses**; more data and longer training didn't fix it, a longer chunk did. (ACT's paper uses 100-step chunks at 50 Hz: the same 2 s.)
**Extensions:** compare with Diffusion Policy / flow on the same data; ablate the wrist camera; vary the number of demos (10/25/50); add a simulated-human operator (S06) and measure the effect of demo quality.
**Rubric:** expert data with randomization (20%), trained policy with a CI (40%), one ablation with a clear plot (30%), video (10%).

## C3 · SO-101 sim-to-real starter kit (🟡, optional hardware) · `C3_so101_starter.ipynb`
**Why it's hard (and instructive):** the SO-101 is 5-DoF (no full 6-D pose), its jaw axis is the gripper site's **x** axis (not z), its reach is about 0.4 m, and it grasps with a rotating jaw that sweeps an arc, so the pre-grasp offset matters. Our first attempts pushed the cube away instead of grasping it.
**Steps:** (1) get a reliable sim grasp; (2) script 100 randomized demos in joint space; (3) store them as LeRobot v3; (4) train ACT or SmolVLA; (5) optionally record 50 real episodes with a leader-follower SO-101 (`lerobot-record`) and compare sim-only vs sim+real fine-tuning on real trials.
**Rubric:** working sim grasp with a success rate (30%), trained policy in sim with a CI (30%), sim-to-real comparison *or* a domain-randomization study (30%), documentation (10%).

## C4 · Humanoid VLA: "pick up the red cube" on a Unitree G1 (🔴 hard) · `C4_g1_humanoid_vla.ipynb` (reference solution)
**Robot:** Unitree G1 with Dex3 three-finger hands (Menagerie): two 7-DoF arms, head camera + one camera on each wrist.
**Scope, stated in the notebook:** the pelvis is fixed (no balance control), and grasping uses a contact-triggered weld (as in Habitat/BEHAVIOR): a real Dex3 grasp is extension 1.
**Pipeline:** scripted two-arm expert (picks with the hand on the cube's side; 71% of its attempts succeed, successes kept) → 1,000 demos, 535 pick / 465 place → CLIP instruction embedding → ACT (instruction as a token) **vs** FiLM (instruction modulates vision), 20-step chunks → evaluation with a "went for the named cube" metric separate from grasp success.
**Reference results (40 episodes):**

| Policy | Execution | Success | Went for the named cube |
|---|---|---|---|
| ACT | temporal ensembling | 28% | 85% |
| ACT | re-plan every 4 steps | 35% | 82% |
| FiLM | temporal ensembling | 38% | 100% |
| **FiLM** | **re-plan every 4 steps** | **57%** (pick 74%, place 35%) | **98%** |

Grounding test (same layout, three instructions): went for the named cube **30/30** (chance ≈ 10), picked it up 13/30. Unseen phrasings: still 100% to the right cube, 22% success.
**What it took (each step measured):** head camera only → ~30% and a grasp-based grounding score at chance level; balanced pick/place data → no better; two wrist cameras + 1,000 demos → the trace showed it goes for the right cube with the right hand but closes 2–4 cm off; re-planning instead of temporal ensembling → 57%. Lessons: FiLM grounds language better than concatenation (S09, now on a humanoid); **separate "chose correctly" from "executed precisely"** in your metrics; the execution loop is part of the policy.
**Extensions:** a real Dex3 grasp (no weld); hand-overs when cube and bowl are on opposite sides; an LLM planner for "clear the table"; free the pelvis and balance (S16 151); relative-target labels or wrist-camera visual servoing for precision.
**Rubric:** working pipeline (25%), evaluation that separates choice from execution, with CIs (35%), one measured improvement in precision (30%), video (10%).

## C5 · Tower builder: planner → stacking policy → task-level feedback (🟡) · `C5_tower_builder.ipynb` (reference solution)
**Task:** "build a tower with the blue cube at the bottom, the red cube in the middle and the green cube on top" (four phrasings, two of which name the tower top-down). Three cubes, two precise stacks, the second onto a cube that is itself balanced.
**Pipeline:** 400 scripted towers (one episode per stacking step, `vlacourse.tower`) → S09 FiLM language-conditioned chunk policy for "stack X on Y" → System 2 LLM that outputs the tower order as a JSON list → C1 safety shield → **task-level feedback:** after every step the executive measures the tower and asks for "the next cube on top of what is actually built" (a fallen cube is retried, a knocked level is rebuilt).
**Reference results (20 episodes, plans from the 7B planner, no fallback):** 13/20 full towers open loop → **17/20 with task-level feedback** (2 episodes recovered a failed stack); per-step success 90% (level 1), 80% (level 2). Repeated runs vary by a few episodes (12–13 → 14–17): report intervals, not single numbers. Planner: SmolLM2-1.7B **50%** (100% on the two phrasings that list the colours bottom-up, **0%** on the two that must be reversed: it copies the mention order) vs **Qwen2.5-7B 100%**. 
**Lessons:** a closed loop at the *task* level is cheap and catches what the motor policy misses; planning needs language capacity (the same lesson as S11 115).
**Extensions:** replace the privileged tower check with a camera classifier; DAgger data on failed level-2 stacks; a fourth cube; negations in the request.
**Rubric:** pipeline end to end (30%), honest evaluation incl. planner accuracy and per-step success (30%), one measured improvement (30%), video (10%).

## C6 · Data engine: 10 human demonstrations → 500 generated (🟡) · `C6_data_engine.ipynb` (reference solution)
**Task:** make a working pixel policy from only **10** (simulated-)human demonstrations, MimicGen-style (Mandlekar et al. 2023).
**Pipeline:** simulated human operator (S06 55: lag, tremor, hesitations; 29% of its attempts succeed) records 10 demos with object poses → split each demo into a cube-relative and a tray-relative segment → move the segments rigidly with the objects into a new layout, connect them with straight moves, replay in simulation, keep only successes (`vlacourse.mimicgen`) → S06 project 62's full recipe (16-step chunks, two cameras) trained on each dataset with the same number of gradient steps.
**Reference results (50 new layouts):**

| Training data | Success |
|---|---|
| the 10 human demos | **6%** |
| 100 generated from them | **50%** |
| 500 generated from them | **70%** |
| (reference: 450 scripted-expert demos, S06 62) | 100% |

Generation yield: 500 of 502 replays succeeded. The generated data inherits the human's tremor and style, which is why it doesn't reach the scripted expert's 100%.
**Extensions:** generate from 1/3/10 sources; stacking on the "full" scene; filter generated demos by speed; feed in your own keyboard-teleop demos.
**Rubric:** generation with a reported yield (30%), policy comparison with CIs (40%), one ablation (20%), video (10%).

## C7 · Mobile fetch: "bring the green cube to the box" (🟡) · `C7_mobile_fetch.ipynb` (reference solution)
**Robot:** Stanford TidyBot (Menagerie): holonomic base, Kinova Gen3 arm, Robotiq 2F-85 gripper; real friction grasps.
**Pipeline:** System 2 (a parser; swap in the S11 LLM) → **drive** to a fixed parking spot at the table (P control with speed/acceleration limits) → **pick** with a *learned* FiLM language-conditioned chunk policy (mast + wrist cameras) → lift to a carry pose → **drive** to the box → **drop**. Every stage is checked, so failures are attributed.
**Reference results (30 episodes):** **23/30 delivered** (CI 59–88%); drive 100%, pick 87%, wrong cube 3%, drive 100%. Grounding test (same scene, three colours): **20/24** named cubes picked.
**The lesson from the first run:** the TidyBot's stock base camera sits *below the tabletop* and the arm's home pose blocks the view. The policy was blind, went straight down at the table's centre, and picked 10%. A camera mast and an arm "ready" pose that keeps the table in view took it to 87%. **Always look at what your policy sees.**
**Also learned:** carrying a cube in the folded home pose whips the wrist around and throws it 1.7 m; carry with the gripper still pointing down.
**Extensions:** drive through the 42b crowd with MPC; find the table from the camera; two cubes per request; one end-to-end policy that also outputs base velocities.
**Rubric:** pipeline with per-stage success (30%), grounding test (20%), one measured improvement (40%), video (10%).

## C8 · A self-driving car in a MuJoCo city (🟡) · `C8_city_driver.ipynb` (reference solution) · full notes: `C8_city_driver.md`
**World:** a procedural MuJoCo city (16 intersections with traffic lights, 24 IDM traffic cars, 40 pedestrians, a jaywalker); the ego car is controlled by acceleration and curvature at 10 Hz.
**Pipeline:** privileged expert (pure pursuit + IDM, says *why*) → 800 drives with DART noise → a mini driving VLA (2 camera frames + speed + CLIP instruction → flow-matching head: 6 sampled 2 s plans of (a, κ); reasoning head: 1 of 6 reasons) → closed-loop evaluation per scenario.
**Reference results (150 episodes):** model **90%** routes completed (expert 99%), **1.7 km per incident** (expert: 0 incidents in 19.8 km); reasoning head 96% on held-out frames.
**Lesson:** open-loop numbers (96% reasons, low validation loss) don't predict the 11 incidents; only closed loop does.
**Extensions:** no noise injection; one frame vs two; six cameras; an unseen city layout; night driving; early stopping.
**Rubric:** closed-loop table with CIs per scenario (30%), one ablation with a measured effect (40%), failure analysis from videos (20%), video (10%).

---

## What to submit (all capstones)
1. Notebook(s) that run top to bottom.
2. `results/*.json` with every number in your report.
3. A 2-page report: three-slot description of your system (lecture 07), evaluation protocol, results with Wilson CIs, failure analysis, one limitation you'd fix next.
4. A ≤ 90 s video.
