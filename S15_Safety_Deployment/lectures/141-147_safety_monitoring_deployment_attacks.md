# 141–147 · Evaluation, Safety, and Deployment

**Section:** S15 Safety and Deployment · **Type:** Theory + labs · **Duration:** ~65 min
**Notebooks:** `142_safety_shield` (covers 142–143), `144_runtime_monitor`, `145_logging_rollback`, `146_adversarial_attacks`

---

## 141 · Safety as a layer, not a property
No learned policy can be *certified* safe: its behavior on unseen inputs isn't specified. Safety therefore comes from **layers around the policy** whose behavior *can* be stated:
```
VLA policy ─► [shield: force / workspace / velocity / keep-out zones] ─► low-level controller (impedance, torque limits) ─► robot
                ▲ runtime monitors (uncertainty, OOD, failure predicates) ─► alerts / stop / fallback
                ▲ logging ─► dashboards ─► A/B evaluation ─► rollback
```

## 142–143 · Shields and fault injection
`Shield` checks every action against cheap, network-independent predicates:
- **force:** wrist F/T magnitude below 25 N unless the action moves away (up),
- **workspace:** the next target stays inside a box,
- **velocity:** per-step displacement ≤ 4 cm.
On violation it returns a **decelerate-and-hold** action ("hold position" is well defined on a moving robot, "do nothing" isn't), lifting slightly on force violations. Every intervention is logged.

Fault injection (notebook 142): pushing into the table, scale ×5 (the wrong-`unnorm_key` bug from S10), sign flip, random spikes. With and without the shield we compare success, unsafe steps and peak force. The shield **cannot make a broken policy succeed**; it bounds the damage. Its guarantees are stated over *sensor values*, not over the network.

**Measured (15 episodes per row, limits: 25 N, workspace box, 6 cm/step):**
| Fault | Shield | Success | Unsafe steps / episode | Peak force |
|---|---|---|---|---|
| none | off / on | 100% / **100%** | 0 / 0 | 10.5 N / 10.5 N |
| push into table | off / on | 0% / 0% | **185 → 0** | **472 N → 9 N** |
| scale ×5 | off / on | 0% / 0% | 0 / 0 | 12.6 N / 9.1 N |
| sign flip | off / on | 0% / 0% | 36 → 12 | 9.8 N / 9.8 N |
| random spikes | off / on | 80% / **100%** | 8 → 0 | **561 N → 10.5 N** |

Read it row by row:
- **A shield must be invisible to a healthy policy.** Our first version used a workspace ceiling below the robot's own home pose and a speed limit below the action limit, and it blocked 100% of nominal episodes. Always run the no-fault row first.
- It removes force spikes completely, and for intermittent faults it even raises success, because the bad steps are replaced by "hold".
- It can't fix systematic faults (scale, sign flip). Those need the runtime monitor (144) and a rollback (145).
- The sign flip still produces 12 unsafe steps: the shield only checks what it measures, and moving *away* inside the box is "legal".

## 144 · Runtime monitors
- **Failure-drill predicates** (S10 notebook 100): action-magnitude statistics that flag known bugs.
- **Uncertainty signals:** for generative heads, draw K action chunks and measure their spread. Out-of-distribution inputs (camera moved, dim light) often raise the spread before failure. Notebook 144 tests this on our flow policy (20 episodes per condition):

| Condition | Success | Mean sample spread |
|---|---|---|
| nominal | 100% | 0.040 |
| camera moved 6 cm | 90% | 0.039 |
| dim light (×0.5) | 75% | **0.035** |
| grey table | 75% | 0.036 |

**The spread does not rise under the perturbations that cause failures. If anything it drops.** This policy fails *confidently*: the visual change shifts all K samples together instead of scattering them. That's the honest lesson. Sample spread measures disagreement *within the model*, not distance from the training data, so it can miss exactly the out-of-distribution failures you care about. Complement it with input-side OOD detectors (feature-distance to the training set), task-progress checks and VLM monitors, and always validate any monitor's ROC on real failures before trusting it.
- **VLM-as-monitor (Code-as-Monitor and related):** a VLM checks constraints and step-level success from images and can veto or trigger a recovery.

## 145 · Logging, alerting, rollback
- Log every episode: inputs digest, actions, interventions, outcome, timing (JSONL is enough to start).
- **Canary deployment:** route a fraction (20%) of episodes to the candidate.
- **Sequential decision rule:** a Bayesian A/B with Beta posteriors; roll back when $P(p_{new} < p_{old}) > 0.95$. It stops early on clear regressions without waiting for hundreds of episodes.
- Keep the previous checkpoint *and* its normalization statistics deployable at all times.

**Measured (notebook 145):** current policy = expert with 4 mm hand noise, candidate = 12 mm noise, 20% canary traffic. The rule fired a **rollback at episode 20**, after only **5** canary episodes. Over the full simulated log (which keeps running for comparison), the candidate reached 7/16 (44%, and slow: 178 steps/episode) versus 64/64 for the current policy. The stopping rule caught it after a third of that exposure.

## 146 · What we still can't certify: attacks at test time
- **Sensor perturbations:** pixel noise and lighting changes (the S12 suite).
- **Adversarial patches:** a small optimized sticker in the scene, learned by gradient ascent on the policy's action deviation, can derail a policy that ignores a *random* patch of the same size and place (notebook 146).

  **Measured (chunked-regression policy, "single" scene, 30 episodes each):**
  | Input to the front camera | Success |
  |---|---|
  | clean | 100% |
  | Gaussian pixel noise σ = 0.03 | 100% |
  | random patch | 97% |
  | **adversarial patch, same size and place** | **0%** |

  The policy shrugs off noise and a random sticker, so it *looks* robust. One optimized patch takes it from 100% to 0%. Robustness to random perturbations says nothing about robustness to adversarial ones. The video shows the same episode with clean vs attacked input; the scene is identical, and only the pixels the policy sees differ.
- Attacks on VLA language and reasoning: adversarial instructions, perturbed chain-of-thought, typographic attacks on the VLM.
Defenses: augmentation (random shift, color jitter), adversarial training, input sanitization, ensembles, and the shield, which bounds the consequences regardless of the cause.

## 147 · Training-time attacks: poisoning and backdoors
- **Data poisoning:** a small fraction of manipulated demonstrations degrades or biases behavior. Robot datasets are crowdsourced (LeRobot Hub, OXE), so provenance matters.
- **Backdoors** (BackdoorVLA, AttackVLA and related work): demonstrations containing a visual *trigger* (a sticker, an object) paired with malicious actions. The policy behaves normally until the trigger appears; reports show success with only a few percent poisoned data.
- **Mitigations:** dataset provenance and hashing, outlier detection on (observation, action) pairs, fine-tuning on trusted data, trigger-scanning evaluation suites, and the runtime layers above.

## Key takeaways
- Safety is a stack of verifiable layers around an uncertifiable policy.
- Monitor uncertainty and known failure signatures; log everything; roll back with a sequential test.
- Test-time and training-time attacks are real; robustness has to be measured, not assumed.

## References
- Alshiekh et al. 2018 (safe RL via shielding); Brown et al. 2017 (adversarial patch)
- Code-as-Monitor (2025); BackdoorVLA / AttackVLA (2025)
