# 66–69 · Decision Transformer Labs: Return Conditioning, Its Limits, and the Price of Action Tokens

**Section:** S07 Sequence Models Meet Control · **Type:** Labs + results · **Duration:** ~35 min
**Notebooks:** `66_decision_transformer`, `67_return_knob`, `68_stitching`, `69_action_tokenization_bins`

## Learning objectives
- Train a Decision Transformer (DT) on offline data and condition it on a target return.
- Read the "return knob" curve and explain why it breaks outside the data.
- Show the stitching failure that separates sequence modeling from dynamic programming.
- Measure what discretizing actions into tokens costs, the design decision every token-based VLA makes.

---

## 1. The setup (notebook 66)
- **Task:** MuJoCo pendulum swing-up (continuous torque, 200 steps, reward ≤ 0).
- **Offline data:** 300 episodes from SAC checkpoints at different training stages plus noise. Returns range from **−1903 to −1** (mean −821). It's a mixed-quality dataset, like real robot logs.
- **Model:** a causal transformer over interleaved (return-to-go, state, action) tokens, context K = 20, 622k parameters, a learned timestep embedding. Trained in about 4 minutes.
- **Inference:** feed the desired return-to-go, act, subtract each reward from the return-to-go, repeat.

**Result:** asking for the dataset's mean (−821) gives **−811**. The model reproduces "average behavior" on demand. Asking for the best return in the data (−1) gives only **−572**. It's better than average, but nowhere near the request.

## 2. The return knob (notebook 67)
| Requested return | −1903 | −1631 | −1359 | −1088 | −816 | −544 | −272 | **0** |
|---|---|---|---|---|---|---|---|---|
| Achieved | −1772 | −1744 | −1662 | −1397 | −741 | −592 | −572 | **−1292** |

- **Inside the data (about −1400 to −550):** the knob works and achieved return tracks the request.
- **At the top of the range:** it saturates around −570. Few trajectories in the data are that good, so the model has seen little of that behavior.
- **Outside the data (requesting 0):** performance *collapses* to −1292. The model was never shown a return-to-go of 0 at the start of an episode, so the input is out of distribution and it behaves arbitrarily.

**Teaching point:** a return-conditioned policy interpolates between behaviors it has seen. It doesn't *optimize*. You can't ask it for something better than its data and expect to get it. The same lesson appears with the Go1 joystick (S05: 1.5 m/s → stops) and with out-of-range action bins (S10).

## 3. Where stitching fails (notebook 68)
- **Toy graph:** start S → hub X → dead end (return 1), and a different start Z → X → goal (return 1).
- **The trap:** the best behavior from S is S → X → goal (return 2), but **no single trajectory in the data contains it**.

| Method | Return from S |
|---|---|
| Offline Q-learning | **2.0**: it stitches through X because Q(X, goal) = 0.95 regardless of how X was reached |
| Decision Transformer, requested return 2 | **1.0**: it takes the goal branch at X in **0%** of runs |

Why: DT conditions on the *return-to-go so far*. Arriving at X from S with a requested return of 2 is a situation it has never seen, so it copies the only S-trajectory it knows. Dynamic programming propagates value backwards through shared states, while sequence modeling only imitates whole trajectories.

**Teaching point:** this is why foundation action models rely on imitation of *good* demonstrations (or RL fine-tuning, S16) rather than return conditioning on mixed data.

## 4. The price of action tokens (notebook 69)
The same DT, with continuous actions replaced by *n* uniform bins (cross-entropy over bins, as in RT-2/OpenVLA):

| Action head | Achieved return (mean ± std over seeds) |
|---|---|
| Continuous (MSE) | −796 |
| 256 bins | −2036 ± 2571 |
| 64 bins | −975 ± 745 |
| **16 bins** | **−695 ± 581** |
| 8 bins | −725 ± 611 |
| 4 bins | −763 ± 631 |
| 2 bins | −1121 ± 533 |

Two failure modes bracket a sweet spot:
- **Too many bins (256):** each bin sees very few training examples, so predictions are noisy and training is unstable (look at the variance). OpenVLA gets away with 256 bins because it has about a million episodes, not 300.
- **Too few bins (2):** resolution is too coarse to balance the pendulum (bang-bang control).
- **In between (8–16 bins):** it matches or beats the continuous head on this small dataset. Discretization also acts as a regularizer, and the cross-entropy loss can represent multimodal actions.

**Teaching point:** the bin count is a data-size decision, not a free design choice. That's the motivation for smarter action tokenizers (FAST, S10 notebook 95) and continuous heads (flow matching, S08/S11).

## Key takeaways
- A Decision Transformer turns control into conditional sequence modeling: return-to-go in, actions out.
- The return knob only works *inside* the data distribution, and an extreme request can make things worse.
- Sequence models don't stitch; dynamic programming does.
- Action tokenization has a resolution/data trade-off: with little data, fewer bins win.

## References
- Chen et al. 2021, *Decision Transformer*; Brandfonbrener et al. 2022, *When does return-conditioned supervised learning work for offline RL?*
- Brohan et al. 2023 (RT-2); Kim et al. 2024 (OpenVLA): 256-bin action tokens
