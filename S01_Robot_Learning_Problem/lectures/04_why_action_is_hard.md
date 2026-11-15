# 04 · Why "Action" Is the Hard Part of Robotics

**Section:** S01 The Robot Learning Problem · **Type:** Theory · **Duration:** ~15 min
**Companion notebook:** `04_open_vs_closed_loop.ipynb`

## Learning objectives
- Explain why a model that perceives correctly can still fail to act correctly.
- Distinguish open-loop from closed-loop execution, and feed-forward from feedback.
- Recognize model mismatch and disturbances as the two sources of execution error.

---

## 1. Perception is a one-shot problem; action is a loop
A classifier makes one prediction per image, and its errors are independent. A robot policy makes a decision every few milliseconds, and **each decision changes the input to the next one**:

$$o_{t+1} \sim P(\cdot \mid s_t, a_t), \qquad a_{t+1} = \pi(o_{t+1})$$

So the policy's errors become part of its own future data. This feedback is what separates control from prediction.

## 2. Two ways to execute a plan
**Open loop (feed-forward).** Compute the commands ahead of time from a model, then play them back blindly:

$$\tau_{ff}(t) = M(q_d)\ddot q_d + c(q_d,\dot q_d) \quad\text{(inverse dynamics, MuJoCo's } \texttt{mj\_inverse}\text{)}$$

**Closed loop (feedback).** Measure the actual state and correct the deviation:

$$\tau(t) = \tau_{ff}(t) + K_p\,(q_d - q) + K_d\,(\dot q_d - \dot q)$$

The feedback terms react to anything the model did not predict.

## 3. What goes wrong without feedback (notebook results)
The notebook drives a 2-link arm through a smooth quintic trajectory three ways:

| Case | Open-loop final error | Closed-loop final error |
|---|---|---|
| Exact model | ~11 cm | ~0 cm |
| Link-2 mass +40% | ~96 cm | ~0 cm |
| 50 ms push | ~95 cm | ~0 cm |

Three lessons:
1. **Even a perfect model drifts.** Small numerical differences grow because the arm, held up against gravity, sits in an unstable configuration. Open loop has nothing to pull it back.
2. **Model mismatch** (wrong mass, friction, latency) is unavoidable on real hardware. This is the **sim-to-real gap** in miniature.
3. **Disturbances** (a push, a slipping object, a person) can't be predicted at all.

## 4. Gains, stability, and a practical warning
Feedback isn't free. In the notebook, a derivative gain of $K_d = 40$ made the *simulation* unstable. With explicit time stepping, a velocity gain must satisfy roughly

$$\frac{K_d\,\Delta t}{I} < 2$$

where $I$ is the link inertia ($\approx 0.0125$ kg·m², so $K_d\Delta t/I = 6.4$ was too large). Real servo loops have the same limit, imposed by sensor rate and latency. Faster control loops allow stiffer feedback. This is one reason VLAs moved from 3 Hz (RT-1) to 50 Hz (π0) action generation.

## 5. Why this matters for VLAs
A learned policy *is* a feedback controller: it re-observes the world each step. But its "model" is only the training data. When it drifts into states the data never covered, its corrections are wrong, and the next lecture shows how fast that compounds. Much of modern VLA design is about restoring good feedback behavior:
- closed-loop re-planning at high rate,
- action **chunks** with receding-horizon execution (S08),
- a low-level controller (IK + PD) under the learned policy (S04, S11),
- safety shields that react to force and workspace limits (S15).

## Key takeaways
- Acting is a loop: errors change future inputs.
- Open-loop execution fails under model error and disturbances; feedback fixes both.
- Feedback gain is limited by loop rate and latency, so faster loops mean more robust control.
- A VLA is a learned feedback controller whose "model" is its training distribution.

## References
- MuJoCo docs, *Computation → Inverse dynamics* (`mj_inverse`)
- Lynch & Park, *Modern Robotics*, Ch. 11 (robot control)
