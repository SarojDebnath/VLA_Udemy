# 14 · Optimization: Gradient Descent, Momentum, Adam, Schedules, Clipping

**Section:** S02 · **Type:** Math · **Duration:** ~25 min
**Companion notebook:** `14_optimization.ipynb`

## Learning objectives
- Write the update rules for SGD, momentum and Adam.
- Explain divergence from a too-large learning rate using curvature.
- Use warm-up + cosine schedules and gradient clipping the way VLA training recipes do.

---

## 1. Gradient descent
$$\theta_{k+1} = \theta_k - \eta\,\nabla L(\theta_k)$$
On a quadratic with curvature $\lambda$ (Hessian eigenvalue), GD converges only if $\eta < 2/\lambda_{\max}$. Above that it oscillates and diverges, which is the "loss explodes to NaN" failure of lecture 16. In ill-conditioned valleys like Rosenbrock, a step size small enough for the steep direction crawls along the flat one.

## 2. Momentum
$$v_{k+1} = \beta v_k + \nabla L(\theta_k),\qquad \theta_{k+1} = \theta_k - \eta v_{k+1}$$
It accumulates velocity along consistent directions and damps oscillation across the valley.

## 3. Adam
$$m_k = \beta_1 m_{k-1} + (1-\beta_1)g_k,\qquad v_k = \beta_2 v_{k-1} + (1-\beta_2)g_k^2$$
$$\hat m_k = \frac{m_k}{1-\beta_1^k},\quad \hat v_k = \frac{v_k}{1-\beta_2^k},\qquad \theta_{k+1} = \theta_k - \eta\frac{\hat m_k}{\sqrt{\hat v_k}+\epsilon}$$
A per-parameter step size, normalized by the gradient's recent magnitude. **AdamW** decouples weight decay and is the default optimizer for every VLA in this course (OpenVLA, Octo, π0, SmolVLA).

## 4. Learning-rate schedules
- **Warm-up** (linear ramp over the first few hundred steps): Adam's second-moment estimate is noisy early on, and large pretrained models get damaged by big early steps.
- **Cosine decay:** $\eta_t = \eta_{\max}\cdot\tfrac12\big(1+\cos(\pi t/T)\big)$, for a smooth anneal to fine convergence.
- Typical VLA fine-tune: AdamW, lr $\approx 10^{-4}$–$5\times10^{-4}$ for LoRA, warm-up 500–1000 steps, cosine decay.

## 5. Gradient clipping
$$g \leftarrow g \cdot \min\Big(1, \frac{c}{\|g\|}\Big)$$
A single outlier label (a teleop glitch, a wrong normalization constant) produces a huge gradient that throws parameters far away. Clipping bounds the step. In the notebook, one outlier target kicks the unclipped parameter far away; the clipped one barely moves. Most robot-learning recipes clip at norm 1.0.

## 6. Batch size and noise
SGD noise scales roughly as $\eta/B$. Doubling the batch size usually allows a somewhat higher learning rate. On a 4 GB GPU, use **gradient accumulation** to simulate large batches.

## Key takeaways
- Stability requires $\eta < 2/\lambda_{\max}$; too large diverges, too small stalls.
- Momentum and Adam fix ill-conditioning; AdamW is the VLA default.
- Warm-up + cosine and gradient clipping are standard because robot data is noisy and pretrained weights are fragile.

## References
- Kingma & Ba 2015, *Adam*; Loshchilov & Hutter 2019, *AdamW*
- Goodfellow, Bengio, Courville, *Deep Learning*, Ch. 8
- Kandapagari, *Action Models for Robot Learning*, §3.3, §3.5
