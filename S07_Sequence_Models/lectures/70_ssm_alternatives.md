# 70 · Beyond Attention: State-Space Models (Mamba, RoboMamba) and the Bridge to Foundation Action Models

**Section:** S07 · **Type:** Theory · **Duration:** ~20 min

## Learning objectives
- Explain the linear state-space model behind S4/Mamba and why it scales linearly with sequence length.
- Understand selectivity (input-dependent parameters) and why it matters for control.
- Place SSM-based policies (RoboMamba) relative to transformer VLAs.

---

## 1. The cost of attention
Self-attention over $T$ tokens costs $O(T^2)$ compute and memory. Robot policies want long histories (minutes of video at 10–50 Hz), which quickly means thousands of tokens.

## 2. Linear state-space models
Continuous-time:
$$\dot h(t) = A h(t) + B x(t),\qquad y(t) = C h(t) + D x(t)$$
Discretized with step Δ (zero-order hold):
$$h_t = \bar A h_{t-1} + \bar B x_t,\qquad \bar A = e^{\Delta A},\ \bar B = (\Delta A)^{-1}(e^{\Delta A} - I)\Delta B$$
- **Recurrent form:** O(1) memory per step, ideal for streaming control at high rates.
- **Convolutional form** (fixed A, B, C): parallel training with an FFT, $O(T\log T)$.
- S4 (2022) made this work for long sequences with a structured A (HiPPO initialization).

Familiar to roboticists: it's the linear dynamical system / Kalman-filter state equation, applied as a learned sequence layer.

## 3. Mamba: selective state spaces (Gu & Dao 2023)
Make $B$, $C$ and Δ **functions of the input** $x_t$: the model chooses what to write into and read from its state. That breaks the convolutional shortcut but keeps the linear-time recurrence with a hardware-aware parallel scan. It matches transformers on language at small/medium scale with linear-time inference.

## 4. SSMs for robots
- **RoboMamba (2024):** a vision encoder + Mamba language model for reasoning and manipulation pose prediction, with efficient fine-tuning and fast inference.
- Mamba-based diffusion-policy backbones and long-horizon memory policies.
- Hybrids (attention for local detail, SSM for long memory) are common in 2025–2026 models.

## 5. The bridge to foundation action models (book §8.5)
Decision Transformer showed that "control = sequence modeling" works with plain supervised learning. Foundation action models scale that idea:
1. **Tokens:** images and language (pretrained encoders and LLMs) replace low-dimensional states.
2. **Conditioning:** language instructions replace return-to-go.
3. **Data:** many robots and tasks (Open X-Embodiment) replace single-task offline RL datasets.
4. **Heads:** discrete tokens (RT-2/OpenVLA) or diffusion/flow chunks (Octo/π0) replace per-step regression.

Return conditioning mostly disappeared because demonstrations are near-expert and language specifies the task; it may come back via RL fine-tuning (S16).

## Key takeaways
- Attention is quadratic; SSMs are linear-time recurrences with parallel training.
- Mamba's input-dependent parameters give attention-like selectivity.
- Transformer VLAs dominate today; SSMs and hybrids target long horizons and fast inference.

## References
- Gu, Goel & Ré 2022 (S4); Gu & Dao 2023 (Mamba); Liu et al. 2024 (RoboMamba)
- Kandapagari, *Action Models for Robot Learning*, §8.5
