# 63–64 · The Transformer in Two Pages, for Control

**Section:** S07 Sequence Models Meet Control · **Type:** Math · **Duration:** ~40 min
**Companion notebooks:** `63_attention`, `64_transformer_from_scratch`

## Learning objectives
- Derive scaled dot-product attention and explain each design choice (scaling, masking, positions).
- Assemble a pre-norm GPT block and train it from scratch.
- Use a causal transformer as a policy over a history of states.

---

## 1. Attention as soft lookup
Each token produces a query $q$, key $k$ and value $v$ (linear projections of its embedding). The output for query $i$ is a weighted average of values:
$$\mathrm{Attn}(Q,K,V) = \mathrm{softmax}\!\Big(\frac{QK^\top}{\sqrt{d_k}} + M\Big)V$$
- **Why $\sqrt{d_k}$:** with unit-variance entries, $q\cdot k$ has variance $d_k$. Unscaled logits grow with dimension and push the softmax toward one-hot, where gradients vanish. The notebook shows std(q·k) ≈ √d, and ≈ 1 after scaling.
- **Mask $M$:** $-\infty$ above the diagonal gives **causal** attention (token t sees only ≤ t), which is needed for autoregressive generation and for policies that must not peek at the future.
- **Multi-head:** $h$ heads of size $d/h$ attend to different relations; their outputs are concatenated and projected.
- Our implementation matches `F.scaled_dot_product_attention` to $10^{-6}$.

## 2. Positions
Self-attention is **permutation-equivariant** (verified in the notebook): without position information a trajectory is a bag of tokens. Fixes:
- sinusoidal encodings $PE_{t,2i} = \sin(t/10000^{2i/d})$, $PE_{t,2i+1} = \cos(\cdot)$,
- learned absolute embeddings (GPT; our Decision Transformer uses a learned **timestep** embedding),
- rotary embeddings (RoPE; Llama/Gemma and so OpenVLA and π0).

## 3. The block
Pre-norm GPT block:
$$x \leftarrow x + \mathrm{MHA}(\mathrm{LN}(x)),\qquad x \leftarrow x + \mathrm{MLP}(\mathrm{LN}(x)),\quad \mathrm{MLP} = W_2\,\mathrm{GELU}(W_1 x),\ W_1\in\mathbb R^{4d\times d}$$
Residual streams and pre-norm make deep stacks trainable. Parameters per block ≈ $12d^2$. Attention costs $O(T^2 d)$, the reason long histories and high-rate control are expensive.

## 4. Training a GPT from scratch (notebook 64)
Sanity task: sort 8 digits autoregressively. A 2-layer, ~100k-parameter model learns it. Then the same code becomes a **policy**: the input is the last $K=8$ privileged TableTop states, the output is the action at every position (causal), trained with MSE on expert data and evaluated closed-loop.

**Measured (notebook 64):**
| | |
|---|---|
| Sorting 8 digits (102k params) | **100%** of sequences exactly right |
| Causal transformer policy, full TableTop scene (state history, 9 s of training) | **32%** success (95% CI 21–44%): pick 65%, place 21%, stack 0% |

The second row is deliberately modest: a tiny model with 9 seconds of training already solves most picks from state history alone. The harder multi-step tasks (place, stack) are where the rest of the course earns its keep.

## 5. Why transformers for control
- **History:** handles partial observability (occluded objects, velocity from frames, which subtask am I in?).
- **Multimodal tokens:** images, language and proprioception become one sequence. That's the core of every VLA.
- **Scale:** the same architecture and tooling as LLMs, so web-pretrained weights can be reused.
- Costs: quadratic attention, latency (S11), and a tendency to copy history ("causal confusion") if the history is too informative about past actions.

## Key takeaways
- Attention = softmax(QKᵀ/√d)V; scaling keeps the softmax trainable; masks enforce causality.
- Positions must be injected; transformers are otherwise order-blind.
- A causal transformer over state history is a drop-in policy class.

## References
- Vaswani et al. 2017, *Attention Is All You Need*; Radford et al. 2019 (GPT-2)
- Karpathy, *minGPT / nanoGPT*
