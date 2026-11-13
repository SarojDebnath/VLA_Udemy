# 65 · What Gets Tokenized: States, Actions, Returns, Language

**Section:** S07 · **Type:** Theory · **Duration:** ~20 min

## Learning objectives
- List the tokenization choices for each modality in a control transformer.
- Understand the trade-offs between continuous embeddings and discrete tokens.
- Read any VLA paper's tokenization scheme with this vocabulary.

---

## 1. A trajectory is a sequence
$$\tau = (\hat R_1, s_1, a_1,\ \hat R_2, s_2, a_2,\ \dots)$$
To feed it to a transformer, each element becomes one or more **tokens**: vectors of width $d$.

## 2. Options per modality
| Modality | Continuous embedding | Discrete tokens | Used by |
|---|---|---|---|
| Proprio state | linear projection | binned per dimension | DT, π0 (linear), RT-1 (none) |
| Image | CNN feature / ViT patches → projector | VQ codes | OpenVLA (patch tokens), RT-1 (TokenLearner, 8 tokens), Genie (VQ) |
| Language | frozen encoder embedding (USE, T5, CLIP) | LLM vocabulary tokens | RT-1 (USE + FiLM), RT-2 / OpenVLA (LLM tokens) |
| Return-to-go | linear projection | — | Decision Transformer |
| Action | linear in / regression out | **per-dimension bins (256)**, FAST (DCT + BPE), VQ | RT-1/2, OpenVLA (bins), π0-FAST (FAST), Octo/π0 (continuous, diffusion/flow) |
| Time | learned timestep embedding | — | DT, ACT |

## 3. Discrete action tokens: pros and cons
**Pros:** reuse the LLM's softmax head and cross-entropy loss; multimodal distributions for free; "actions are just another language".
**Cons (each a later notebook or lecture):**
- **Resolution floor:** 256 bins over a range gives a fixed precision (notebook 69 sweeps bins; S10 measures millimeters).
- **Sequence length:** 7 dims × H steps of tokens decoded autoregressively, which hurts high-rate control (S11).
- **Out-of-range clipping:** bins fit on a narrow data range saturate on new motions (S10).
- **Vocabulary hijacking:** OpenVLA overwrites the 256 least-used Llama tokens.

## 4. Continuous actions
Regress directly (MSE/L1, ACT) or generate with diffusion or flow (Octo, π0). Precise and fast, but needs a separate head instead of the LM vocabulary.

## 5. How many tokens per timestep?
Decision Transformer: 3 (R, s, a). RT-1: 6 images × 8 TokenLearner tokens + 11 action tokens. OpenVLA: 256 image patches + text + 7 action tokens. π0: 3 cameras × 256 patches + text + a chunk of 50 continuous actions processed by the action expert. Token count drives latency, which is the dual-system story of S11.

## Key takeaways
- Every modality can be embedded continuously or tokenized discretely.
- Discrete action tokens buy LLM compatibility and multimodality at the price of resolution and decode length.
- Read VLA papers by asking: what are the tokens, how many per step, and which loss trains each?

## References
- Chen et al. 2021 (Decision Transformer); Brohan et al. 2022 (RT-1); Pertsch et al. 2025 (FAST)
- Kandapagari, *Action Models for Robot Learning*, §8.4, §11.3
