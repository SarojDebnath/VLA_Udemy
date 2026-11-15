# 105–108 · Smooth Control: π0 and Flow-Matching Action Experts

**Section:** S11 Continuous VLAs and Dual Systems · **Type:** Theory + math · **Duration:** ~55 min
**Companion notebooks:** `108_minivla_flow` (+ S10 `103_project_minivla` for comparison)

---

## 105 · The trouble with discrete action tokens
Three costs that cluster on exactly the tasks we care about:
1. **Resolution:** 256 bins per dimension give a precision floor. Fine for sliding a can; not for seating a peg or tensioning cloth.
2. **Sequence length:** a 1-second chunk at 50 Hz × 7 dims = 350 tokens, decoded one at a time.
3. **Latency:** every token is a full pass of a billion-parameter decoder. OpenVLA reaches ~6 Hz; dexterous bimanual work needs ~50 Hz.

And plain regression isn't the escape: it averages modes (S08). We need **continuous, multimodal, chunked** outputs that are **fast**.

## 106 · π0 architecture (Black et al. 2024)
- **VLM backbone:** PaliGemma 3B (SigLIP 400M + Gemma 2B). It receives up to three camera images and the language instruction, and is *not* asked to output actions.
- **Action expert:** ≈300M parameters with Gemma-like blocks. It receives the proprioceptive state $q_t$ and the **noisy action chunk** $A^\tau_t$ (H = 50).
- **"Mixture of experts" by token type:** image/text tokens use the VLM weights; state/action tokens use the expert weights; **one shared self-attention** spans all tokens. It's two transformers that read each other's keys and values, not sparse routing.
- **Block-wise causal mask:** [images + text] ← attend among themselves only (so the VLM prefix can be computed once and cached); [state] sees the prefix; [actions] see everything, including each other (bidirectional within the chunk).
- **Inference:** one prefix pass (cached KV), then **10 Euler steps** through the small expert, about 73 ms for a 50-action chunk on an RTX 4090.
- **Data:** ~10,000 hours across 7 robot configurations and 68 tasks, plus OXE; pretrain then post-train on curated high-quality data.

## 107 · Flow matching as a control objective
$$A^\tau_t = \tau A_t + (1-\tau)\,\epsilon,\qquad \mathcal L(\theta) = \mathbb E\,\big\|v_\theta(A^\tau_t, o_t, \tau) - (A_t - \epsilon)\big\|^2$$
- π0 samples τ from a **Beta distribution emphasizing low τ (high noise)**: the hard part of action generation is the coarse decision, not the fine detail.
- Integrate $A^{\tau+\delta} = A^\tau + \delta\,v_\theta$ from pure noise; δ = 0.1.
- Why flow over diffusion here: a simpler objective, straighter paths and fewer steps; the multimodality is the same (S08 notebook 81).

## 108 · MiniVLA v2: our π0-style model
| | MiniVLA (S10) | MiniVLA v2 (S11) |
|---|---|---|
| Backbone | SmolVLM-256M + LoRA | same |
| Head | the LM's own vocabulary: 7 tokens/step | **separate 4.5M-parameter flow expert** |
| Expert ↔ VLM | — | expert cross-attends to the VLM's last hidden states + a state token (cross-attention variant, as in SmolVLA/GR00T, instead of π0's shared self-attention) |
| Output | 1 action per decode (7 passes) | 16-action chunk (1 prefix pass + 10 small expert passes) |
| Loss | cross-entropy | flow matching (MSE on velocity) |

**Measured** ("single" scene: one cube + a tray, pick or place, 450 demos, front camera only, 30 new layouts):

| | Token MiniVLA (S10 103) | MiniVLA v2: flow expert (S11 108) |
|---|---|---|
| Action-token accuracy | 85–88% | n/a |
| **Closed-loop success** | **0%** (3% with relative-target labels) | **53%** (95% CI 36–70%) |
| Inputs | image + instruction | image + instruction + **robot state** |
| Output | 1 action per 7 sequential decoder passes | 16-action chunk per 1 prefix pass + 10 expert passes |
| Latency | 7 LM passes *per action* | **4.3 ms per action** (69 ms per chunk) |

**The cliffhanger and the payoff.** S10's token model looked healthy on paper: token accuracy above 85%. On the robot it never succeeds. Three things kill it:
1. **Per-step tokens with no proprioception:** it must infer from one image where the gripper is and how far to move in the next 0.1 s, at 256-bin resolution.
2. **No chunking:** every step is a fresh, independent decision. S08 notebook 73 showed single-step policies scoring 14% where chunks score 100%.
3. **Discretization:** a near-miss bin is as wrong as a far one under cross-entropy.

The flow expert changes the *head*, not the backbone: continuous outputs, 16-step chunks, robot state as an input, and the action representation that worked in S08 (commanded targets relative to the gripper). Same SmolVLM, same data, and 0% becomes 53%. That's why π0, SmolVLA and GR00T all moved to continuous action experts.

There's honest headroom left: 53% from one camera and 450 demos is below S08's 100% (two cameras, a small dedicated encoder). Student extensions: add the wrist camera as a second image, train longer, or fine-tune the vision LoRA rank.

## Key takeaways
- Discrete tokens pay in resolution, length and latency, all at once on dexterous tasks.
- π0 = VLM prefix + a separate flow-matching action expert sharing attention; the prefix is computed once per observation.
- The same split (big perception and language model, small fast action generator) reappears in dual systems.

## References
- Black et al. 2024, *π0* (arXiv:2410.24164); Pertsch et al. 2025 (π0-FAST)
- Shukor et al. 2025, *SmolVLA*; NVIDIA 2025, *GR00T N1*
