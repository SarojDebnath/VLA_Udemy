# 89–92 · VLM Anatomy, Language Conditioning, and Grounding vs Alignment

**Section:** S09 · **Type:** Theory + labs · **Duration:** ~45 min
**Companion notebooks:** `89_vlm_anatomy`, `90_language_conditioning`, `91_project_language_grounding`

---

## 1. Anatomy of a vision-language model (notebook 89)
```
image ──► vision encoder (ViT/SigLIP) ──► patch features ──► projector (MLP / pixel-shuffle) ──► "image tokens"
text  ──► tokenizer ──► text tokens ─────────────────────────────────────────────────────────► [image tokens | text tokens] ──► decoder-only LLM ──► next-token logits
```
- **SmolVLM-256M:** SigLIP-based vision encoder + pixel-shuffle connector + a small SmolLM2 decoder. It runs on our 4 GB laptop GPU and answers questions about TableTop renders.
- **Token budget:** a single image becomes dozens to hundreds of tokens. That dominates the prompt and therefore latency (S11).
- **Training stages** (typical): (1) train the projector on captions with frozen towers, (2) instruction-tune everything on VQA/chat, (3, for VLAs) co-train or fine-tune on robot data.
- Small VLMs describe scenes well and are unreliable on precise spatial questions, which is where action fine-tuning has to supply grounding.

Every large VLA follows this template: OpenVLA = Prismatic (DINOv2+SigLIP → MLP → Llama-2 7B); π0 = PaliGemma (SigLIP → Gemma 2B) + action expert; SmolVLA = SmolVLM backbone + action expert.

## 2. How a policy can consume language (notebook 90)
| Fusion | Mechanism | Example |
|---|---|---|
| Late (concat) | sentence embedding concatenated to the visual features before the MLP head | many BC baselines |
| **Early (FiLM)** | the embedding produces per-channel scale γ and shift β inside the vision backbone: $h \leftarrow h\,(1+\gamma(\ell)) + \beta(\ell)$ | BC-Z, **RT-1** (FiLM-EfficientNet) |
| Cross-attention | action/visual queries attend to language tokens | Octo, ACT variants |
| Prefix tokens | language tokens and image tokens in one LLM sequence | RT-2, OpenVLA, π0 |

FiLM lets the instruction change *what the vision network looks for* ("red" upweights red-sensitive channels) instead of only telling the head what to do with a fixed visual summary. Initializing the FiLM layers at zero keeps the pretrained backbone intact at the start.

**Paraphrase robustness:** we train with three phrasings per task (and frozen CLIP text embeddings) and test on held-out phrasings ("raise the red block off the table"). A pretrained text encoder maps paraphrases to nearby embeddings (cosine ≈ 0.98 for "pick up" vs "grab"), so policies generalize to unseen wording for free. That's the first taste of what web pretraining buys.

**Measured (notebook 90).** "Sort" scene: two cubes + a tray, pick or place, so the instruction must say *which* cube. 450 demos, the same encoder and head; only the fusion differs. 50 episodes each:

| Fusion | Training phrasing | Held-out paraphrases |
|---|---|---|
| Late (concat) | **10%** | 12% |
| Early (FiLM) | **98%** | **96%** |

This is the most dramatic result in the course so far. With concat, the vision network summarizes the scene *before* it knows which cube matters, so its 32 keypoints never reliably encode the right cube and the head can't recover it. With FiLM, "red" re-weights the visual channels, and the keypoints land on the red cube. The video cell shows both policies on the same sentence and episode, one where FiLM succeeds and concat fails, so students can see *how* the late-fusion policy goes wrong. And FiLM loses just 2 points on wording it never saw.

## 3. Grounding vs alignment (lecture 92, notebook 91)
**Alignment:** an embedding space where images and text about the same thing are close (CLIP).
**Grounding:** the instruction changes the *behavior* in the correct, physically meaningful way.

The grounding test in notebook 91 keeps the scene fixed and swaps only the instruction ("pick up the red / green / blue cube"), then records **which cube actually moved**. The resulting confusion matrix separates:
- policies that **follow** the instruction (diagonal),
- policies that do the **most common thing** regardless of language (one dominant column),
- policies that **move the wrong object** confidently (off-diagonal mass): a silent failure that success rate alone can hide when tasks are sampled randomly.

A meaningless instruction ("do something") is a useful control: whatever success remains comes from luck, not language.

**Measured (notebook 91, FiLM policy):** the instructed cube is the one that moves in **100%** of trials (a perfectly diagonal confusion matrix), with 100% task success. With "do something" the policy succeeds **7%** of the time: it really is reading the words. The video shows one scene, one policy and two different instructions, and the robot goes for a different cube each time.

What CLIP-style alignment lacks for acting:
- **Time:** it's a single static image, with no notion of progress.
- **Force and contact:** invisible in pixels.
- **Action:** an embedding isn't a motor command; the policy must learn the mapping from demonstrations.

## Key takeaways
- VLM = vision encoder → projector → LLM; image tokens dominate the sequence.
- Language enters policies by concat, FiLM, cross-attention or prefix tokens. On our multi-object task, concat reaches 10% and FiLM 98%; FiLM and prefixes ground language in perception more deeply.
- Test grounding directly (swap instructions in a fixed scene); success rates can hide language-ignoring policies.

## References
- Perez et al. 2018 (FiLM); Jang et al. 2021 (BC-Z); Brohan et al. 2022 (RT-1)
- Marafioti et al. 2025 (SmolVLM); Beyer et al. 2024 (PaliGemma)
