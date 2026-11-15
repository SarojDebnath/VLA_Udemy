# 85–88 · Vision Foundations for VLAs: ViT, CLIP, SigLIP, DINOv2

**Section:** S09 Vision and Language Foundations · **Type:** Math + theory · **Duration:** ~55 min
**Companion notebooks:** `85_vit_from_scratch`, `86_contrastive_clip_siglip`, `87_clip_zero_shot`, `88_dinov2_vs_siglip`

---

## 1. From CNNs to Vision Transformers (notebook 85)
- **CNN:** convolution builds in locality and translation equivariance, so it's data-efficient. ResNet-18 (11M parameters) is our policy backbone so far.
- **ViT (Dosovitskiy 2021):** split the image into P×P patches (16×16 → 196 tokens at 224²), embed each linearly (a stride-P convolution), add positions and a CLS token, then run a transformer. Almost no inductive bias, so it needs large-scale pretraining, but it then scales better and shares the token interface with language models.
- Patch tokens are exactly what VLMs feed into the LLM: OpenVLA passes 256 patch tokens from each vision encoder through a projector.
- **DINOv2 attention maps** on a TableTop render (no labels ever) highlight the robot, the cubes and the tray: self-supervised ViTs learn object-centric features.

## 2. Contrastive image-text pretraining: CLIP (notebook 86)
Two encoders $f$ (image) and $g$ (text) produce unit vectors; for a batch of N matched pairs:
$$S_{ij} = \tau\, f(x_i)^\top g(t_j),\qquad \mathcal L_{CLIP} = \tfrac12\Big[\mathrm{CE}(S, I) + \mathrm{CE}(S^\top, I)\Big]$$
(InfoNCE in both directions, with a learned temperature τ.) Each image must pick its own caption out of N, so the loss improves with batch size (CLIP used 32,768 over 400M web pairs).
- **Zero-shot classification:** embed "a photo of a {class}" for each class and pick the nearest.
- The notebook trains a tiny CLIP from scratch on 6,000 TableTop top-down renders with *relational* captions ("the red cube is left of the yellow tray") and tests zero-shot relation classification on new scenes.

## 3. SigLIP: the sigmoid loss
Treat every pair independently as a binary match / non-match:
$$\mathcal L_{SigLIP} = -\frac1N\sum_{i,j}\log\sigma\big(z_{ij}(\tau\, f_i^\top g_j + b)\big),\quad z_{ij} = \pm1$$
No softmax over the batch → no all-gather of the full similarity matrix, better at small batches, and it scales well. SigLIP is the vision tower of PaliGemma (π0) and part of OpenVLA's fused encoder.

## 4. Pretrained CLIP on robot scenes (notebook 87)
- Scene-level recognition works out of the box ("a robot arm and colored blocks on a table").
- Object color from segmentation crops works for most objects.
- **State and relations are weak:** "is the red cube inside the yellow tray?" is close to chance. CLIP aligns *what* is in the image with words; it doesn't encode *where*, *how* or *whether a task is done*. Hence the caution: **alignment is not grounding in action.**

## 5. DINOv2 vs SigLIP (notebook 88)
Linear probes on frozen features from 1,000 TableTop scenes:
- **Spatial probe:** regress the red cube's xy position.
- **Relational probe:** which cube is nearest the tray?

Self-supervised DINOv2 (trained to match views of the same image) keeps fine spatial and geometric information in its patch tokens. Language-supervised SigLIP is aligned with text but discards spatial detail it doesn't need for captions. OpenVLA concatenates **DINOv2 + SigLIP** channel-wise for exactly this reason: geometry from one, semantics and language alignment from the other. The OpenVLA paper reports that fine-tuning the vision encoder (not freezing it) was crucial.

## Key takeaways
- ViT turns images into tokens; pretrained ViTs are the eyes of every VLA.
- CLIP and SigLIP align images with text through contrastive objectives; SigLIP's sigmoid loss scales more easily.
- Contrastive features know *what*, not *where* or *whether*; DINOv2 adds spatial structure. Fusing both is a common VLA design.

## References
- Dosovitskiy et al. 2021 (ViT); Radford et al. 2021 (CLIP); Zhai et al. 2023 (SigLIP); Oquab et al. 2024 (DINOv2)
- Kim et al. 2024 (OpenVLA)
