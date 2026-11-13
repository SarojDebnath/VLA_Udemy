# 93–95 · Action Tokenization: Uniform Bins, Their Limits, and FAST

**Section:** S10 Building a VLA · **Type:** Math + labs · **Duration:** ~40 min
**Companion notebooks:** `93_action_tokenizer`, `94_bins_and_distribution_shift`, `95_fast_tokenizer`

---

## 1. Why tokenize actions at all?
A language model outputs a categorical distribution over its vocabulary. To reuse it as an action head with no new architecture and the same cross-entropy loss, each continuous action dimension becomes a **token**. RT-1, RT-2 and OpenVLA all do this. It's "a small idea with large consequences" (book §11.3).

## 2. Uniform per-dimension binning
For dimension d with range $[l_d, h_d]$ and $K$ bins:
$$k = \Big\lfloor K\cdot\mathrm{clip}\Big(\frac{a_d - l_d}{h_d - l_d}, 0, 1-\varepsilon\Big)\Big\rfloor,\qquad \hat a_d = l_d + \frac{k + 0.5}{K}(h_d - l_d)$$
- **Range from percentiles, not min/max.** OpenVLA uses the 1st–99th percentiles. One teleop glitch with min/max ranges inflates every bin. In notebook 93 a single glitch frame makes the dx bin width jump from 0.38 mm to **2.15 mm (≈6× coarser)** and the round-trip error from 0.14 to 0.75 mm, while 1st/99th-percentile ranges keep it at 0.22 mm / 0.20 mm.
- **Round-trip error:** ≈ bin width / 4 on average, at most half a bin. Measured on our TableTop data with 256 bins: translation MAE **0.16–0.20 mm**, yaw 1.2 mrad. Only 175–227 of the 256 tokens per dimension are ever used, and roll/pitch use exactly one (the expert never rotates them). Convert it to **physical units** (mm, mrad) and compare with the task tolerance. A few millimeters is fine for pick-and-place but not for inserting a connector.
- **OpenVLA specifics:** 256 bins per dimension, mapped onto the 256 *least-used* Llama tokens (ids 31744–31999); decoding is `bin_center[vocab_size − token_id − 1]`, then un-normalized with the dataset's q01/q99 selected by `unnorm_key`. Notebook 99 decodes by hand and matches `predict_action`.

## 3. The bin-count trade-off (notebook 94)
- Error falls as 1/K (a clean power law on a log-log plot).
- But every bin is a class that needs training examples. Measured on the 8.4k-frame dataset (dx):

  | Bins | 8 | 32 | 128 | 256 | 1024 |
  |---|---|---|---|---|---|
  | Median examples per used bin | 370 | 94 | 25 | 12 | 7 |
  | Empty bins | 0% | 0% | 13% | 27% | **61%** |

  S07 notebook 69 shows the behavioral consequence: on a small dataset, 16 bins beat 256.
- **Distribution shift:** a tokenizer fit on slow motions clips fast ones to the end bins, which silently caps the robot's speed. Measured: fit on slow episodes, the error is 0.14 mm in-distribution but **0.55 mm on fast episodes, with 8.6% of fast translation values clipped**. Fit ranges on data that covers the deployment behavior.

## 4. The deeper costs (book §13.1)
1. **Resolution floor:** precision is capped by the bin width.
2. **Sequence length:** 7 dims × H steps of tokens. A 50-step chunk would be 350 autoregressive tokens.
3. **Latency:** each token is a full forward pass of a billion-parameter decoder. OpenVLA manages ~6 Hz on an RTX 4090.
All three bite hardest on exactly the dexterous, high-frequency tasks we care about.

## 5. FAST: Frequency-space Action Sequence Tokenization (Pertsch et al. 2025)
Compress a whole chunk before tokenizing:
1. Normalize each dimension to [−1, 1] (percentiles).
2. **DCT along time** per dimension. Smooth trajectories put most of their energy in the first few coefficients (the notebook's energy plot).
3. **Scale and round** the coefficients to integers; most high-frequency ones become 0.
4. Flatten and apply **byte-pair encoding**: frequent integer patterns (long runs of zeros) merge into single tokens.

Result (notebook 95): a 16-step chunk that naive binning turns into 96 tokens becomes **22.5 ± 10.4 FAST tokens (4.3× compression)** with 0.8 mm translation error (58 base symbols, 966 BPE merges). The rounding **scale** trades token count for precision:

| Scale | 2 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|
| Tokens per chunk | 6.5 | 14.6 | 21.6 | 29.5 | 36.0 |
| Translation MAE | 3.06 mm | 1.46 mm | 0.79 mm | 0.39 mm | 0.20 mm |

The error halves each time the scale doubles, at a cost of about 7 extra tokens. π0-FAST uses this to train an autoregressive VLA with chunks of up to 1 s at 50 Hz, about 5× faster to train than diffusion/flow π0 at similar quality.

## Key takeaways
- Per-dimension binning makes any LLM an action model; fit ranges with percentiles and check errors in physical units.
- Bin count trades precision against data per class; tokenizers clip what they never saw.
- FAST = DCT + quantization + BPE: compressed, chunk-level action tokens.

## References
- Brohan et al. 2022 (RT-1); Kim et al. 2024 (OpenVLA); Pertsch et al. 2025 (FAST, arXiv:2501.09747)
- Kandapagari, *Action Models for Robot Learning*, §11.3, §13.1, Exercises 11.x.1–11.x.4
