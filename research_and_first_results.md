# Group 8 (Dataset C, 6-bit): research notes and first results

Everything below uses **provisional conventions** (OCP MX v1.0 spec, plus my own assumptions for E4M1 and BFP6). Re-run it once the class conventions are known. Code: `formats.py` (the five quantizers) and `explore.py` (all experiments), in this folder.

## 1. What the literature says

| Paper | Relevant takeaway |
|---|---|
| [OCP MX spec v1.0](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf) | Block 32, E8M0 power-of-two shared scale `2^(floor(log2 amax) − emax)`. Round-to-nearest-even, saturate on overflow, flush below the min subnormal. FP6 E2M3 (bias 1, max 7.5) and E3M2 (bias 3, max 28), no Inf/NaN. MXINT8 = k/64. **E4M1 and BFP6 are not defined there.** |
| [Rouhani et al., Microscaling Data Formats for DL (2023)](https://arxiv.org/pdf/2310.10537) | Introduces MXFP6/MXINT8. Reports MXFP6 as near-lossless for weights at inference. |
| [Rouhani et al., Shared Microexponents (ISCA 2023)](https://arxiv.org/abs/2302.08007) | Defines BFP/MSFP: a shared exponent from the block max with per-element sign+mantissa. Compares formats by **QSNR** (signal-to-noise ratio of quantization) and frames the trade-off between block size and scale granularity. |
| [Kuzmin et al., FP8 Quantization: The Power of the Exponent (NeurIPS 2022)](https://proceedings.neurips.cc/paper_files/paper/2022/file/5e07476b6bd2497e1fbd11b8f0b2de3c-Paper-Conference.pdf) | **Closed-form expected MSE** = rounding term + clipping terms, integrated over the value distribution. More exponent bits only pay off for heavy-tailed or outlier distributions. |
| [van Baalen et al., FP8 vs INT8 (2023)](https://arxiv.org/pdf/2303.17951) | "It's all about the outliers": uniform → INT, Gaussian → FP with few exponent bits (≈ INT), Student-t → more exponent bits. |
| [Zhang et al., Integer or Floating Point? MoFQ (2023)](https://arxiv.org/pdf/2305.12356) | Per-layer selection of INT vs FP by minimum tensor MSE or layer-output MSE. Simple, and close to exhaustive model-level selection. This is closest to what your assignment asks. |
| [Egiazarian et al., MXFP4 promise vs performance (ICLR 2026)](https://arxiv.org/html/2509.23202) | Power-of-two E8M0 scales add noticeable error. An **MSE-optimal scale search** helps. Includes analysis of block size vs distribution (Laplace vs Normal). |

How this maps onto your three required answers:
- **Q1** (which format): minimise output error ‖(W − Q(W))·X‖. With uncorrelated rounding error this is ≈ Σ_j ‖ΔW[:,j]‖² · (XXᵀ)_jj, so input statistics weight the columns.
- **Q2** (cheap selection): Kuzmin-style analytic MSE (rounding + clipping, computed from block statistics without rounding anything), or quantize a sample of blocks. Compare both with exhaustive evaluation.
- **Q3** (outliers, sparsity, scale): controlled perturbations of W.

## 2. First results on W_C (provisional conventions)

Output error measured on X_cal (higher ySQNR is better):

| Format | default scale ySQNR | after per-block exponent search | analytic predictor (no rounding) | 20% rows sampled |
|---|---|---|---|---|
| **MXFP6 E2M3** | **30.23 dB** (3.08% rel. err) | 30.29 dB | 30.18 dB | 30.12 dB |
| MXINT6 | 27.86 dB | 28.19 dB | 27.84 dB | 28.55 dB |
| BFP6 | 27.60 dB | 28.19 dB | 27.61 dB | 28.16 dB |
| MXFP6 E3M2 | 25.29 dB | 25.62 dB | 25.09 dB | 25.50 dB |
| MXFP6 E4M1 | 19.10 dB | 19.85 dB | 18.69 dB | 18.94 dB |

Observations:
- **E2M3 wins clearly** (by ~2.4 dB over MXINT6). W_C is only mildly heavy-tailed (kurtosis 5.8), so with block-32 scaling the extra mantissa bit beats extra exponent range. E3M2 and E4M1 waste bits on range the block doesn't need.
- **The analytic predictor gets the exact exhaustive ranking**, to within 0.4 dB, without rounding any value. That's a strong answer to Q2. Sampling 20% of rows also ranks correctly, but it's noisier (it swaps closer pairs more easily).
- **Scale search helps a little** (+0.05 to 0.75 dB). It helps most for formats whose default scale clips or wastes range (E4M1, BFP6). The output-aware objective (H = XXᵀ) is no better than plain weight MSE here, because X_cal is nearly white (kurtosis 3.0, so XXᵀ is close to diagonal).
- **No overfitting to X_cal**: tuning on half the columns and testing on the other half gives the same gains.
- **Under these conventions MXINT6 and BFP6 end up on the same grid** after scale search. The class definition of BFP6 must differ somehow (block size? exponent width? two's complement?), and the report should show where they diverge.

Sensitivity (who wins when W is perturbed):

| Perturbation | Winner | Note |
|---|---|---|
| Original | E2M3 | |
| Clip outliers at 3σ | E2M3 | all formats gain about 1 dB |
| 20 outliers ×50 | E2M3 (margin shrinks) | E4M1 *gains*: its range starts to matter |
| 50% sparse | E2M3 | INT closes the gap |
| **90% sparse** | **MXINT6** | blocks become a few non-zeros, so a uniform grid wins |
| Row or column scale spread 1e-2 to 1e2 | E2M3 | per-block scaling absorbs it |
| **Uniform data** | **MXINT6** | matches theory (van Baalen) |
| Student-t ν=3 | E2M3 | gap to INT widens |

So the selection does change with the data, and the reasons can be explained. That's what Q3 asks for.

Storage: 6 bits + an 8-bit scale per 32-block works out to 6.32 bits/element. The 100-column rows have a 4-element tail block, which costs extra scale bits.

## 3. Open points that depend on the class notes
- E4M1 bias and special values (I assumed bias 7, no Inf/NaN, max 384).
- BFP6 definition (block size, exponent width, mantissa sign convention).
- MXINT6 fixed-point layout (I assumed k/16, |k| ≤ 31, with symmetric clipping).
- Whether the scale must follow the OCP floor rule, or a scale search is allowed. A search changes the scale *value*, not the format, so it should be allowed.
- How to handle the partial last block (I used a 4-element block with its own scale).

## 4. Histogram predictor (selection without quantizing W) — `predictor.py`, `validate_predictor.py`

**Metric.** Predicted output error P_f ≈ E‖(W − Q_f(W))·X‖² = Σ_ij E[ΔW_ij²]·(XXᵀ)_jj.

**Key observation.** Every candidate scales a block by 2^(e_b − emax_f), with e_b = floor(log2 amax_b). So after normalising each value by its block's binade, u = |w|/2^e_b ∈ (0, 2), the error of format f is 2^(2e_b)·g_f(u). Here g_f depends only on the format: uniform rounding noise step²/12, the clipping error (u − max)², or flush-to-zero u². All the data dependence then collapses into **one format-independent weighted histogram** of u, with weight 2^(2e_b)·H_jj.

**Algorithm.**
1. One pass over W: block maxima, normalised magnitudes u, and the weighted histogram (512 log-spaced bins). The block maximum itself is kept aside (one value per block) because it carries the most weight and its error is deterministic. BFP6's exponent rule needs a single extra flag per block (u_max > 31/16), so two histograms are kept.
2. For each format: P_f = Σ_bins hist·g_f(u_bin) + Σ_blocks weight·exact_err_f(u_max).
3. Choose argmin_f P_f.

**Accuracy.**
- On W_C, the predicted SQNR is within 0.05 dB of exhaustive for four formats and 0.22 dB for E4M1. The ranking is identical (Spearman 1.0).
- Stress test: 90 synthetic matrices (Gaussian, Laplace, Student-t ν=3/6, uniform, 90% sparse, ×50 outliers, row-scaled, shuffled W_C), each with a random X. **The top-1 choice agrees with exhaustive 90/90 times.**
- The winners differ across families (E2M3, MXINT6, E3M2), so the predictor isn't just always saying E2M3.
- An earlier version that put the block maxima in the histogram too got only 6/10 on the outlier family, with a 0.24 dB average regret. Treating the block max exactly is what fixed it. This is worth reporting as an ablation.

**Cost.**
- Exhaustive: 5 × (10,000 roundings + 2.56M multiply-adds for Q(W)·X).
- Predictor: 10,000 binning operations once, plus 5 × (1,028 bin evaluations + 400 block-max roundings).
- Against exhaustive output-error evaluation, that's roughly 750× fewer operations. Against an exhaustive weight-only MSE check (no matmul) it's only about 3× fewer, so state the comparison baseline clearly.
- The cost of scoring a format doesn't depend on matrix size; only the shared stats pass grows with W.

**Assumptions to state in the report.** Rounding errors are treated as independent, so off-diagonal terms of XXᵀ are ignored (X_cal is nearly white, so this is fine here). Errors within a bin are averaged with the uniform-noise model, and the conventions in section 3 apply.
