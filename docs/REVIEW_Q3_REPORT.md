# Review of `q3_report_three_paragraphs.md` (Nikita), Thu 22 Oct

**Verdict:** the experiments are sound and **every headline number reproduces** (I re-ran them independently against the current repo, `32530fa`, including Akash's floor_log2 fix). What needs work is (a) conventions and metric not matching our locked decisions, (b) two conclusions that are weaker than stated, (c) the text, which is a list of caveats rather than an answer to Q3. Scripts for my re-run: `docs/prototypes/review_q3/`.

## 1. What reproduces
| Claim in the report | My re-run |
|---|---|
| C: E2M3 3.079%, MXINT6 4.047%; A–E winners A E2M3, B MXINT6, C E2M3, D E2M3, E MXINT6 | identical |
| Block size: MXINT6 at block 4 = 2.300% (8.000 bits); E2M3 block 25 = 3.061% vs block 32 = 3.079% (6.320 bits); block 50 = 3.200% vs 64 = 3.212% (6.160 bits) | identical |
| Outliers: with 100 replaced, winner flips E2M3 → E3M2 between 10σ and 20σ; with 20, between 20σ and 50σ; with 1 or 5, E2M3 through 50σ | identical (n=5, 50σ is close: 0.47 dB) |
| Sparsity: E2M3 through 50%; MXINT6 at 75/90/95% | 90/95% yes; **75% is a tie, see 3.1** |
| Scale spread up to 10⁴: E2M3 always best | yes, but **see 3.2** |

## 2. Must fix (blocking)
1. **Conventions are stale.** The report uses the repo's old BFP6 (block 32, `ceil`), which is numerically MXINT6 (the two columns in the block-size study are identical). Our decision D1 is BFP6 = 16|4|5 (floor exponent, RNE, 4-bit exponent). Rerun once Akash's update lands. Also state the conventions once, in a table with source tags (block 32 along rows + a tail block of 4 with its own scale; E8M0 power-of-two scale; `ocp` rule; RNE; saturation; E4M1 vmax 384; MXINT6 k ∈ [−31, 31]; BFP6 16|4|5; weights-only; FP64 evaluation; storage 6.32 bits/elem). The sentence "the team's notes still mark some conventions provisional" must go.
2. **Metric.** D6: primary metric is the relative Frobenius error of Y on the **held-out** X columns (`split_X`: odd columns), with all-X and W error secondary. The report uses all X_cal and calls it "mean relative output error" (it is a Frobenius ratio, not a mean of relative errors). I checked: held-out vs all-X changes margins by ≤ 0.02 dB here, so the conclusions survive, but follow D6.
3. **Scale policy.** D5: report `ocp` (class) **and** `opt`. The report uses default policies only. I checked that the outlier and sparsity crossovers do **not** move under `opt` (table below); say so with the numbers.
4. **It is not a report yet.** About 1,000 words of protocol and hedging in three paragraphs, no table, no figure, and almost no mechanism. Internal to-dos ("Before treating this as assignment-complete…", "Confirm with the course staff whether…", the provenance remark about A–E) belong in team notes, not in the report. Q3 asks *how* outliers, sparsity and scale variation affect the selection; each effect needs: the claim in one sentence, the crossover (with spread over seeds), the mechanism, and the evidence (counts of zeroed / saturated / subnormal values from `quantize(..., return_stats=True)`).
5. **Reproducibility.** "Standard deviations are in the summary CSV": no such file or script is in the repo. Push the experiment script and CSVs; one command must regenerate every number.

## 3. Conclusions that are weaker than stated
**3.1 Sparsity crossover.** "Between 50% and 75%" is too coarse to say anything, and the 75% result is a coin flip. With the same protocol (5 seeds) on a finer grid:

| zeros | winner (ocp) | E2M3 | MXINT6 | margin | opt policy | held-out X |
|---|---|---|---|---|---|---|
| 65% | E2M3 | 2.94 | 3.12 | 0.25 dB | E2M3 | E2M3 |
| 70% | E2M3 | 2.93 | 3.03 | 0.14 dB | E2M3 | E2M3 |
| **75%** | E2M3 (**0.01 dB tie**) | 2.90 | 2.91 | 0.01 dB | **E2M3** | E2M3 |
| 80% | **MXINT6** | 2.90 | 2.73 | 0.26 dB | MXINT6 | MXINT6 |
| 90% | MXINT6 | 2.84 | 2.27 | 0.98 dB | MXINT6 | MXINT6 |

So the crossover is ≈ 75–80%, and "MXINT6 won at 75%" depends on seed and policy. Report the crossover as ≈ 77% ± a few points with a mean ± sd of the E2M3 − MXINT6 difference over ≥ 10 seeds. Likely mechanism (verify with counts before writing it): in sparse blocks the surviving values are few and mostly close to the block maximum, where MXINT6's grid (sign + 5 magnitude bits) is finer than E2M3's top binade, and there are few small values for E2M3's finer low-end steps to help with.

**3.2 "No crossover under scale variation" is largely a metric artifact.** The global Frobenius error is dominated by the high-gain columns. Under column-group spread 10⁴ (5 seeds), error on the **weakest 30 columns only**:

| format | global Y error | weakest-30-columns error |
|---|---|---|
| E3M2 | 5.88% | **41.9%** |
| E2M3 | 3.18% | 78.8% |
| MXINT6 | 4.01% | 87.6% |

The low-gain weights are nearly destroyed by every format; E3M2 keeps them best because of its exponent range. The honest statement: with a global output-error metric and a white X, weak groups contribute almost nothing to Y, so E2M3 stays best; judged per group (or if X amplifies the weak columns) the ranking flips. Report both metrics. Also, the **row-group** experiment is nearly a null experiment by construction: each block scale is per row, so a row gain is absorbed (only the position within the binade changes); say that, or replace it with within-row/within-block spread, which is the informative one.

**3.3 Block size: the main finding is not stated.** From the table: MXINT6 beats E2M3 at block ≤ 8 (2.300 vs 2.862 at block 4; 2.833 vs 2.913 at 8) and loses from block 15 (3.380 vs 2.985); the crossover lies between 8 and 15, but those blocks cost 7.04–8.00 bits/elem. At the **same number of scales per row**, E2M3 wins (4 scales/row: 3.061 vs MXINT6 3.887). Blocks 15, 25 and 50 are artifacts of K = 100 (only the number of blocks per row sets the cost: block lengths 25–33 all mean 4 scales), so 25 vs 32 (0.6%) is not a design insight. The block-size study is also a sensitivity analysis outside the class convention (MX is 32 by definition, D9); say so, and use each format's own scale width (BFP 16|4|5 pays 4 bits per scale).

**3.4 A–E table.** E: "MXINT6 2.872" vs E2M3 2.885 is a 0.04 dB tie, not a win; B and E also carry the BFP6 duplicate. Say "tie".

## 4. Outliers: good, but add the mechanism and a better x-axis
Reproduced. Add: one outlier forces its block's scale, so the other values in that block are rounded on a coarse grid; E3M2's extra exponent bit keeps them representable while E2M3 flushes them to zero (use `n_zeroed`). A better x-axis than the raw count is the **fraction of blocks that contain an outlier** (100 random outliers touch ≈ 22% of the 400 blocks; 5 touch ≈ 1%), which explains why 100 flips the winner and 5 does not. Refine the magnitude grid around the flips (the winners differ by only 0.4–0.5 dB there). E4M1 never wins in any experiment; one sentence.

## 5. Conventions checklist for the Q3 section
| Item | Status in the report |
|---|---|
| Block 32, tail 4 with its own scale, storage 6.32 | stated (good) |
| E8M0 scale, `ocp` rule, RNE, saturation, no Inf/NaN | missing |
| E4M1 vmax 384; MXINT6 symmetric [−31, 31] | missing |
| BFP6 = 16|4|5 | **wrong (old default)** |
| Policies `ocp` and `opt` | **only defaults** |
| Held-out primary metric | **all X_cal** |
| Weights-only, X FP32; FP64 evaluation | missing |
| Datasets A–E are synthetic (README) | stated, but as a to-do; make it a footnote |

## 6. Suggested structure (≈ 1.5 pages + 3 figures)
1. Setup (5 lines + the conventions table + metric).
2. Outliers → figure (winner and margin vs fraction of blocks with an outlier, per magnitude) → mechanism with counts.
3. Sparsity → figure (error of E2M3 and MXINT6 vs % zeros) → crossover ≈ 75–80% ± → mechanism.
4. Scale variation → global vs weakest-group error → what changes when the metric changes; row vs within-block spread.
5. Block organization at equal blocks-per-row (and the equal-bits table).
6. A–E contrast as a small table, ties marked.
7. One paragraph: what this means for the selector (inputs the one-shot rule needs: sparsity, outlier/tail shape, within-block range).
