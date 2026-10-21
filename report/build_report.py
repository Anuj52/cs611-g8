"""Build the draft report (report.docx) from text + figures. Numbers are from the provisional run."""
import os
from docx import Document
from docx.shared import Pt, Inches, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "..", "figures")
d = Document()
st = d.styles["Normal"]; st.font.name = "Calibri"; st.font.size = Pt(11)

def P(text, bold=False, italic=False, color=None):
    p = d.add_paragraph(); r = p.add_run(text); r.bold = bold; r.italic = italic
    if color: r.font.color.rgb = RGBColor.from_string(color)
    return p
def todo(text): return P("[TO CONFIRM] " + text, italic=True, color="B5452B")
def bullets(items):
    for it in items: d.add_paragraph(it, style="List Bullet")
def numbered(items):
    for it in items: d.add_paragraph(it, style="List Number")
def table(rows, header=True):
    t = d.add_table(rows=len(rows), cols=len(rows[0])); t.style = "Light Grid Accent 1"
    for i, row in enumerate(rows):
        for j, v in enumerate(row):
            c = t.cell(i, j); c.text = str(v)
            if i == 0 and header:
                for r in c.paragraphs[0].runs: r.bold = True
    d.add_paragraph()
def fig(name, caption, w=6.0):
    d.add_picture(os.path.join(FIG, name), width=Inches(w))
    d.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    c = P(caption, italic=True); c.alignment = WD_ALIGN_PARAGRAPH.CENTER
def eq(text):
    p = d.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run(text); r.font.name = "Cambria Math"; r.italic = True

# ---------------------------------------------------------------- title
d.add_heading("Data-Dependent Selection of 6-bit MX/BFP Formats", 0)
P("CS611 Assignment 1 · Group 8 · Dataset C · 6-bit element width", bold=True)
todo("Group member names and roll numbers.")

d.add_heading("Summary", 1)
P("We select a 6-bit format for the weight matrix W_C ∈ R^(100×100) from the candidates MXFP6 E3M2, "
  "MXFP6 E2M3, MXFP6 E4M1, MXINT6 and BFP6. We use the output error ‖(W − Q(W))X‖² as the quality "
  "metric. We then propose a histogram-based predictor that estimates this error for every format "
  "without quantizing W or multiplying by X.")
bullets([
    "Q1 (best format): MXFP6 E2M3. Output SQNR 30.2 dB (3.1% relative output error), 2.4 dB ahead of MXINT6.",
    "Q2 (efficient selection): the predictor reproduces the exhaustive ranking on W_C (Spearman 1.0, error "
    "within 0.22 dB). It also picks the exhaustive winner in 90/90 synthetic matrices from nine distribution "
    "families, using about 750× fewer operations than exhaustive output-error evaluation.",
    "Q3 (data effects): heavy tails favour E2M3 over INT, and uniform or very sparse data flip the choice to "
    "MXINT6. Per-block scaling absorbs row and column scale variation. Formats whose maximum value "
    "falls below the shared-scale binade (E3M2, E4M1) lose accuracy to clipping.",
])

# ---------------------------------------------------------------- 1
d.add_heading("1. Problem and data", 1)
P("Given W (FP32, 100×100) and a calibration input X_cal (FP32, 100×256), choose the element format that "
  "best preserves Y = W·X at 6 bits per element. A separate X_test is used for grading, so the method "
  "must not overfit X_cal.")
table([["Statistic", "W_C", "X_cal"],
       ["shape", "100 × 100", "100 × 256"],
       ["mean / std", "0.0047 / 1.000", "−0.0003 / 0.100"],
       ["min / max", "−5.70 / 5.59", "−0.41 / 0.40"],
       ["excess tail (kurtosis, Gaussian = 3)", "5.76", "3.02"],
       ["entries beyond 4σ", "0.45%", "0.008%"],
       ["exact zeros", "0", "0"],
       ["negative entries", "49.5%", "50.1%"]])
fig("fig1_weight_distribution.png",
    "Figure 1. W_C is close to Gaussian in the bulk but has clearly heavier tails (log scale).")
P("W_C has no sparsity to exploit. Its row max |w| ranges from 1.9 to 5.7 and its row std from 0.78 to 1.26, "
  "so scale variation is moderate. X_cal is close to white Gaussian, so XXᵀ is close to a scaled identity.")

# ---------------------------------------------------------------- 2
d.add_heading("2. Format definitions and conventions", 1)
todo("Replace with the exact classroom conventions. The values below follow the OCP MX v1.0 specification "
     "where it defines a format; E4M1, MXINT6 and BFP6 are our assumptions.")
table([["Format", "Element encoding", "Bias", "Max |element|", "Min subnormal", "Shared scale"],
       ["MXFP6 E3M2", "1s + 3e + 2m", "3", "28", "0.0625", "E8M0, 2^(⌊log2 amax⌋ − 4)"],
       ["MXFP6 E2M3", "1s + 2e + 3m", "1", "7.5", "0.125", "E8M0, 2^(⌊log2 amax⌋ − 2)"],
       ["MXFP6 E4M1", "1s + 4e + 1m", "7 (assumed)", "384 (assumed)", "2^−7", "E8M0, 2^(⌊log2 amax⌋ − 8)"],
       ["MXINT6", "sign + 5-bit magnitude, k/16", "–", "31/16", "1/16", "E8M0, 2^⌊log2 amax⌋"],
       ["BFP6", "sign + 5-bit mantissa", "–", "31·scale", "scale", "2^⌈log2(amax/31)⌉ (no clipping)"]])
bullets([
    "Block size 32 along the reduction dimension (columns of W). 100 = 3·32 + 4, so each row has a "
    "final 4-element block with its own scale.",
    "Rounding: round-to-nearest, ties-to-even. Overflow: saturate to ±max. Values below half the smallest "
    "subnormal round to zero.",
    "FP6 formats have no Inf/NaN encodings. A NaN shared scale (E8M0 = 0xFF) is never produced by finite FP32 input.",
    "Storage: 6 bits per element + 8 bits per block = 6.32 bits/element for W_C (the 4-element tail "
    "block raises this from the ideal 6.25).",
])
P("No candidate is ineligible: W_C is finite, signed, and inside every format's range after scaling.")

# ---------------------------------------------------------------- 3
d.add_heading("3. Method", 1)
d.add_heading("3.1 Quality metric", 2)
P("What matters is the effect of quantization on the layer output, not on the weights themselves. "
  "With ΔW = W − Q(W) and H = X Xᵀ:")
eq("E_out(f) = ‖ΔW·X‖²_F = Σ_i  ΔW_i · H · ΔW_iᵀ  ≈  Σ_ij  E[ΔW_ij²] · H_jj")
P("The approximation assumes rounding errors are uncorrelated across columns. It holds well here because "
  "X_cal is nearly white, and it lets input statistics weight each column of W. We report "
  "output SQNR = 10·log10(‖WX‖² / E_out).")

d.add_heading("3.2 Scale optimisation", 2)
P("The default shared exponent comes from the block maximum. For each block we also try exponent shifts of −2…+2 "
  "and keep the one that minimises the block's error, using either weight MSE or the output-aware error "
  "ΔW_b·H_bb·ΔW_bᵀ. This changes only the scale value, not the format definition.")

d.add_heading("3.3 Histogram predictor (selection without quantizing)", 2)
P("Every MX/BFP candidate scales a block by a power of two tied to the block maximum, "
  "s_f = 2^(e_b − emax_f) with e_b = ⌊log2 amax_b⌋. Normalise every weight by its block binade:")
eq("u = |w| / 2^(e_b) ∈ (0, 2)")
P("Then the squared error that format f makes on w is 2^(2e_b)·g_f(u). Here g_f depends only on the format (Figure 2):")
bullets(["uniform rounding noise step_f(u)²/12 inside the representable range,",
         "clipping error (u − max_f)² above the format's maximum,",
         "flush-to-zero error u² below half the smallest step."])
fig("fig2_error_model.png",
    "Figure 2. Relative error of each format against position within the block. E2M3 is most accurate "
    "where most weights lie. E3M2 and E4M1 clip the top of every binade (u > 1.75 and u > 1.5).")
P("Combining this with the metric of Section 3.1, all data dependence collapses into a single "
  "format-independent histogram of u, where weight w_ij contributes 2^(2e_b)·H_jj:")
numbered([
    "Statistics pass (once, shared by all formats): compute block maxima and normalised magnitudes u, and "
    "accumulate the weighted histogram over 512 log-spaced bins. The block maximum itself is stored "
    "separately (one value per block): it carries the most weight and its error is deterministic. One flag "
    "per block records whether u_max > 31/16, which BFP6's exponent rule needs.",
    "Scoring (per format): P_f = Σ_bins hist·g_f(u_bin) + Σ_blocks weight·exact_err_f(u_max).",
    "Selection: f* = argmin_f P_f.",
])
P("Cost: one pass over W, then O(#bins + #blocks) per format, independent of the number of columns of X. "
  "Exhaustive evaluation needs, per format, a full quantization of W plus a 100×100×256 matrix product.")

# ---------------------------------------------------------------- 4
d.add_heading("4. Results", 1)
d.add_heading("4.1 Exhaustive evaluation (ground truth)", 2)
table([["Format", "Weight SQNR (dB)", "Output SQNR (dB)", "Rel. output error", "After scale search (dB)", "Clipped elements"],
       ["MXFP6 E2M3", "30.22", "30.23", "3.08%", "30.29", "0.43%"],
       ["MXINT6", "27.91", "27.86", "4.05%", "28.19", "0.20%"],
       ["BFP6", "27.67", "27.60", "4.17%", "28.19", "0%"],
       ["MXFP6 E3M2", "25.32", "25.29", "5.44%", "25.62", "0.96%"],
       ["MXFP6 E4M1", "19.13", "19.10", "11.09%", "19.85", "2.45%"]])
P("Scale search helps most where the default scale clips (E4M1, E3M2) or wastes range (BFP6), but it "
  "never changes the ranking. Output-aware search gives the same result as weight-MSE search, as expected "
  "for a nearly white X. Tuning on half of the X_cal columns and testing on the other half gives the "
  "same gains, so there is no sign of overfitting to X_cal.")

d.add_heading("4.2 Predictor vs. exhaustive", 2)
fig("fig3_predicted_vs_exhaustive.png", "Figure 3. Predicted and exhaustive output SQNR on W_C.")
table([["Test set", "Top-1 agreement", "Mean regret (dB)", "Spearman ρ", "Exhaustive winners"],
       ["W_C", "1/1", "0", "1.00", "E2M3"],
       ["Gaussian", "10/10", "0", "1.00", "E2M3"],
       ["Laplace", "10/10", "0", "1.00", "E2M3"],
       ["Student-t ν=3", "10/10", "0", "1.00", "E2M3"],
       ["Student-t ν=6", "10/10", "0", "1.00", "E2M3"],
       ["Uniform", "10/10", "0", "1.00", "MXINT6"],
       ["90% sparse", "10/10", "0", "1.00", "MXINT6"],
       ["20 outliers ×50", "10/10", "0", "0.96", "E2M3 ×6, E3M2 ×4"],
       ["Row-scaled 1e±2", "10/10", "0", "1.00", "E2M3"],
       ["Shuffled W_C", "10/10", "0", "1.00", "E2M3"]])
P("Ablation: when block maxima are left inside the histogram instead of being scored exactly, agreement on "
  "the outlier family drops to 6/10 (mean regret 0.24 dB). A single extreme value per block dominates "
  "that block's error, and the averaged noise model cannot capture it.")
d.add_heading("4.3 Cost", 2)
table([["", "Exhaustive (output error)", "Predictor"],
       ["work per format", "10,000 roundings + 2.56 M MACs", "1,028 bin evaluations + 400 roundings"],
       ["shared work", "–", "10,000 binning operations"],
       ["total (5 formats)", "≈ 12.9 M operations", "≈ 17 k operations"]])
P("Against exhaustive weight-MSE evaluation, which needs no matrix product, the saving is only about 3×. "
  "The predictor's advantage is that the per-format cost does not depend on the matrix size or on X, "
  "and it adds the output-awareness of H at no extra cost.")

# ---------------------------------------------------------------- 5
d.add_heading("5. Effect of outliers, sparsity and scale variation", 1)
fig("fig4_sensitivity.png", "Figure 4. Output SQNR of every format under controlled perturbations of W_C.")
bullets([
    "Outliers. Inside a 32-element block, the largest value sets the scale and the others share the remaining "
    "levels. E2M3 spends its levels near the top of the binade (fine relative steps from 1 to 8), "
    "so it degrades gracefully. Clipping W_C at 3σ improves every format by about 1 dB but does not change "
    "the winner. Extreme outliers (×50) shrink E2M3's lead, and in 4 of 10 synthetic trials E3M2's range wins.",
    "Sparsity. W_C has no zeros. With synthetic sparsity, blocks contain a few non-zeros whose magnitudes are "
    "close to the block max, so a uniform grid fits them better. At 90% sparsity MXINT6 overtakes E2M3. "
    "Exact zeros are representable in every format.",
    "Scale variation. Per-block power-of-two scaling absorbs row or column scale differences of up to 10^±2 "
    "with almost no change in SQNR. Within-block variation (outliers) is what matters.",
    "Distribution shape. Uniform data favours MXINT6 by more than 4 dB. Gaussian, Laplace and Student-t data "
    "favour E2M3. This is consistent with the INT-vs-FP analyses of Kuzmin et al. and van Baalen et al.",
])

# ---------------------------------------------------------------- 6
d.add_heading("6. Recommendation", 1)
P("Use MXFP6 E2M3 (block 32, E8M0 shared scale) for W_C. It gives the lowest output error (30.2 dB SQNR, "
  "3.1% relative error) at 6.32 bits/element, and the histogram predictor selects it without "
  "quantizing any candidate. MXINT6/BFP6 are the runner-up. E4M1 should not be used for this data: "
  "it has too little mantissa precision and clips 2.45% of the weights under the default scale.")

d.add_heading("7. Limitations", 1)
bullets(["Conventions for E4M1, MXINT6 and BFP6 are assumptions until checked against the classroom definitions.",
         "The predictor ignores off-diagonal terms of XXᵀ. This is fine for a near-white X but should be "
         "revisited for strongly correlated inputs.",
         "The uniform-noise model is an average. Exact scoring of block maxima corrects the worst case "
         "(outliers), but small blocks with few distinct values can still deviate.",
         "Only one weight matrix is available. The synthetic families are used to test generality."])

d.add_heading("8. Reproducibility", 1)
P("python3 validate_predictor.py  ·  python3 explore.py  ·  python3 make_figures.py   (numpy; matplotlib for figures)")

d.add_heading("References", 1)
for r in [
    "Open Compute Project, OCP Microscaling Formats (MX) Specification v1.0, 2023.",
    "B. D. Rouhani et al., Microscaling Data Formats for Deep Learning, arXiv:2310.10537, 2023.",
    "B. D. Rouhani et al., With Shared Microexponents, A Little Shifting Goes a Long Way, ISCA 2023.",
    "A. Kuzmin et al., FP8 Quantization: The Power of the Exponent, NeurIPS 2022.",
    "M. van Baalen et al., FP8 versus INT8 for Efficient Deep Learning Inference, arXiv:2303.17951, 2023.",
    "Y. Zhang et al., Integer or Floating Point? New Outlooks for Low-Bit Quantization on LLMs, arXiv:2305.12356, 2023.",
    "V. Egiazarian et al., Bridging the Gap Between Promise and Performance for Microscaling FP4 Quantization, ICLR 2026.",
]: d.add_paragraph(r, style="List Number")

out = os.path.join(HERE, "report_draft.docx"); d.save(out); print("saved", out)
