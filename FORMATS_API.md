# formats.py v1: how to use it (Akash)

Old calls still work unchanged: `quantize(W, name)` and `quantize(W, name, shift_search=2, H=H)` give the same matrices as before.

## Quantize

```python
from formats import FORMATS, VARIANTS, quantize, metrics

Wq, exps = quantize(W, "MXFP6_E2M3")                    # class rule ("ocp"), block 32
Wq, exps = quantize(W, "MXINT6", policy="ceil")         # "ocp" | "ceil" | "opt"
Wq, exps = quantize(W, "MXFP6_E3M2", policy="opt", H=X @ X.T)   # output-aware exponent search
Wq, exps, st = quantize(W, "BFP6", return_stats=True)   # + saturated / zeroed / subnormal counts, bits/elem
```

- `FORMATS`: the 5 mandatory candidates. `VARIANTS`: `MXFP6_E4M1_v192` (top exponent code reserved) and `BFP6_row` (one block per row).
- `exps[i, b]` is the shared exponent of block `b` in row `i`. It is clamped to [-127, 127], and an all-zero block gets 0.
- Policies: `ocp` = floor(log2 max) - emax (class), `ceil` = smallest scale that clips nothing, `opt` = try ocp + {-2, -1, 0, +1} per block and keep the lowest error.

## Measure

```python
m = metrics(W, Wq, X)   # w_rel_err, w_sqnr_db, y_rel_err, y_sqnr_db, y_cos_row_mean/min, zeroed_frac
```

## Ground truth for the selector (Aryan)

```python
from formats import exhaustive, ranking
rows = exhaustive(W, X)            # 5 formats x {ocp, ceil, opt, opt_out}; tqdm bar; held-out split
ranking(rows, policies=("ocp",))   # [(format, policy, y_rel_err), ...], best first
```

Or from the shell: `python run_baseline.py [--W path --tag A] [--variants]`. It writes `results/baseline_<tag>.csv/.json` and can resume from a checkpoint. Each row has `n_roundings` and `n_macs` for the cost model.

## Tests

`python test_formats.py` runs 17 tests. They check the FP grids bit-exactly, RNE ties, saturation, the zero block, the E8M0 clamp, the tail block of 4, sign symmetry, the policies, storage, the exhaustive harness with its checkpoint, and a regression against the plan's numbers on C.

## Findings for the report (dataset C)

| Y error (%) | ocp (class) | ceil | opt |
|---|---|---|---|
| **MXFP6_E2M3** | **3.079** | 3.103 | **3.058** |
| MXINT6 | 4.047 | 4.168 | 3.896 |
| BFP6 (its default rule is ceil) | 4.047 | 4.168 | 3.896 |
| MXFP6_E3M2 | 5.439 | 5.236 | 5.236 |
| MXFP6_E4M1 | 11.088 | 10.176 | 10.176 |
| BFP6_row (6.08 bits/elem) | 5.162 | 5.340 | 4.693 |

1. E2M3 wins under every policy, and the held-out X columns agree (3.10% under the class rule).
2. With block 32, BFP6 is exactly MXINT6 under the `ceil` policy, so it needs a different definition (D1) to be a separate candidate. `BFP6_row` is different but cheaper (6.08 bits/elem), so it is not an equal-budget comparison.
3. E4M1 gives the same result with vmax 384 or 192, because the shared scale absorbs the top exponent code. The D3 choice does not change the conclusion.
4. True storage is 6.32 bits/elem, not 6.25, because each row's 4-value tail block has its own 8-bit scale.
5. Under the class rule, E2M3 saturates 43 weights and zeroes 461; MXINT6 saturates 20 and zeroes 891.
