"""6-bit candidate formats for CS611 A1, Group 8 (dataset C).  Owner: Akash.

Conventions. "class" = Lecture 19 OCP-MX notebook; "team" = our decision (D1-D6),
to be stated and defended in the report.

  Blocks    32 consecutive values along each row of W, the reduction axis of Y = W @ X.
            K = 100 -> blocks of 32, 32, 32 and a tail of 4 real values. (class)
            Class zero-pads the tail; that gives exactly the same quantized values,
            because padded zeros never change a block's max. Statistics count only
            real elements. Storage counts the tail with its own 8-bit scale.
  Scale     power of two 2^e, e stored as E8M0, clamped to [-127, 127]. (class)
            An all-zero block gets e = 0, i.e. scale 1. (class)
  Policies  "ocp"   e = floor(log2 amax) - emax_elem                 (class default)
            "ceil"  e = ceil(log2(amax / vmax_elem)): the smallest power-of-two
                    scale that clips nothing (class suggests ceil as a follow-up)
            "opt"   per block, try e_ocp + {-2, -1, 0, +1}, keep the lowest error:
                    weight MSE, or output-aware r^T H_bb r with H = X X^T if H is given.
                    Ties keep the offset closest to 0.
  Elements  round to nearest, ties to even, then saturate at vmax (no Inf/NaN). (class)
            FP subnormals: fixed spacing 2^(emin - m) below 2^emin. (class)

  MXFP6_E3M2       m=2, bias 3, emin=-2, emax=4, vmax=28                         (class)
  MXFP6_E2M3       m=3, bias 1, emin=0,  emax=2, vmax=7.5                        (class)
  MXFP6_E4M1       m=1, bias 7, emin=-6, emax=8, vmax=1.5*2^8=384, top code finite (team, D3)
  MXFP6_E4M1_v192  variant: top exponent code reserved -> emax=7, vmax=192         (team, D3)
  MXINT6           k * 2^-4 with k in [-31, 31]; the -32 code is never produced,
                   vmax = 31/16, emax_elem = 0 (OCP MXINT8 scaled down)          (team)
  BFP6             sign + 5-bit magnitude, k in [-31, 31], shared exponent by
                   "ceil" (block max never clipped), RNE, block 32   (team, D1 default proposal)
  BFP6_row         variant: one block per row (no partial tail block)       (team, D1 variant)

  NOTE (report this): with block 32 and the "ceil" policy, BFP6 and MXINT6 use
  exactly the same grid, so their results are identical. Only the block
  organisation (BFP6_row) or a different exponent rule separates them.

Public API (everything the other scripts use):
  FORMATS, VARIANTS, POLICIES, BLOCK, spec(name)
  quantize(W, name, block=None, shift_search=0, H=None, policy=None, return_stats=False)
  metrics(W, Wq, X), split_X(X), exhaustive(W, X, ...), ranking(rows), save_results(rows, path)
  storage_bits(n_rows, n_cols, block), blocks(n, block), default_exp(...), elem_round(...)
"""
import csv
import json
import os
import time

import numpy as np

try:                                    # progress bars if tqdm is installed, silent otherwise
    from tqdm.auto import tqdm
except ImportError:                     # pragma: no cover
    def tqdm(it, **_):
        return it

BLOCK = 32
E8M0_MIN, E8M0_MAX = -127, 127
OPT_OFFSETS = (-2, -1, 0, 1)
POLICIES = ("ocp", "ceil", "opt")


# --------------------------------------------------------------------------- exact log2
def floor_log2(x):
    """Exact floor(log2(x)) for x > 0.

    np.floor(np.log2(x)) is wrong for float64 values one ulp below a power of two:
    log2 rounds up to the integer, e.g. floor(log2(8 - ulp)) gives 3 instead of 2.
    frexp splits x = m * 2^k with m in [0.5, 1) exactly, so floor(log2 x) = k - 1.
    """
    return (np.frexp(x)[1] - 1).astype(np.float64)


# --------------------------------------------------------------------------- elements
def fp_round(v, ebits, mbits, bias, maxval):
    """Round to a sign-magnitude FP grid with subnormals: RNE, then saturate at maxval.

    ebits is not used in the arithmetic (bias and maxval fix the range); it is kept so
    the parameter set reads like the format name.
    """
    a = np.abs(v)
    emin = 1 - bias
    e = floor_log2(np.where(a > 0, a, 1.0))
    e = np.maximum(e, emin)              # below 2^emin: subnormal, fixed spacing
    step = 2.0 ** (e - mbits)
    q = np.round(a / step) * step        # np.round is round-half-to-even
    q = np.minimum(q, maxval)            # saturate: no Inf/NaN codes
    return np.copysign(q, v)


def int_round(v, frac_bits, kmax):
    """Fixed point k * 2^-frac_bits with symmetric k in [-kmax, kmax], RNE."""
    step = 2.0 ** -frac_bits
    return np.clip(np.round(v / step), -kmax, kmax) * step


FORMATS = {
    # name: (kind, element params, emax_elem used by the "ocp" scale rule)
    "MXFP6_E3M2": ("fp", dict(ebits=3, mbits=2, bias=3, maxval=28.0), 4),
    "MXFP6_E2M3": ("fp", dict(ebits=2, mbits=3, bias=1, maxval=7.5), 2),
    "MXFP6_E4M1": ("fp", dict(ebits=4, mbits=1, bias=7, maxval=384.0), 8),
    "MXINT6":     ("int", dict(frac_bits=4, kmax=31), 0),
    "BFP6":       ("bfp", dict(kmax=31), 4),   # 31 = 1.9375 * 2^4
}
# Variants are kept out of FORMATS so that code looping over FORMATS still sees
# exactly the five mandatory candidates.
VARIANTS = {
    "MXFP6_E4M1_v192": ("fp", dict(ebits=4, mbits=1, bias=7, maxval=192.0), 7),
    "BFP6_row":        ("bfp", dict(kmax=31), 4),
}
DEFAULT_POLICY = {"BFP6": "ceil", "BFP6_row": "ceil"}   # everything else: "ocp"
DEFAULT_BLOCK = {"BFP6_row": "row"}                      # everything else: BLOCK


def spec(name):
    """(kind, params, emax_elem) for a format or variant name."""
    if name in FORMATS:
        return FORMATS[name]
    if name in VARIANTS:
        return VARIANTS[name]
    raise KeyError(f"unknown format {name!r}; choose from {list(FORMATS) + list(VARIANTS)}")


def vmax_elem(kind, p):
    """Largest element magnitude before scaling."""
    if kind == "fp":
        return p["maxval"]
    if kind == "int":
        return p["kmax"] * 2.0 ** -p["frac_bits"]
    return float(p["kmax"])


def elem_round(kind, p, v):
    if kind == "fp":
        return fp_round(v, **p)
    if kind == "int":
        return int_round(v, **p)
    if kind == "bfp":
        return np.clip(np.round(v), -p["kmax"], p["kmax"])
    raise ValueError(kind)


# --------------------------------------------------------------------------- blocks + scale
def blocks(n, block=BLOCK):
    """Column slices of the blocks of a row of length n ("row" or None = one block)."""
    if block in (None, "row"):
        block = n
    return [slice(s, min(s + block, n)) for s in range(0, n, block)]


def default_exp(kind, p, emax, amax, policy=None):
    """Shared exponent (log2 of the scale) for blocks with max |w| = amax.

    policy None -> the format's own rule ("ceil" for BFP, "ocp" otherwise).
    All-zero blocks get 0; every exponent is clamped to the E8M0 range.
    """
    amax = np.asarray(amax, dtype=np.float64)
    if policy is None:
        policy = "ceil" if kind == "bfp" else "ocp"
    pos = amax > 0
    safe = np.where(pos, amax, 1.0)
    if policy == "ocp":
        e = floor_log2(safe) - emax
    elif policy == "ceil":
        # smallest e with amax / 2^e <= vmax. log2 of the ratio can be off by one
        # near a power of two, so correct it with exact power-of-two divisions.
        vmax = vmax_elem(kind, p)
        e = np.ceil(np.log2(safe / vmax))
        e = np.where(safe / 2.0 ** e > vmax, e + 1, e)
        e = np.where(safe / 2.0 ** (e - 1) <= vmax, e - 1, e)
    else:
        raise ValueError(f"default_exp needs policy 'ocp' or 'ceil', got {policy!r}")
    return np.clip(np.where(pos, e, 0.0), E8M0_MIN, E8M0_MAX)


def storage_bits(n_rows, n_cols, block=BLOCK, elem_bits=6, scale_bits=8):
    """(total bits, bits per real element); every block, the tail included, has one scale."""
    nblk = n_rows * len(blocks(n_cols, block))
    total = n_rows * n_cols * elem_bits + nblk * scale_bits
    return total, total / (n_rows * n_cols)


# --------------------------------------------------------------------------- quantize
def quantize(W, name, block=None, shift_search=0, H=None, policy=None, return_stats=False):
    """Quantize W row-wise in blocks along its columns.

    name          a key of FORMATS or VARIANTS
    block         block size; None = the format's default (32, or the whole row for BFP6_row)
    policy        "ocp" | "ceil" | "opt"; None = the format's default
    shift_search  legacy option: also try base exponent +-k (ignored when policy="opt")
    H             if given (H = X X^T), the exponent search minimises the output-aware
                  error r^T H_bb r per block instead of the weight MSE. It uses only
                  the block's own diagonal slice of H, which is exact when X X^T is
                  diagonal and an approximation otherwise.
    Returns (Wq, exps), or (Wq, exps, stats) with return_stats=True;
    exps[i, b] = chosen exponent of block b in row i.
    """
    kind, p, emax = spec(name)
    W = np.asarray(W, dtype=np.float64)
    if W.ndim != 2:
        raise ValueError("W must be a 2-D matrix")
    policy = policy or DEFAULT_POLICY.get(name, "ocp")
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    block = block or DEFAULT_BLOCK.get(name, BLOCK)
    if block == "row":
        block = W.shape[1]

    if policy == "opt":
        base, offsets = "ocp", OPT_OFFSETS
    else:
        base, offsets = policy, range(-shift_search, shift_search + 1)
    offsets = sorted(offsets, key=lambda d: (abs(d), d))   # 0 first, so ties keep the base rule

    vmax = vmax_elem(kind, p)
    Wq = np.empty_like(W)
    exps = []
    n_sat = n_zero = n_sub = 0
    for sl in blocks(W.shape[1], block):
        Wb = W[:, sl]
        e0 = default_exp(kind, p, emax, np.abs(Wb).max(axis=1), policy=base)
        best_err = np.full(W.shape[0], np.inf)
        best_q = np.zeros_like(Wb)
        best_e = e0.copy()
        for d in offsets:
            e = np.clip(e0 + d, E8M0_MIN, E8M0_MAX)
            s = (2.0 ** e)[:, None]
            q = elem_round(kind, p, Wb / s) * s
            r = Wb - q
            err = (r * r).sum(1) if H is None else np.einsum("ij,jk,ik->i", r, H[sl, sl], r)
            better = err < best_err
            best_err[better] = err[better]
            best_q[better] = q[better]
            best_e[better] = e[better]
        Wq[:, sl] = best_q
        exps.append(best_e)
        if return_stats:
            s = (2.0 ** best_e)[:, None]
            n_sat += int((np.abs(Wb) / s > vmax).sum())
            n_zero += int(((Wb != 0) & (best_q == 0)).sum())
            if kind == "fp":
                small = np.abs(best_q) / s < 2.0 ** (1 - p["bias"])
                n_sub += int(((best_q != 0) & small).sum())
    exps = np.stack(exps, 1)
    if not return_stats:
        return Wq, exps

    total, per = storage_bits(W.shape[0], W.shape[1], block)
    n_nonzero = int((W != 0).sum())
    stats = dict(
        format=name, policy=policy, block=int(block),
        n_elements=int(W.size), n_blocks=int(exps.size),
        n_saturated=n_sat,                    # |w| / scale > vmax before rounding
        n_zeroed=n_zero,                      # non-zero weights that became 0
        zeroed_frac=n_zero / max(n_nonzero, 1),
        n_subnormal=n_sub,                    # FP only: non-zero outputs below 2^emin
        bits_total=int(total), bits_per_elem=float(per),
        n_roundings=int(W.size * len(offsets)),   # element quantizations performed
        exp_min=int(exps.min()), exp_max=int(exps.max()),
    )
    return Wq, exps, stats


# --------------------------------------------------------------------------- metrics
def metrics(W, Wq, X):
    """Errors on W and on Y = W @ X. Relative errors are Frobenius-norm ratios."""
    W = np.asarray(W, dtype=np.float64)
    Wq = np.asarray(Wq, dtype=np.float64)
    X = np.asarray(X, dtype=np.float64)
    dW = W - Wq
    Y = W @ X
    dY = dW @ X
    Yq = Y - dY
    with np.errstate(divide="ignore", invalid="ignore"):
        ny, nq = np.linalg.norm(Y, axis=1), np.linalg.norm(Yq, axis=1)
        den = ny * nq
        # both rows zero -> 1 (identical); exactly one zero -> 0 (nothing in common)
        cos = np.where(den > 0, (Y * Yq).sum(1) / np.where(den > 0, den, 1.0),
                       np.where((ny == 0) & (nq == 0), 1.0, 0.0))
        nz = W != 0
        return dict(
            w_mse=float((dW ** 2).mean()),
            w_rel_err=float(np.linalg.norm(dW) / np.linalg.norm(W)),
            w_sqnr_db=float(10 * np.log10((W ** 2).sum() / (dW ** 2).sum())),
            y_rel_err=float(np.linalg.norm(dY) / np.linalg.norm(Y)),
            y_sqnr_db=float(10 * np.log10((Y ** 2).sum() / (dY ** 2).sum())),
            y_cos_row_mean=float(cos.mean()),
            y_cos_row_min=float(cos.min()),
            zeroed_frac=float((nz & (Wq == 0)).sum() / max(nz.sum(), 1)),
        )


def split_X(X):
    """Deterministic calibration split: (even columns to fit, odd columns held out)."""
    return X[:, ::2], X[:, 1::2]


# --------------------------------------------------------------------------- exhaustive
EXHAUSTIVE_POLICIES = ("ocp", "ceil", "opt", "opt_out")


def exhaustive(W, X, names=None, policies=EXHAUSTIVE_POLICIES, block=None,
               checkpoint=None, progress=True):
    """Ground truth for the selector: quantize W with every format x policy.

    policies  "ocp", "ceil", "opt" (exponent search on weight MSE, uses no X) and
              "opt_out" (exponent search on the output-aware error, H fitted on the
              even columns of X only).
    Each row reports errors on all of X and on the held-out odd columns, the
    quantizer statistics, the cost (element roundings + MACs) and the wall time.
    checkpoint  path of a .jsonl file: finished rows are appended as they complete
                and reused on a rerun, so an interrupted run resumes.
    """
    names = list(names or FORMATS)
    X = np.asarray(X, dtype=np.float64)
    X_fit, X_eval = split_X(X)
    H_fit = X_fit @ X_fit.T
    done = {}
    if checkpoint and os.path.exists(checkpoint):
        with open(checkpoint) as f:
            for line in f:
                r = json.loads(line)
                done[(r["format"], r["policy"])] = r
    jobs = [(n, pol) for n in names for pol in policies]
    rows = []
    for n, pol in tqdm(jobs, desc="exhaustive", disable=not progress):
        if (n, pol) in done:
            rows.append(done[(n, pol)])
            continue
        t0 = time.perf_counter()
        if pol == "opt_out":
            Wq, _, st = quantize(W, n, block=block, policy="opt", H=H_fit, return_stats=True)
        else:
            Wq, _, st = quantize(W, n, block=block, policy=pol, return_stats=True)
        m_all, m_ho = metrics(W, Wq, X), metrics(W, Wq, X_eval)
        row = dict(
            format=n, policy=pol,
            y_rel_err=m_all["y_rel_err"], y_rel_err_heldout=m_ho["y_rel_err"],
            y_sqnr_db=m_all["y_sqnr_db"], w_rel_err=m_all["w_rel_err"],
            w_sqnr_db=m_all["w_sqnr_db"], y_cos_row_min=m_all["y_cos_row_min"],
            n_saturated=st["n_saturated"], n_zeroed=st["n_zeroed"],
            n_subnormal=st["n_subnormal"], bits_per_elem=st["bits_per_elem"],
            n_roundings=st["n_roundings"],
            n_macs=int(W.shape[0] * W.shape[1] * X.shape[1]),   # for (W - Wq) @ X
            seconds=time.perf_counter() - t0,
        )
        rows.append(row)
        if checkpoint:
            os.makedirs(os.path.dirname(os.path.abspath(checkpoint)), exist_ok=True)
            with open(checkpoint, "a") as f:
                f.write(json.dumps(row) + "\n")
    return rows


def ranking(rows, metric="y_rel_err", policies=None):
    """Best policy per format, sorted best first: [(format, policy, value), ...]."""
    best = {}
    for r in rows:
        if policies and r["policy"] not in policies:
            continue
        if r["format"] not in best or r[metric] < best[r["format"]][1]:
            best[r["format"]] = (r["policy"], r[metric])
    return sorted(((f, p, v) for f, (p, v) in best.items()), key=lambda t: t[2])


def save_results(rows, path):
    """Write rows to <path>.csv and <path>.json."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(path + ".json", "w") as f:
        json.dump(rows, f, indent=1)
