"""Shared-exponent (scale) policies for the scale-optimization study.  Owner: Anuj.

Written from the team conventions (plan section 2) WITHOUT importing formats.py, so that
scale_study.py can cross-check the two implementations.  The method differs on purpose:
  * element rounding is a table lookup on the grid of values enumerated from the bit
    codes (formats.py computes the local step with log2);
  * floor(log2 amax) comes from frexp, which is exact (formats.py uses np.log2);
  * "ceil" is derived as e_ocp or e_ocp + 1 (formats.py uses ceil(log2(amax / vmax))).

A format is an element grid g (sorted non-negative values) plus the element exponent
emax used by the class rule, the block size and the allowed shared-exponent range.
An element decodes to  sign * g[code] * 2^e,  e = the shared exponent of its block.

Policies, per block, with e_ocp = floor(log2 amax) - emax:
  ocp     e_ocp                                                              [CLASS]
  ceil    e_ocp + 1 if the block max would clip at e_ocp, else e_ocp (never clips)
  opt     e_ocp + d for d in a window (default -2..+1), keep the lowest measured error:
          weight SSE, or r^T H_bb r when H = X X^T is given (block-diagonal slice).
          Ties keep the smallest |d|, then the negative d (the formats.py convention).
  model   same window, but the error is predicted without rounding anything: exact for
          clipped and flushed-to-zero elements, step^2 / 12 for the others.
  joint_search()  per row, all block exponents chosen together for r^T H r with the
          FULL H (cross-block terms included); the output-error oracle.
"""
import itertools
from typing import NamedTuple

import numpy as np

E8M0 = (-127, 127)
OPT_WINDOW = (-2, -1, 0, 1)
WIDE_WINDOW = (-3, -2, -1, 0, 1, 2)
ORACLE_WINDOW = tuple(range(-8, 9))


# --------------------------------------------------------------------------- element grids
def fp_grid(ebits, mbits, bias, vmax):
    """Non-negative values of a sign-magnitude FP format with subnormals, up to vmax.

    The list is in code order (exponent field, then mantissa field) and strictly
    increasing, so a value's index has the parity of its mantissa code: rounding ties
    to the even index is ties-to-even.
    """
    vals = []
    for E in range(2 ** ebits):
        for M in range(2 ** mbits):
            lead, ex = (0, 1 - bias) if E == 0 else (1, E - bias)
            v = (lead + M / 2 ** mbits) * 2.0 ** ex
            if v <= vmax:
                vals.append(v)
    return np.array(vals)


def int_grid(kmax, frac_bits):
    """k * 2^-frac_bits for k = 0..kmax (the index is k, so even index = even code)."""
    return np.arange(kmax + 1) * 2.0 ** -frac_bits


def round_to_grid(x, grid):
    """Nearest grid value, ties to the even index, saturate at grid[-1]; keeps the sign."""
    a = np.abs(x)
    i = np.clip(np.searchsorted(grid, a, side="right") - 1, 0, len(grid) - 1)
    j = np.minimum(i + 1, len(grid) - 1)
    lo, hi = grid[i], grid[j]
    up = (hi - a < a - lo) | ((hi - a == a - lo) & (i % 2 == 1))
    return np.copysign(np.where(up, hi, lo), x)


class Fmt(NamedTuple):
    grid: np.ndarray        # non-negative element values, sorted
    emax: int               # element exponent used by the class rule
    block: int              # values per shared exponent
    e_range: tuple          # allowed shared exponents
    e_bits: int             # bits that store one shared exponent
    default: str = "ocp"    # policy used when none is given


FMTS = {
    "MXFP6_E3M2": Fmt(fp_grid(3, 2, 3, 28.0), 4, 32, E8M0, 8),
    "MXFP6_E2M3": Fmt(fp_grid(2, 3, 1, 7.5), 2, 32, E8M0, 8),
    "MXFP6_E4M1": Fmt(fp_grid(4, 1, 7, 384.0), 8, 32, E8M0, 8),      # D3: no reserved code
    "MXINT6":     Fmt(int_grid(31, 4), 0, 32, E8M0, 8),
    "BFP6":       Fmt(int_grid(31, 4), 0, 16, (-8, 7), 4),           # D1: 16|4|5, k*2^(e-4)
}
VARIANTS = {
    "MXFP6_E4M1_v192":   Fmt(fp_grid(4, 1, 7, 192.0), 7, 32, E8M0, 8),  # top exponent reserved
    "MXFP6_E4M1_nan256": Fmt(fp_grid(4, 1, 7, 256.0), 8, 32, E8M0, 8),  # one NaN (E4M3-style)
    "BFP6_b32ceil":      Fmt(int_grid(31, 4), 0, 32, E8M0, 8, "ceil"),  # formats.py's BFP6 today
}


def spec(name):
    if name in FMTS:
        return FMTS[name]
    if name in VARIANTS:
        return VARIANTS[name]
    raise KeyError(f"unknown format {name!r}")


def block_slices(n, block):
    return [slice(s, min(s + block, n)) for s in range(0, n, block)]


def bits_per_elem(name, n_cols=100, elem_bits=6):
    """True storage per real element; every block, the tail included, has its own exponent."""
    f = spec(name)
    return elem_bits + len(block_slices(n_cols, f.block)) * f.e_bits / n_cols


# --------------------------------------------------------------------------- shared exponent
def floor_log2(a):
    """floor(log2 a) for a > 0, exactly: frexp gives a = m * 2^k with m in [0.5, 1)."""
    return np.frexp(a)[1] - 1


def base_exp(f, amax, rule="ocp"):
    """Unclamped class-rule exponent per block ("ocp"), or the no-clip exponent ("ceil").

    The block max sits in [2^emax, 2^(emax+1)) * 2^e_ocp and vmax >= 2^emax, so the
    smallest exponent that clips nothing is e_ocp or e_ocp + 1. All-zero block: 0.
    """
    pos = amax > 0
    e = floor_log2(np.where(pos, amax, 1.0)) - f.emax
    if rule == "ceil":
        e = e + (amax > np.ldexp(f.grid[-1], e))
    elif rule != "ocp":
        raise ValueError(rule)
    return np.where(pos, e, 0)


def qblock(f, Wb, e):
    """Quantize the rows of one block, row i with shared exponent e[i]."""
    s = np.ldexp(1.0, e)[:, None]
    return round_to_grid(Wb / s, f.grid) * s


def _sse(r, Hbb):
    return (r * r).sum(1) if Hbb is None else np.einsum("ij,jk,ik->i", r, Hbb, r)


def local_step(f, a):
    """Width of the grid cell that holds the element magnitude a (the top cell above vmax)."""
    g = f.grid
    i = np.clip(np.searchsorted(g, a, side="right") - 1, 0, len(g) - 2)
    return g[i + 1] - g[i]


def model_err(f, Wb, e, h=None):
    """Predicted per-row error of a block at exponents e, without rounding.

    clipped (|w|/s > vmax): exact (|w| - vmax s)^2;  flushed (|w|/s < g[1]/2): exact w^2;
    otherwise uniform rounding noise step^2 / 12 with the step of the grid cell that
    holds |w|/s.  h = diag(X X^T) for this block weights each column (None = 1).
    """
    s = np.ldexp(1.0, e)[:, None]
    a = np.abs(Wb) / s
    g = f.grid
    step = local_step(f, a)
    err = np.where(a > g[-1], (a - g[-1]) ** 2,
                   np.where(a < g[1] / 2, a ** 2, step ** 2 / 12)) * s ** 2
    return err.sum(1) if h is None else err @ h


def _preferred(window):
    return sorted(window, key=lambda d: (abs(d), d))


def quantize(W, name, policy=None, window=OPT_WINDOW, H=None):
    """Quantize W row-wise in blocks.  Returns (Wq, exps), exps[i, b] = exponent of block b."""
    f = spec(name)
    W = np.asarray(W, dtype=np.float64)
    policy = policy or f.default
    if policy not in ("ocp", "ceil", "opt", "model"):
        raise ValueError(policy)
    offsets = _preferred(window) if policy in ("opt", "model") else [0]
    Wq = np.empty_like(W)
    exps = []
    for sl in block_slices(W.shape[1], f.block):
        Wb = W[:, sl]
        e0 = base_exp(f, np.abs(Wb).max(1), "ceil" if policy == "ceil" else "ocp")
        cand = np.stack([np.clip(e0 + d, *f.e_range) for d in offsets], 1)
        if len(offsets) == 1:
            pick = np.zeros(len(Wb), dtype=int)
        elif policy == "opt":
            Hbb = None if H is None else H[sl, sl]
            err = np.stack([_sse(Wb - qblock(f, Wb, c), Hbb) for c in cand.T], 1)
            pick = err.argmin(1)                  # first minimum = preferred offset
        else:
            h = None if H is None else np.diag(H)[sl]
            err = np.stack([model_err(f, Wb, c, h) for c in cand.T], 1)
            pick = err.argmin(1)
        e = cand[np.arange(len(Wb)), pick]
        Wq[:, sl] = qblock(f, Wb, e)
        exps.append(e)
    return Wq, np.stack(exps, 1)


# --------------------------------------------------------------------------- output oracle
def _cross_terms(W, f, H, window):
    """Residual candidates and G[b, k, c, l, i] = r_b^(k)T H_bc r_c^(l) for every row i."""
    sls = block_slices(W.shape[1], f.block)
    offsets = _preferred(window)
    cands, resid = [], []
    for sl in sls:
        Wb = W[:, sl]
        e0 = base_exp(f, np.abs(Wb).max(1), "ocp")
        c = np.stack([np.clip(e0 + d, *f.e_range) for d in offsets], 1)
        cands.append(c)
        resid.append([Wb - qblock(f, Wb, ck) for ck in c.T])
    nb, no = len(sls), len(offsets)
    G = np.empty((nb, no, nb, no, W.shape[0]))
    for b, sb in enumerate(sls):
        for k in range(no):
            P = resid[b][k] @ H[sb, :]
            for c, sc in enumerate(sls):
                for l in range(no):
                    G[b, k, c, l] = (P[:, sc] * resid[c][l]).sum(1)
    return sls, cands, G


def joint_search(W, name, H, window=WIDE_WINDOW, exact=False, max_sweeps=50):
    """Per row, choose all block exponents together to minimise r^T H r with the full H.

    Coordinate descent starting from the block-diagonal optimum (= policy "opt" with H):
    one block at a time is re-optimised with the others fixed, accepting only strict
    improvements, until a sweep changes nothing.  It is never worse than its start but
    is only a local optimum; exact=True enumerates every combination instead.
    Returns (Wq, exps, objective per row).
    """
    f = spec(name)
    W = np.asarray(W, dtype=np.float64)
    sls, cands, G = _cross_terms(W, f, H, window)
    nb, no, n = G.shape[0], G.shape[1], W.shape[0]
    rows = np.arange(n)
    diag = np.stack([G[b, :, b, :][np.arange(no), np.arange(no)] for b in range(nb)])
    choice = diag.argmin(1).T.copy()                       # (n, nb), block-diagonal start

    def total(ch):
        return sum(G[b, ch[:, b], c, ch[:, c], rows] for b in range(nb) for c in range(nb))

    if exact:
        best, best_val = choice.copy(), total(choice)
        for combo in itertools.product(range(no), repeat=nb):
            ch = np.broadcast_to(np.array(combo), (n, nb))
            v = total(ch)
            better = v < best_val
            best[better], best_val[better] = ch[better], v[better]
        choice = best
    else:
        for _ in range(max_sweeps):
            changed = False
            for b in range(nb):
                cost = np.stack([G[b, k, b, k] + 2 * sum(G[b, k, c, choice[:, c], rows]
                                                         for c in range(nb) if c != b)
                                 for k in range(no)], 1)
                cur = cost[rows, choice[:, b]]
                k_best = cost.argmin(1)
                move = cost[rows, k_best] < cur
                if move.any():
                    choice[move, b] = k_best[move]
                    changed = True
            if not changed:
                break
    Wq = np.empty_like(W)
    exps = np.stack([cands[b][rows, choice[:, b]] for b in range(nb)], 1)
    for b, sl in enumerate(sls):
        Wq[:, sl] = qblock(f, W[:, sl], exps[:, b])
    return Wq, exps, total(choice)


# --------------------------------------------------------------------------- diagnostics
def element_exps(exps, n_cols, block):
    """Expand per-block exponents to one exponent per element."""
    out = np.empty((exps.shape[0], n_cols), dtype=exps.dtype)
    for b, sl in enumerate(block_slices(n_cols, block)):
        out[:, sl] = exps[:, b:b + 1]
    return out


def error_parts(W, Wq, exps, name):
    """Split the weight squared error into clipped / flushed-to-zero / rounded elements."""
    f = spec(name)
    s = np.ldexp(1.0, element_exps(exps, W.shape[1], f.block))
    sat = np.abs(W) / s > f.grid[-1]
    zero = (W != 0) & (Wq == 0) & ~sat
    rnd = ~sat & ~zero
    d2 = (W - Wq) ** 2
    return dict(sse_clip=float(d2[sat].sum()), sse_zero=float(d2[zero].sum()),
                sse_round=float(d2[rnd].sum()), n_sat=int(sat.sum()), n_zero=int(zero.sum()))


def rel_err(W, Wq, X=None):
    """Relative Frobenius error of W (X None) or of Y = W @ X."""
    dW = np.asarray(W, float) - Wq
    if X is None:
        return float(np.linalg.norm(dW) / np.linalg.norm(W))
    return float(np.linalg.norm(dW @ X) / np.linalg.norm(W @ X))
