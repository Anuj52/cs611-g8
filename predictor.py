"""Histogram-based format predictor: choose a format without quantizing W.

Idea
----
Every MX format scales a block by a power of two tied to the block's max:
    scale_f = 2^(e_b - emax_f),  e_b = floor(log2 amax_b).
So if we normalise each value by its block's binade, u = |w| / 2^e_b in (0, 2),
the error a format makes on w is 2^(2 e_b) * g_f(u), where g_f is a fixed
function of the format alone (rounding noise step^2/12, or clipping error,
or flush-to-zero error).

The output error of Y = W X, assuming independent rounding errors, is
    E||dW X||_F^2  ~=  sum_ij  E[dW_ij^2] * (X X^T)_jj .
Putting the two together, everything data-dependent collapses into ONE
format-independent weighted histogram over u:
    weight of w_ij = 2^(2 e_b(i,j)) * H_jj,      H = X X^T
and the predicted output error of format f is
    P_f = sum_bins  hist[bin] * g_f(u_bin).
Building the histogram is one pass over W (shared by all formats); scoring a
format costs O(#bins), independent of the matrix size.

BFP6 picks its exponent so the block max is not clipped, which depends on
whether the block max u_max <= 31/16. We therefore keep two histograms,
split by that flag (still format-independent).
"""
import numpy as np
from formats import FORMATS, blocks, BLOCK, elem_round

U_LO, U_HI, NBINS = 2.0 ** -16, 2.0, 512
EDGES = np.geomspace(U_LO, U_HI, NBINS + 1)
CENTERS = np.sqrt(EDGES[:-1] * EDGES[1:])
BFP_SPLIT = 31 / 16


def build_stats(W, H=None, block=BLOCK):
    """One pass over W -> two weighted histograms of normalised magnitude u."""
    W = np.asarray(W, dtype=np.float64)
    h_col = np.ones(W.shape[1]) if H is None else np.diag(H)
    hist = np.zeros((2, NBINS + 2))          # [bfp_flag, bin] incl. under/overflow
    maxima = []                              # (u_max, weight, flag) per block, scored exactly
    for sl in blocks(W.shape[1], block):
        Wb = np.abs(W[:, sl])
        amax = Wb.max(1, keepdims=True)
        ok = amax[:, 0] > 0
        e = np.floor(np.log2(np.where(amax > 0, amax, 1.0)))
        u = Wb / 2.0 ** e
        wgt = (2.0 ** (2 * e)) * h_col[sl][None, :]
        flag = np.broadcast_to((u.max(1, keepdims=True) > BFP_SPLIT).astype(int), u.shape)
        # the block max is a single, heavily weighted value whose error is
        # deterministic, so it is kept out of the histogram and scored exactly
        jmax = u.argmax(1)
        r = np.arange(u.shape[0])
        maxima.append(np.stack([u[r, jmax], wgt[r, jmax], flag[r, 0]], 1)[ok])
        m = np.broadcast_to(ok[:, None], u.shape).copy()
        m[r, jmax] = False
        idx = np.digitize(u, EDGES)          # 0 = below U_LO (incl. exact zeros)
        np.add.at(hist, (flag[m], idx[m]), wgt[m])
    return hist, np.concatenate(maxima)


def g(name, u, flag, shift=0):
    """Expected squared error (in units of 2^e_b) at normalised magnitude u."""
    kind, p, emax = FORMATS[name]
    if kind == "bfp":
        s = 2.0 ** np.where(flag, -3, -4) * 2.0 ** shift   # relative scale, no clipping
        step, vmax = s, p["kmax"] * s
        v = u
    else:
        s = 2.0 ** (-emax + shift)                         # relative scale
        v = u / s
        if kind == "fp":
            ee = np.maximum(np.floor(np.log2(v)), 1 - p["bias"])
            step, vmax = 2.0 ** (ee - p["mbits"]), p["maxval"]
        else:
            step, vmax = 2.0 ** -p["frac_bits"], p["kmax"] * 2.0 ** -p["frac_bits"]
        v, step, vmax = v * s, step * s, vmax * s            # back to u units
    err = np.where(v > vmax, (v - vmax) ** 2,               # clipping
          np.where(v < step / 2, v ** 2,                    # flushed to zero
                   step ** 2 / 12))                         # uniform rounding noise
    return err


def exact_err(name, u, flag, shift=0):
    """Exact squared error (units of 2^e_b) of rounding normalised values u."""
    kind, p, emax = FORMATS[name]
    s = 2.0 ** (np.where(flag, -3, -4) + shift) if kind == "bfp" else 2.0 ** (-emax + shift)
    return (u - elem_round(kind, p, u / s) * s) ** 2


def predict(stats, names=None, shift=0, exact_max=True):
    """Predicted (weighted) squared error per format from the statistics alone."""
    hist, maxima = stats
    names = names or list(FORMATS)
    u = np.concatenate([[U_LO / 2], CENTERS, [U_HI]])     # under/overflow bins
    um, wm, fm = maxima.T
    out = {}
    for n in names:
        tot = sum((hist[f] * g(n, u, f, shift)).sum() for f in (0, 1))
        em = exact_err(n, um, fm, shift) if exact_max else g(n, um, fm, shift)
        out[n] = float(tot + (wm * em).sum())
    return out


def select(W, X, names=None):
    H = X @ X.T
    scores = predict(build_stats(W, H), names)
    return min(scores, key=scores.get), scores
