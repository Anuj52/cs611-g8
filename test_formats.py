"""Unit tests for formats.py.  Run:  python test_formats.py   (or: pytest test_formats.py)

The FP checks compare against a grid built independently from the bit codes
(exponent field x mantissa field), not against fp_round's own arithmetic.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from formats import (FORMATS, VARIANTS, E8M0_MIN, E8M0_MAX, spec, fp_round, int_round,  # noqa: E402
                     quantize, metrics, storage_bits, exhaustive, ranking, split_X, vmax_elem)

FP_NAMES = ["MXFP6_E3M2", "MXFP6_E2M3", "MXFP6_E4M1", "MXFP6_E4M1_v192"]
ALL = list(FORMATS) + list(VARIANTS)
rng = np.random.default_rng(0)


# ---------------------------------------------------------------- helpers
def bit_grid(ebits, mbits, bias, maxval):
    """Every non-negative value the format can encode, with its mantissa code."""
    vals, mant = [], []
    for E in range(2 ** ebits):
        for M in range(2 ** mbits):
            if E == 0:
                v = (M / 2 ** mbits) * 2.0 ** (1 - bias)          # subnormal
            else:
                v = (1 + M / 2 ** mbits) * 2.0 ** (E - bias)       # normal
            if v <= maxval:
                vals.append(v)
                mant.append(M)
    return np.array(vals), np.array(mant)


def ref_round(x, grid, mant):
    """Nearest grid value; ties go to the even mantissa code; saturate at the top."""
    out = np.empty_like(x)
    for i, v in enumerate(x):
        a = min(abs(v), grid.max())
        d = np.abs(grid - a)
        best = np.flatnonzero(d == d.min())
        j = best[0] if len(best) == 1 else [b for b in best if mant[b] % 2 == 0][0]
        out[i] = np.copysign(grid[j], v)
    return out


def row(*vals, n=32):
    w = np.zeros((1, n))
    w[0, :len(vals)] = vals
    return w


# ---------------------------------------------------------------- element level
def test_fp_grid_limits():
    expect = {"MXFP6_E3M2": (28, 0.0625, 4), "MXFP6_E2M3": (7.5, 0.125, 2),
              "MXFP6_E4M1": (384, 2 ** -7, 8), "MXFP6_E4M1_v192": (192, 2 ** -7, 7)}
    for n in FP_NAMES:
        _, p, emax = spec(n)
        g, _ = bit_grid(**p)
        vmax, min_sub, e = expect[n]
        assert g.max() == vmax == p["maxval"], n
        assert g[g > 0].min() == min_sub, n
        assert emax == e and g.max() < 2.0 ** (emax + 1) and g.max() >= 2.0 ** emax, n


def test_fp_round_matches_bit_grid():
    for n in FP_NAMES:
        _, p, _ = spec(n)
        g, mant = bit_grid(**p)
        mids = (g[:-1] + g[1:]) / 2                               # exact RNE ties
        x = np.concatenate([rng.uniform(-1.3 * p["maxval"], 1.3 * p["maxval"], 5000),
                            mids, -mids, g, -g, [g[1] / 2, g[1] / 2 * 0.999, 1e-30, 1e9, -1e9]])
        assert np.array_equal(fp_round(x, **p), ref_round(x, g, mant)), n


def test_saturation_no_inf_nan():
    for n in FP_NAMES:
        _, p, _ = spec(n)
        q = fp_round(np.array([1e30, -1e30]), **p)
        assert np.all(np.isfinite(q)) and q[0] == p["maxval"] and q[1] == -p["maxval"], n


def test_mxint6_codes_and_ties():
    x = np.concatenate([rng.uniform(-3, 3, 5000), (np.arange(-63, 64) + 0.5) / 16])
    k = int_round(x, 4, 31) * 16
    assert np.array_equal(k, np.round(k))
    assert k.min() == -31 and k.max() == 31                       # -32 never produced
    assert list(int_round(np.array([0.5, 1.5, 2.5, -0.5]) / 16, 4, 31) * 16) == [0, 2, 2, 0]


# ---------------------------------------------------------------- scale + blocks
def test_all_zero_block_scale_one():
    for n in ALL:
        Wq, e, st = quantize(np.zeros((3, 100)), n, return_stats=True)
        assert np.all(Wq == 0) and np.all(e == 0), n
        for pol in ("ceil", "opt"):
            assert np.all(quantize(np.zeros((2, 64)), n, policy=pol)[1] == 0), (n, pol)


def test_exponent_clamped_to_e8m0():
    W = np.array([[2.0 ** -200] * 32, [2.0 ** 200] * 32])
    for n in ALL:
        for pol in ("ocp", "ceil", "opt"):
            e = quantize(W, n, policy=pol)[1]
            assert e.min() >= E8M0_MIN and e.max() <= E8M0_MAX, (n, pol)


def test_power_of_two_block_max():
    # block max exactly 4 = 2^2: exactly representable under the class rule
    for n in ["MXFP6_E3M2", "MXFP6_E2M3", "MXFP6_E4M1", "MXINT6", "BFP6"]:
        Wq, _ = quantize(row(4.0, -4.0, 1.0), n)
        assert Wq[0, 0] == 4.0 and Wq[0, 1] == -4.0, n


def test_just_below_power_of_two():
    # max = 3.999: the "ocp" floor rule saturates it for E2M3 (7.5 * 2^-1 = 3.75);
    # "ceil" picks the next scale up and keeps it (rounds to 4.0). Known trade-off.
    assert quantize(row(3.999), "MXFP6_E2M3")[0][0, 0] == 3.75
    Wq, _, st = quantize(row(3.999), "MXFP6_E2M3", policy="ceil", return_stats=True)
    assert Wq[0, 0] == 4.0 and st["n_saturated"] == 0


def test_one_ulp_below_power_of_two():
    # np.floor(np.log2(x)) rounds up here for float64; the scale must still follow the rule
    for k in range(-10, 12):
        a = np.nextafter(2.0 ** k, 0)
        for n in ["MXFP6_E3M2", "MXFP6_E2M3", "MXFP6_E4M1", "MXINT6"]:
            kind, p, emax = spec(n)
            e = quantize(row(a), n, policy="ocp")[1][0, 0]
            assert e == (k - 1) - emax, (n, k, e)
    # E2M3, block max 8 - ulp: scale 1 (not 2), so the max saturates to 7.5
    assert quantize(row(np.nextafter(8.0, 0)), "MXFP6_E2M3")[0][0, 0] == 7.5
    # ceil: a max one ulp above vmax * 2^k must move to the next scale (nothing clipped)
    for n in ["MXFP6_E2M3", "MXINT6", "BFP6"]:
        kind, p, _ = spec(n)
        vm = vmax_elem(kind, p)
        for k in range(-5, 6):
            st = quantize(row(np.nextafter(vm * 2.0 ** k, np.inf)), n, policy="ceil", return_stats=True)[2]
            assert st["n_saturated"] == 0, (n, k)


def test_rne_tie_inside_quantize():
    # E2M3, block max 7.5 -> scale 1. 1.0625 is a tie between 1.0 and 1.125 -> even code (1.0)
    Wq, e = quantize(row(7.5, 1.0625, 1.1875), "MXFP6_E2M3")
    assert e[0, 0] == 0 and Wq[0, 1] == 1.0 and Wq[0, 2] == 1.25


def test_tail_block_of_four():
    W = rng.standard_normal((5, 100))
    W[:, 96:] *= 0.01                                            # tiny tail
    Wq, e, st = quantize(W, "MXFP6_E2M3", return_stats=True)
    assert e.shape == (5, 4)                                     # 32 + 32 + 32 + 4
    expect = np.floor(np.log2(np.abs(W[:, 96:]).max(1))) - 2     # tail scale from its own 4 values
    assert np.array_equal(e[:, 3], expect)
    assert st["n_elements"] == 500                               # no padded elements counted
    assert abs(st["bits_per_elem"] - 6.32) < 1e-12               # tail pays its own 8-bit scale


def test_sign_symmetry():
    W = rng.laplace(size=(20, 100))
    for n in ALL:
        for pol in ("ocp", "ceil", "opt"):
            assert np.array_equal(quantize(-W, n, policy=pol)[0], -quantize(W, n, policy=pol)[0]), (n, pol)


# ---------------------------------------------------------------- policies
def test_ceil_never_saturates_and_opt_never_worse():
    W = rng.standard_t(3, (30, 100))
    for n in ALL:
        assert quantize(W, n, policy="ceil", return_stats=True)[2]["n_saturated"] == 0, n
        e_ocp = ((W - quantize(W, n, policy="ocp")[0]) ** 2).sum(1)
        e_opt = ((W - quantize(W, n, policy="opt")[0]) ** 2).sum(1)
        assert np.all(e_opt <= e_ocp + 1e-12), n


def test_bfp6_differs_from_mxint6_only_by_rule():
    W = rng.standard_normal((40, 100))
    # same grid when BFP6 uses MXINT6's ceil policy (documented equivalence)
    assert np.array_equal(quantize(W, "BFP6")[0], quantize(W, "MXINT6", policy="ceil")[0])
    # the whole-row variant is genuinely different
    assert not np.array_equal(quantize(W, "BFP6_row")[0], quantize(W, "MXINT6")[0])
    assert quantize(W, "BFP6_row")[1].shape == (40, 1)


def test_storage():
    assert storage_bits(100, 128)[1] == 6.25
    assert abs(storage_bits(100, 100)[1] - 6.32) < 1e-12
    assert abs(storage_bits(100, 100, block="row")[1] - 6.08) < 1e-12


def test_metrics_basic():
    W = rng.standard_normal((10, 100)); X = rng.standard_normal((100, 50))
    m = metrics(W, W, X)
    assert m["y_rel_err"] == 0 and m["w_rel_err"] == 0 and abs(m["y_cos_row_min"] - 1) < 1e-12
    m = metrics(W, np.zeros_like(W), X)
    assert abs(m["y_rel_err"] - 1) < 1e-12 and m["zeroed_frac"] == 1 and m["y_cos_row_mean"] == 0


def test_exhaustive_and_checkpoint(tmp=None):
    import tempfile
    W = rng.standard_normal((8, 64)); X = rng.standard_normal((64, 20))
    ck = os.path.join(tmp or tempfile.mkdtemp(), "ck.jsonl")
    rows = exhaustive(W, X, checkpoint=ck, progress=False)
    assert len(rows) == 5 * 4 and sum(1 for _ in open(ck)) == 20
    again = exhaustive(W, X, checkpoint=ck, progress=False)     # resumes from checkpoint
    assert [r["y_rel_err"] for r in again] == [r["y_rel_err"] for r in rows]
    assert len(ranking(rows)) == 5


# ---------------------------------------------------------------- regression on dataset C
def test_dataset_C_matches_prototype():
    path = os.path.join(HERE, "data", "W_C.npy")
    if not os.path.exists(path):
        print("  (skipped: data/W_C.npy not found)")
        return
    W = np.load(path).astype(np.float64)
    X = np.load(os.path.join(HERE, "data", "X_cal.npy")).astype(np.float64)
    proto = {  # % Y error from the team plan (ocp, ceil, block-optimal)
        "MXFP6_E2M3": (3.08, 3.10, 3.06), "MXINT6": (4.05, 4.17, 3.90),
        "MXFP6_E3M2": (5.44, 5.24, 5.24), "MXFP6_E4M1": (11.09, 10.18, 10.18),
    }
    for n, vals in proto.items():
        for pol, v in zip(("ocp", "ceil", "opt"), vals):
            got = 100 * metrics(W, quantize(W, n, policy=pol)[0], X)["y_rel_err"]
            assert abs(got - v) < 0.006, (n, pol, round(got, 3), v)


if __name__ == "__main__":
    tests = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {name}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
