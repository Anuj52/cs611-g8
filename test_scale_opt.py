"""Tests for scale_opt.py.  Run:  python test_scale_opt.py   (or: pytest test_scale_opt.py)

The hand examples are the ones worked on paper in notes/anuj.md.
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import scale_opt as S  # noqa: E402

rng = np.random.default_rng(1)
ALL = list(S.FMTS) + list(S.VARIANTS)


# ---------------------------------------------------------------- grids and rounding
def test_grid_limits():
    expect = {"MXFP6_E3M2": (32, 28.0, 2 ** -4), "MXFP6_E2M3": (32, 7.5, 2 ** -3),
              "MXFP6_E4M1": (32, 384.0, 2 ** -7), "MXINT6": (32, 31 / 16, 2 ** -4),
              "BFP6": (32, 31 / 16, 2 ** -4), "MXFP6_E4M1_v192": (30, 192.0, 2 ** -7),
              "MXFP6_E4M1_nan256": (31, 256.0, 2 ** -7)}
    for n, (size, vmax, smallest) in expect.items():
        g = S.spec(n).grid
        assert len(g) == size and g[-1] == vmax and g[1] == smallest, n
        assert np.all(np.diff(g) > 0), n


def test_vmax_ratio_below_two():
    # vmax / 2^emax < 2 for every format: the floor rule can clip the block max
    r = {n: S.spec(n).grid[-1] / 2.0 ** S.spec(n).emax for n in S.FMTS}
    assert r == {"MXFP6_E3M2": 1.75, "MXFP6_E2M3": 1.875, "MXFP6_E4M1": 1.5,
                 "MXINT6": 1.9375, "BFP6": 1.9375}


def test_rne_ties_and_saturation():
    g = S.spec("MXFP6_E2M3").grid
    x = np.array([1.0625, 1.1875, 0.0625, 0.1875, 7.75, 100.0, -1.0625, -0.0])
    q = S.round_to_grid(x, g)
    assert list(q[:6]) == [1.0, 1.25, 0.0, 0.25, 7.5, 7.5] and q[6] == -1.0
    assert q[7] == 0 and np.signbit(q[7])                     # -0 stays -0
    k = S.round_to_grid((np.arange(-63, 64) + 0.5) / 16, S.spec("MXINT6").grid) * 16
    assert np.all((k % 2 == 0) | (np.abs(k) == 31))           # ties to even codes or saturated
    assert k.min() == -31 and k.max() == 31


def test_floor_log2_exact_near_powers_of_two():
    a = np.array([4.0, np.nextafter(4.0, 0), 1.0, np.nextafter(1.0, 0), 2.0 ** -130, 3.0])
    assert list(S.floor_log2(a)) == [2, 1, 0, -1, -130, 1]


# ---------------------------------------------------------------- hand examples (notes/anuj.md)
def test_hand_example_mxint6_shrinking_the_scale():
    x = np.array([[4.05, 0.3, -0.6, 0.2, 0.9, -0.1, 0.45, 1.1]])
    f = S.spec("MXINT6")
    sse = {e: ((x - S.qblock(f, x, np.array([e]))) ** 2).sum() for e in (1, 2, 3)}
    assert np.allclose([sse[1], sse[2], sse[3]], [0.040625, 0.05, 0.125])
    assert S.quantize(x, "MXINT6")[1][0, 0] == 2                    # class rule
    assert S.quantize(x, "MXINT6", "opt")[1][0, 0] == 1             # opt clips the max
    assert S.quantize(x, "MXINT6", "model")[1][0, 0] == 1           # model agrees


def test_hand_example_e2m3_floor_rule_clips():
    x = np.array([[3.9, 0.3, -0.2, 0.1, 1.3, -0.7, 0.45, 2.2]])
    f = S.spec("MXFP6_E2M3")
    q_ocp, q_up = S.qblock(f, x, np.array([-1])), S.qblock(f, x, np.array([0]))
    assert q_ocp[0, 0] == 3.75 and q_up[0, 0] == 4.0
    assert np.isclose(((x - q_ocp) ** 2).sum(), 0.02875)
    assert np.isclose(((x - q_up) ** 2).sum(), 0.025625)
    assert q_ocp[0, 4] == q_up[0, 4] == 1.25 and q_ocp[0, 7] == q_up[0, 7] == 2.25   # normals keep their step
    assert S.quantize(x, "MXFP6_E2M3")[1][0, 0] == -1
    assert S.quantize(x, "MXFP6_E2M3", "ceil")[1][0, 0] == 0


# ---------------------------------------------------------------- policies
def test_ceil_is_ocp_or_one_more_and_never_clips():
    W = rng.standard_t(3, (40, 100))
    for n in ALL:
        f = S.spec(n)
        e_ocp = S.quantize(W, n, "ocp")[1]
        Wq, e_ceil = S.quantize(W, n, "ceil")
        assert np.all((e_ceil - e_ocp == 0) | (e_ceil - e_ocp == 1)), n
        assert S.error_parts(W, Wq, e_ceil, n)["n_sat"] == 0, n
        # exactly the blocks whose max clips at e_ocp move up
        amax = np.stack([np.abs(W[:, sl]).max(1) for sl in S.block_slices(100, f.block)], 1)
        assert np.array_equal(e_ceil - e_ocp, (amax > np.ldexp(f.grid[-1], e_ocp)).astype(int)), n


def test_opt_never_worse_than_ocp_or_ceil_per_block():
    W = rng.laplace(size=(30, 100))
    for n in ALL:
        f = S.spec(n)
        errs = {}
        for pol in ("ocp", "ceil", "opt"):
            Wq = S.quantize(W, n, pol)[0]
            errs[pol] = np.stack([((W - Wq)[:, sl] ** 2).sum(1) for sl in S.block_slices(100, f.block)])
        assert np.all(errs["opt"] <= errs["ocp"] + 1e-15), n
        assert np.all(errs["opt"] <= errs["ceil"] + 1e-15), n


def test_wider_window_never_worse():
    W = rng.standard_normal((30, 100)) * 10.0 ** rng.uniform(-1, 1, (30, 1))
    for n in S.FMTS:
        e4 = ((W - S.quantize(W, n, "opt")[0]) ** 2).sum()
        e6 = ((W - S.quantize(W, n, "opt", S.WIDE_WINDOW)[0]) ** 2).sum()
        e17 = ((W - S.quantize(W, n, "opt", S.ORACLE_WINDOW)[0]) ** 2).sum()
        assert e17 <= e6 + 1e-12 and e6 <= e4 + 1e-12, n


def test_zero_block_tail_block_and_clamps():
    for n in ALL:
        for pol in ("ocp", "ceil", "opt", "model"):
            Wq, e = S.quantize(np.zeros((2, 100)), n, pol)
            assert np.all(Wq == 0) and np.all(e == 0), (n, pol)
    W = rng.standard_normal((5, 100))
    W[:, 96:] *= 0.01
    e = S.quantize(W, "MXFP6_E2M3")[1]
    assert e.shape == (5, 4) and np.array_equal(e[:, 3], np.floor(np.log2(np.abs(W[:, 96:]).max(1))) - 2)
    assert S.quantize(W, "BFP6")[1].shape == (5, 7)                   # 6 x 16 + tail of 4
    assert np.isclose(S.bits_per_elem("BFP6"), 6.28) and np.isclose(S.bits_per_elem("MXINT6"), 6.32)
    # BFP6's 4-bit exponent: clamp to [-8, 7]
    big = np.full((1, 16), 1000.0)
    Wq, e = S.quantize(big, "BFP6")
    assert e[0, 0] == 7 and Wq[0, 0] == 31 * 2.0 ** 3              # saturates at 248
    assert S.quantize(np.full((1, 16), 2.0 ** -20), "BFP6")[1][0, 0] == -8
    huge = np.array([[2.0 ** 200] * 32, [2.0 ** -200] * 32])
    for n in S.FMTS:
        if S.spec(n).e_bits == 8:
            e = S.quantize(huge, n, "opt")[1]
            assert e.min() >= -127 and e.max() <= 127, n


def test_sign_symmetry():
    W = rng.laplace(size=(20, 100))
    for n in ALL:
        for pol in ("ocp", "ceil", "opt", "model"):
            assert np.array_equal(S.quantize(-W, n, pol)[0], -S.quantize(W, n, pol)[0]), (n, pol)


def test_joint_search_never_worse_and_matches_enumeration():
    # correlated X so the cross-block terms of H matter
    Z = rng.standard_normal((100, 3))
    X = Z @ rng.standard_normal((3, 64)) + 0.3 * rng.standard_normal((100, 64))
    H = X @ X.T
    W = rng.standard_normal((12, 100))
    for n in ("MXFP6_E2M3", "MXINT6"):
        Wq_bd = S.quantize(W, n, "opt", S.OPT_WINDOW, H=H)[0]
        Wq_cd, _, obj_cd = S.joint_search(W, n, H, S.OPT_WINDOW)
        _, _, obj_ex = S.joint_search(W, n, H, S.OPT_WINDOW, exact=True)
        obj_bd = np.einsum("ij,jk,ik->i", W - Wq_bd, H, W - Wq_bd)
        assert np.all(obj_cd <= obj_bd * (1 + 1e-12)), n            # never worse than its start
        assert np.all(obj_ex <= obj_cd * (1 + 1e-12)), n            # enumeration is the optimum
        r = W - Wq_cd
        assert np.allclose(np.einsum("ij,jk,ik->i", r, H, r), obj_cd)


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
