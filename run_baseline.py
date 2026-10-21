"""Exhaustive baseline: the ground truth the selector is graded against.  Owner: Akash.

    python run_baseline.py                          # dataset C from data/
    python run_baseline.py --W path/W_A.npy --tag A # any other matrix
    python run_baseline.py --variants               # also E4M1 vmax=192 and BFP6 one-block-per-row

Writes results/baseline_<tag>.csv and .json, plus a resumable checkpoint
results/baseline_<tag>.ckpt.jsonl (delete it to force a fresh run).
Policies: ocp (class rule), ceil, opt (exponent search, weight MSE),
opt_out (exponent search, output-aware, H fitted on even X columns only).
Errors are reported on all of X_cal and on the held-out odd columns.
"""
import argparse
import os

import numpy as np

from formats import FORMATS, VARIANTS, exhaustive, ranking, save_results, split_X

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--W", default=os.path.join(HERE, "data", "W_C.npy"))
ap.add_argument("--X", default=os.path.join(HERE, "data", "X_cal.npy"))
ap.add_argument("--tag", default="C")
ap.add_argument("--out", default=os.path.join(HERE, "results"))
ap.add_argument("--variants", action="store_true")
ap.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint")
args = ap.parse_args()

W = np.load(args.W).astype(np.float64)
X = np.load(args.X).astype(np.float64)
names = list(FORMATS) + (list(VARIANTS) if args.variants else [])
base = os.path.join(args.out, f"baseline_{args.tag}")
ckpt = base + ".ckpt.jsonl"
if args.fresh and os.path.exists(ckpt):
    os.remove(ckpt)

print(f"W {W.shape}  X {X.shape}  formats {len(names)}  dataset {args.tag}")
rows = exhaustive(W, X, names=names, checkpoint=ckpt)
save_results(rows, base)

# ---- readable summary --------------------------------------------------------------------
pols = ["ocp", "ceil", "opt", "opt_out"]
by = {(r["format"], r["policy"]): r for r in rows}
print(f"\nY relative error (%) on all of X_cal  [held-out odd columns in brackets]")
print(f"{'format':17s}" + "".join(f"{p:>17s}" for p in pols) + f"{'bits/el':>9s}")
for n in names:
    cells = "".join(f"{100*by[n,p]['y_rel_err']:9.3f} [{100*by[n,p]['y_rel_err_heldout']:5.2f}]" for p in pols)
    print(f"{n:17s}{cells}{by[n,'ocp']['bits_per_elem']:9.2f}")

print("\nSaturated / zeroed / subnormal element counts under the class 'ocp' rule:")
for n in names:
    r = by[n, "ocp"]
    print(f"  {n:17s} saturated {r['n_saturated']:5d}   zeroed {r['n_zeroed']:5d}   subnormal {r['n_subnormal']:5d}")

print("\nRanking, each format at its best policy (Y error on all of X_cal):")
for i, (f, p, v) in enumerate(ranking(rows, policies=("ocp", "ceil", "opt")), 1):
    print(f"  {i}. {f:17s} {100*v:6.3f}%  ({p})")
print("Ranking under the class 'ocp' rule only:")
for i, (f, p, v) in enumerate(ranking(rows, policies=("ocp",)), 1):
    print(f"  {i}. {f:17s} {100*v:6.3f}%")

same = [p for p in pols if by["BFP6", p]["y_rel_err"] == by["MXINT6", p]["y_rel_err"]]
same_cross = by["BFP6", "ceil"]["y_rel_err"] == by["MXINT6", "ceil"]["y_rel_err"]
print(f"\nBFP6 vs MXINT6: identical under policies {same}; BFP6 default (ceil) == MXINT6 ceil: {same_cross}")
nr = sum(r["n_roundings"] for r in rows)
nm = sum(r["n_macs"] for r in rows)
print(f"Cost of this exhaustive run: {nr:,} element roundings + {nm:,} MACs for (W - Wq) X")
print(f"\nSaved {base}.csv / .json")
