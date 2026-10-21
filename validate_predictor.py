"""Validate the histogram predictor against exhaustive (brute-force) evaluation."""
import glob, time, numpy as np
from formats import FORMATS, quantize
from predictor import build_stats, predict
from scipy_free_stats import spearman

from data import load
W, X = load()
names = list(FORMATS)

def exhaustive(W, X):
    Y = W @ X
    return {n: float(((Y - quantize(W, n)[0] @ X) ** 2).sum()) for n in names}

def db(sig, err): return 10 * np.log10(sig / err)

# --- W_C ---------------------------------------------------------------------
H = X @ X.T
sig = ((W @ X) ** 2).sum()
t0 = time.perf_counter(); ex = exhaustive(W, X); t_ex = time.perf_counter() - t0
t0 = time.perf_counter(); pr = predict(build_stats(W, H)); t_pr = time.perf_counter() - t0
print("W_C: output SQNR (dB)   exhaustive   predicted   diff")
for n in sorted(names, key=lambda n: ex[n]):
    print(f"  {n:12s} {db(sig,ex[n]):10.2f} {db(sig,pr[n]):11.2f} {db(sig,pr[n])-db(sig,ex[n]):+7.2f}")
print(f"  winner exhaustive={min(ex,key=ex.get)} predicted={min(pr,key=pr.get)}  "
      f"spearman={spearman([ex[n] for n in names],[pr[n] for n in names]):.2f}")

# timing: repeat to get stable numbers
R = 20
t0 = time.perf_counter(); [exhaustive(W, X) for _ in range(R)]; t_ex = (time.perf_counter()-t0)/R
t0 = time.perf_counter(); [build_stats(W, H) for _ in range(R)]; t_st = (time.perf_counter()-t0)/R
hs = build_stats(W, H)
t0 = time.perf_counter(); [predict(hs) for _ in range(R)]; t_sc = (time.perf_counter()-t0)/R
print(f"  time: exhaustive {t_ex*1e3:.1f} ms | predictor stats pass {t_st*1e3:.1f} ms + scoring 5 formats {t_sc*1e3:.2f} ms")

# operation count (format-independent, more honest than Python timing)
N = W.size
print(f"  ops: exhaustive ~ 5 formats x ({N} roundings + {W.shape[0]*W.shape[1]*X.shape[1]} MACs for Q(W)X)")
print(f"       predictor  ~ 1 x {N} binnings + 5 formats x ({hs[0].size} bin evaluations + {len(hs[1])} block-max roundings)")

# --- synthetic stress test -----------------------------------------------------
rng = np.random.default_rng(1)
def gen(kind):
    if kind == "gauss":   return rng.standard_normal((100, 100))
    if kind == "laplace": return rng.laplace(size=(100, 100))
    if kind == "t3":      return rng.standard_t(3, (100, 100))
    if kind == "t6":      return rng.standard_t(6, (100, 100))
    if kind == "uniform": return rng.uniform(-1, 1, (100, 100))
    if kind == "sparse90":return rng.standard_normal((100, 100)) * (rng.random((100, 100)) > .9)
    if kind == "outliers":
        A = rng.standard_normal((100, 100)); A.flat[rng.choice(A.size, 20, replace=False)] *= 50; return A
    if kind == "rowscale":return rng.standard_normal((100, 100)) * 10 ** rng.uniform(-2, 2, (100, 1))
    if kind == "W_C_perm":return rng.permutation(W.ravel()).reshape(100, 100)
kinds = ["gauss", "laplace", "t3", "t6", "uniform", "sparse90", "outliers", "rowscale", "W_C_perm"]
print("\nSynthetic matrices, 10 trials each, random X ~ N(0, 0.1^2) with random column gains")
print(f"  {'kind':10s} {'top-1 agree':>11s} {'regret dB':>10s} {'spearman':>9s}  exhaustive winners")
tot = agree_tot = 0
for k in kinds:
    agree, regret, rho, wins = 0, [], [], {}
    for _ in range(10):
        A = gen(k)
        Xs = rng.standard_normal((100, 256)) * 0.1 * 10 ** rng.uniform(-0.5, 0.5, (100, 1))
        e = exhaustive(A, Xs); p = predict(build_stats(A, Xs @ Xs.T))
        we, wp = min(e, key=e.get), min(p, key=p.get)
        agree += we == wp; regret.append(10 * np.log10(e[wp] / e[we]))
        rho.append(spearman([e[n] for n in names], [p[n] for n in names]))
        wins[we] = wins.get(we, 0) + 1
    tot += 10; agree_tot += agree
    print(f"  {k:10s} {agree:8d}/10 {np.mean(regret):10.3f} {np.mean(rho):9.2f}  {wins}")
print(f"  overall top-1 agreement {agree_tot}/{tot}")
