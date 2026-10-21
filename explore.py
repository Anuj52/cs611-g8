"""Exploratory run: exhaustive evaluation, scale search, cheap predictors, sensitivity."""
import glob, numpy as np
from formats import *

from data import load
W, X = load()
H = X @ X.T
names = list(FORMATS)

def table(W, X, H=None, shift=0, label=""):
    rows = {}
    for n in names:
        Wq, _ = quantize(W, n, shift_search=shift, H=H)
        rows[n] = metrics(W, Wq, X)
    print(f"\n== {label}")
    print(f"{'format':12s} {'wSQNR dB':>9s} {'ySQNR dB':>9s} {'y rel err':>10s}")
    for n, m in sorted(rows.items(), key=lambda kv: -kv[1]['y_sqnr_db']):
        print(f"{n:12s} {m['w_sqnr_db']:9.2f} {m['y_sqnr_db']:9.2f} {m['y_rel_err']:10.4%}")
    return rows

base = table(W, X, label="Exhaustive, default OCP scale (floor rule)")
ws   = table(W, X, shift=2, label="Per-block exponent search +-2, weight-MSE objective")
os_  = table(W, X, H=H, shift=2, label="Per-block exponent search +-2, output-aware objective (H = X X^T)")

# generalisation: tune on half of X_cal columns, evaluate on the other half
Xa, Xb = X[:, ::2], X[:, 1::2]
print("\n== Split X_cal: output-aware search on even cols, evaluate on odd cols")
for n in names:
    Wq, _ = quantize(W, n, shift_search=2, H=Xa @ Xa.T)
    Wq0, _ = quantize(W, n)
    print(f"{n:12s} default {metrics(W,Wq0,Xb)['y_sqnr_db']:7.2f}  tuned {metrics(W,Wq,Xb)['y_sqnr_db']:7.2f}")

# ---- cheap predictor 1: analytic noise model, no rounding performed --------
def predict(W, name, H):
    kind, p, emax = FORMATS[name]
    pred_out = 0.0; pred_w = 0.0
    for sl in blocks(W.shape[1]):
        Wb = W[:, sl]
        e = default_exp(kind, p, emax, np.abs(Wb).max(1))[:, None]
        s = 2.0 ** e
        a = np.abs(Wb) / s
        if kind == "fp":
            ee = np.maximum(np.floor(np.log2(np.where(a > 0, a, 1))), 1 - p["bias"])
            step = 2.0 ** (ee - p["mbits"]); vmax = p["maxval"]
        elif kind == "int":
            step = np.full_like(a, 2.0 ** -p["frac_bits"]); vmax = p["kmax"] * 2.0 ** -p["frac_bits"]
        else:
            step = np.ones_like(a); vmax = p["kmax"]
        var = np.where(a > vmax, (a - vmax) ** 2, step ** 2 / 12) * s ** 2   # clip error or uniform noise
        pred_w += var.sum()
        pred_out += (var.sum(0) * np.diag(H)[sl]).sum()     # assumes uncorrelated errors
    return pred_w / W.size, pred_out

# ---- cheap predictor 2: quantize a random 20% of blocks ----------------------
rng = np.random.default_rng(0)
def sampled(W, name, frac=0.2):
    rows = rng.choice(W.shape[0], int(frac * W.shape[0]), replace=False)
    Wq, _ = quantize(W[rows], name)
    return metrics(W[rows], Wq, X)['y_sqnr_db']

print("\n== Cheap predictors vs exhaustive (default scale)")
Ynorm = ((W @ X) ** 2).sum()
for n in names:
    pw, po = predict(W, n, H)
    print(f"{n:12s} exhaustive ySQNR {base[n]['y_sqnr_db']:7.2f} | analytic {10*np.log10(Ynorm/po):7.2f} | 20% rows sampled {sampled(W, n):7.2f}")

# ---- sensitivity -----------------------------------------------------------
def winner(Wv):
    r = {n: metrics(Wv, quantize(Wv, n)[0], X)['y_sqnr_db'] for n in names}
    best = max(r, key=r.get)
    return best, " ".join(f"{k.split('_')[-1]}={v:.1f}" for k, v in r.items())

print("\n== Sensitivity (default scale, ySQNR dB per format)")
print("original           ", *winner(W))
Wc = np.clip(W, -3, 3); print("outliers clipped 3s", *winner(Wc))
for mult in [10, 50]:
    Wo = W.copy(); idx = rng.choice(W.size, 20, replace=False); Wo.flat[idx] *= mult
    print(f"20 outliers x{mult:<3d}    ", *winner(Wo))
for sp in [0.5, 0.9]:
    Ws = W * (rng.random(W.shape) > sp); print(f"sparsity {sp:.0%}       ", *winner(Ws))
Wr = W * (10.0 ** rng.uniform(-2, 2, (100, 1))); print("row scales 1e-2..1e2", *winner(Wr))
Wk = W * (10.0 ** rng.uniform(-2, 2, (1, 100))); print("col scales 1e-2..1e2", *winner(Wk))
g = rng.standard_normal(W.shape); print("pure Gaussian      ", *winner(g))
u = rng.uniform(-1, 1, W.shape); print("uniform            ", *winner(u))
t = rng.standard_t(3, W.shape); print("Student-t nu=3     ", *winner(t))

print("\nstorage bits/elem (6-bit + 8-bit scale per block):", storage_bits(100, 100)[1])
