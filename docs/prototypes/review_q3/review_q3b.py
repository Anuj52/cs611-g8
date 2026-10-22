import numpy as np, sys
sys.path.insert(0,".")  # run from a clone of the team repo
from formats import FORMATS, quantize, metrics
d=np.load('assignment_data.npz'); X=d['X_cal'].astype(np.float64)
W0=d['W_C'].astype(np.float64); sigma=W0.std(); names=list(FORMATS)
def errs(W): return {n:100*metrics(W,quantize(W,n)[0],X)['y_rel_err'] for n in names}
def avg(gen,seeds=5):
    acc={n:0 for n in names}
    for s in range(seeds):
        e=errs(gen(np.random.default_rng(s)))
        for n in names: acc[n]+=e[n]/seeds
    return acc
def show(tag,a):
    b=min(a,key=a.get); s=sorted(a.values()); print(f"{tag:28s} win {b[-5:]:>5s} {a[b]:6.2f}  2nd margin {10*np.log10(s[1]/s[0]):.2f} dB   E2M3 {a['MXFP6_E2M3']:6.2f} E3M2 {a['MXFP6_E3M2']:6.2f} INT6 {a['MXINT6']:6.2f}")
print("== OUTLIERS: count x magnitude(sigma)")
for cnt in (1,5,20,100):
    for k in (5,10,20,50):
        def g(rng,cnt=cnt,k=k):
            W=W0.copy(); idx=rng.choice(W.size,cnt,replace=False); W.flat[idx]=rng.choice([-1,1],cnt)*k*sigma; return W
        show(f"n={cnt:3d} k={k:2d}",avg(g))
print("\n== SPARSITY (random unstructured zeros)")
for p in (0,25,50,55,60,65,70,75,90,95):
    def g(rng,p=p):
        W=W0.copy(); idx=rng.choice(W.size,int(round(W.size*p/100)),replace=False); W.flat[idx]=0; return W
    show(f"{p}% zeros",avg(g))
print("\n== SCALE SPREAD: ten contiguous row groups / column groups, gains geomspace(10^-a,10^a,10) permuted")
for axis in ('row','col'):
    for a in (0,0.5,1,1.5,2):
        def g(rng,a=a,axis=axis):
            gains=rng.permutation(np.geomspace(10**-a,10**a,10)); W=W0.copy()
            if axis=='row': W=W*np.repeat(gains,10)[:,None]
            else: W=W*np.repeat(gains,10)[None,:]
            return W
        show(f"{axis} spread 10^{2*a:g}",avg(g))
