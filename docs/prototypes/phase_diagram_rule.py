import numpy as np, sys
sys.path.insert(0,".")  # run from a clone of the team repo
from formats import quantize, metrics
names=["MXFP6_E3M2","MXFP6_E2M3","MXFP6_E4M1","MXINT6"]
rng=np.random.default_rng(0)
def kurt(x): x=x-x.mean(); return float((x**4).mean()/(x**2).mean()**2)
def gg(beta,shape):   # generalized Gaussian: |x|^beta density; beta=1 Laplace, 2 Gaussian
    g=rng.gamma(1/beta,1.0,shape)**(1/beta); return g*rng.choice([-1,1],shape)
def student(nu,shape): return rng.standard_t(nu,shape)
def winner(W):
    e={n:np.linalg.norm(W-quantize(W,n)[0])**2 for n in names}   # W-only error (X ~ white); 'ocp' class rule for all
    s=sorted(e.values()); return min(e,key=e.get), 10*np.log10(s[1]/s[0])
rows=[]
shape=(400,416)   # 13 full blocks per row, big enough for stable stats
fams=[('gg',b) for b in (0.5,0.7,1,1.5,2,3,6,20)]+[('t',v) for v in (2.5,3,4,6,10,30)]
print("family      param  sparsity  kurt(all)  kurt(nonzero)  winner (margin dB)")
for fam,par in fams:
    for p in (0.0,0.5,0.8,0.9):
        W=(gg(par,shape) if fam=='gg' else student(par,shape))*(rng.random(shape)>p)
        w,m=winner(W); nz=W[W!=0]
        rows.append((fam,par,p,kurt(W.ravel()),kurt(nz),w,m))
        if p in (0.0,0.8): print(f"{fam:5s} {par:7.1f} {p:8.1f} {rows[-1][3]:10.2f} {rows[-1][4]:13.2f}   {w:11s} ({m:.2f})")
# simple 2-feature rule: dense -> by kurtosis of nonzeros, sparse -> by sparsity.  Fit thresholds by brute force
import itertools
X=np.array([[r[2],r[4]] for r in rows]); y=np.array([r[5] for r in rows])
best=None
for sp in (0.5,0.6,0.7,0.8,0.9):
  for k1 in np.arange(1.5,4.0,0.1):
    for k2 in (6,10,20,50,1e9):
        pred=np.where(X[:,0]>=sp,"MXINT6",np.where(X[:,1]<k1,"MXINT6",np.where(X[:,1]<k2,"MXFP6_E2M3","MXFP6_E3M2")))
        acc=(pred==y).mean()
        if best is None or acc>best[0]: best=(acc,sp,k1,k2)
print("\nbest 2-feature threshold rule on synthetic grid: accuracy %.2f  (sparsity>=%.1f -> MXINT6; kurt_nz<%.1f -> MXINT6; kurt_nz<%.0f -> E2M3 else E3M2)"%best)
from collections import Counter; print("label counts:",Counter(y))
d=np.load('assignment_data.npz')
print("\nreal datasets: features and exhaustive winner (W-only, class rule)")
for k in 'ABCDE':
    W=d['W_'+k].astype(np.float64); nz=W[W!=0]; sp=(W==0).mean()
    acc_pred="MXINT6" if (sp>=best[1] or kurt(nz)<best[2]) else ("MXFP6_E2M3" if kurt(nz)<best[3] else "MXFP6_E3M2")
    w,m=winner(W); print(f"  {k}: sparsity {sp:.2f} kurt_nz {kurt(nz):7.1f} -> rule says {acc_pred:11s} exhaustive {w:11s} (margin {m:.2f} dB)")
