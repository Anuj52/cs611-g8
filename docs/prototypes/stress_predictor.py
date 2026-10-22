import numpy as np, sys
sys.path.insert(0,'.')  # run from a clone of the team repo (needs formats.py, predictor.py)
from formats import FORMATS, quantize
from predictor import build_stats, predict
names=list(FORMATS)
def exh(W,X):
    Y=W@X; return {n:float(((Y-quantize(W,n)[0]@X)**2).sum()) for n in names}
def rep(W,X):
    e=exh(W,X); p=predict(build_stats(W,X@X.T)); we,wp=min(e,key=e.get),min(p,key=p.get)
    srt=sorted(e.values()); margin=10*np.log10(srt[1]/srt[0])
    return we,wp,10*np.log10(e[wp]/e[we]),margin
d=np.load('assignment_data.npz'); X=d['X_cal'].astype(np.float64)   # the course zip's data file
print("REAL datasets A-E, 6-bit, real X_cal  (margin = dB between best and 2nd best)")
for k in 'ABCDE':
    we,wp,reg,mar=rep(d['W_'+k].astype(np.float64),X); print(f"  {k}: exhaustive={we:11s} predictor={wp:11s} regret={reg:.3f} dB  margin={mar:.2f} dB")
rng=np.random.default_rng(7)
print("\nSTRESS: sparse W around the INT/FP crossover x X row-gain spread (30 trials each)")
print("  sparsity gain-spread  top1  mean-regret  max-regret  mean-margin  winners")
for p in (0.5,0.6,0.7,0.8,0.9):
    for g in (0.5,2.0):
        ag=0; reg=[]; mar=[]; wins={}
        for _ in range(30):
            W=rng.standard_normal((100,100))*(rng.random((100,100))>p)
            Xs=rng.standard_normal((100,256))*0.1*10**rng.uniform(-g,g,(100,1))
            we,wp,r,m=rep(W,Xs); ag+=we==wp; reg.append(r); mar.append(m); wins[we[:7]+we[-2:]]=wins.get(we[:7]+we[-2:],0)+1
        print(f"  {p:6.1f} {g:11.1f} {ag:5d}/30 {np.mean(reg):11.3f} {np.max(reg):11.3f} {np.mean(mar):12.2f}  {wins}")
print("\nSTRESS: low-rank correlated X (rank r + 5% noise) with weights aligned to X's dominant direction")
for r_ in (1,3,10):
    ag=0;reg=[]
    for _ in range(30):
        U=rng.standard_normal((100,r_)); Xs=(U@rng.standard_normal((r_,256))); Xs=Xs/Xs.std()*0.1+0.005*rng.standard_normal((100,256))
        W=rng.standard_normal((100,100))*0.7+0.7*(rng.standard_normal((100,r_))@U.T)   # rows partly aligned with X's subspace
        we,wp,rg,m=rep(W,Xs); ag+=we==wp; reg.append(rg)
    print(f"  rank {r_:2d}: top1 {ag}/30, mean regret {np.mean(reg):.3f} dB, max {np.max(reg):.3f}")
