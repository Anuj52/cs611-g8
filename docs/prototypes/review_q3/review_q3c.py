import numpy as np, sys
sys.path.insert(0,".")  # run from a clone of the team repo
from formats import FORMATS, quantize, metrics, split_X
d=np.load('assignment_data.npz'); X=d['X_cal'].astype(np.float64); Xf,Xh=split_X(X)
W0=d['W_C'].astype(np.float64); sigma=W0.std(); names=["MXFP6_E3M2","MXFP6_E2M3","MXINT6"]
def e(W,pol,XX=X): return {n:100*metrics(W,quantize(W,n,policy=pol)[0],XX)['y_rel_err'] for n in names}
def avg(gen,pol,XX=X,seeds=5):
    a={n:0 for n in names}
    for s in range(seeds):
        r=e(gen(np.random.default_rng(s)),pol,XX)
        for n in names:a[n]+=r[n]/seeds
    return a
def line(tag,a):
    b=min(a,key=a.get); s=sorted(a.values()); return f"{tag:16s} win {b[-5:]:>5s}  E2M3 {a['MXFP6_E2M3']:5.2f} E3M2 {a['MXFP6_E3M2']:5.2f} INT6 {a['MXINT6']:5.2f}  margin {10*np.log10(s[1]/s[0]):.2f} dB"
print("SPARSITY: ocp (all X) | opt (all X) | ocp (held-out odd cols)")
for p in (65,70,75,80,85,90):
    def g(rng,p=p):
        W=W0.copy(); W.flat[rng.choice(W.size,int(round(W.size*p/100)),replace=False)]=0; return W
    print(f"{p}%\n   ocp  {line('',avg(g,'ocp'))}\n   opt  {line('',avg(g,'opt'))}\n   held {line('',avg(g,'ocp',Xh))}")
print("\nOUTLIERS under opt")
for cnt,k in ((20,20),(20,50),(100,10),(100,20)):
    def g(rng,cnt=cnt,k=k):
        W=W0.copy(); idx=rng.choice(W.size,cnt,replace=False); W.flat[idx]=rng.choice([-1,1],cnt)*k*sigma; return W
    print(f" n={cnt} k={k}\n   ocp {line('',avg(g,'ocp'))}\n   opt {line('',avg(g,'opt'))}")
# scale variation seen through the metric: global Frobenius vs error restricted to the LOW-gain column groups
print("\nCOLUMN-GROUP SCALE SPREAD 10^4: error on all columns vs on the weakest 3 column groups (30 columns) only")
res={n:[0,0] for n in names}
for s in range(5):
    rng=np.random.default_rng(s); gains=rng.permutation(np.geomspace(1e-2,1e2,10)); gv=np.repeat(gains,10)
    W=W0*gv[None,:]; weak=np.argsort(gv)[:30]
    for n in names:
        Wq=quantize(W,n)[0]; dW=W-Wq
        res[n][0]+=100*np.linalg.norm(dW@X)/np.linalg.norm(W@X)/5
        res[n][1]+=100*np.linalg.norm(dW[:,weak]@X[weak])/np.linalg.norm(W[:,weak]@X[weak])/5
for n in names: print(f"  {n[-5:]:>5s}: global Y err {res[n][0]:6.2f}%   weakest-30-columns-only err {res[n][1]:7.2f}%")
