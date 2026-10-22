import numpy as np
from proto_baseline import *
print("X_cal: cond(XX^T) =", np.linalg.cond(X@X.T), " mean |corr offdiag| =", np.abs(np.corrcoef(X)-np.eye(100)).mean())
ev=np.linalg.eigvalsh(X@X.T); print(" eig range %.3g..%.3g"%(ev.min(),ev.max()))
for k in 'ABCDE':
    W=d['W_'+k].astype(np.float64); print(f'\nDataset {k}')
    Bw,_=blocks(W)
    for n in FMT:
        a0,b0=metrics(W,quant(W,n,'ocp')); a1,b1=metrics(W,quant(W,n,'ceil'))
        cands=[quant(W,n,'ocp',BS,dl) for dl in (-2,-1,0,1,2)]
        errs=np.stack([((blocks(Q)[0]-Bw)**2).sum(-1) for Q in cands]); idx=errs.argmin(0)
        Bq=np.stack([blocks(Q)[0] for Q in cands]); Bsel=np.take_along_axis(Bq,idx[None,...,None],0)[0]
        Qo=Bsel.reshape(100,-1)[:,:100]; a2,b2=metrics(W,Qo)
        print(f'  {n:12s} ocp Y={100*b0:6.3f}  ceil Y={100*b1:6.3f}  block-opt Y={100*b2:6.3f}   delta-hist(-2..+2) {np.bincount(idx.ravel(),minlength=5)}')
W=d['W_C'].astype(np.float64)
print("\nC: Y error on X halves (ocp)")
for n in FMT:
    Q=quant(W,n); e=[np.linalg.norm((Q-W)@X[:,s])/np.linalg.norm(W@X[:,s]) for s in (slice(0,128),slice(128,256))]; print(' ',n,[f'{100*v:.3f}' for v in e])
print("\nC: block size sensitivity (ocp, Y err %)")
for bs in (8,16,32,64,100):
    print(' bs',bs,{n:round(100*metrics(W,quant(W,n,'ocp',bs))[1],3) for n in FMT})
