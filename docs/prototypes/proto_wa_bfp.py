import numpy as np
from proto_baseline import *
names=['MXFP E3M2','MXFP E2M3','MXFP E4M1','MXINT6']
Xc=X
print("X stats: std %.3f kurt %.2f max/std %.1f"%(X.std(),((X-X.mean())**4).mean()/X.var()**2,abs(X).max()/X.std()))
def quantX(X,n,bs=32): return quant(X.T.copy(),n,'ocp',bs).T   # blocks along reduction axis (axis 0 of X)
for k in 'ACE':
    W=d['W_'+k].astype(np.float64); Y=W@X
    print(f'\nDataset {k}: rel Y error %  [W only | X only | W+X]')
    for n in names:
        Qw=quant(W,n); Qx=quantX(X,n)
        f=lambda Yq:100*np.linalg.norm(Yq-Y)/np.linalg.norm(Y)
        print(f'  {n:10s} {f(Qw@X):6.3f} | {f(W@Qx):6.3f} | {f(Qw@Qx):6.3f}')
# MXINT6 two's complement [-32,31] vs symmetric, class 'S=2^floor(log2 absmax)'
def int_tc(z,bits=6):
    q=2.0**-(bits-2); return np.clip(np.round(z/q),-2**(bits-1),2**(bits-1)-1)*q
FMT['MXINT6-2c']=(int_tc,0); fmax_old=fmax
print("\nMXINT6 symmetric vs two's complement (Y err %), per dataset:")
for k in 'ABCDE':
    W=d['W_'+k].astype(np.float64)
    print(' ',k, round(100*metrics(W,quant(W,'MXINT6'))[1],4), round(100*metrics(W,quant(W,'MXINT6-2c'))[1],4))
# BFP g|e|m design: block-max exponent range for C and overhead at equal 6.25
W=d['W_C'].astype(np.float64)
for g in (8,16,32,64):
    B,_=blocks(W,g); mx=np.abs(B).max(-1); E=np.floor(np.log2(mx[mx>0]))
    print(f'C: g={g:3d} shared-exp range [{int(E.min())},{int(E.max())}] ({int(E.max()-E.min()+1)} distinct values)  Y err MXINT-grid {100*metrics(W,quant(W,"MXINT6",bs=g))[1]:.3f}%  bits/elem with 4-bit exp {6+4/g:.3f}, 8-bit exp {6+8/g:.3f}')
for k in 'ABDE':
    Wk=d['W_'+k].astype(np.float64); B,_=blocks(Wk,16); E=np.floor(np.log2(np.abs(B).max(-1)[np.abs(B).max(-1)>0])); print(f' {k}: g=16 exp range [{int(E.min())},{int(E.max())}] -> {int(E.max()-E.min()+1)} values')
