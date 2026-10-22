import numpy as np, sys
sys.path.insert(0,".")  # run from a clone of the team repo
from formats import FORMATS, quantize, metrics, storage_bits
d=np.load('assignment_data.npz'); X=d['X_cal'].astype(np.float64)
names=list(FORMATS)
def yerr(W,n,block=None,policy=None):
    Wq,_=quantize(W,n,block=block,policy=policy); return 100*metrics(W,Wq,X)['y_rel_err']
print("== baseline (default policies) per dataset: Y rel err %")
for k in 'ABCDE':
    W=d['W_'+k].astype(np.float64); r={n:yerr(W,n) for n in names}; b=min(r,key=r.get)
    print(k,b,round(r[b],3),{n:round(v,3) for n,v in r.items()})
W=d['W_C'].astype(np.float64)
print("\n== block-size study on C, class (ocp) rule for every format, bits/elem incl. tail scale")
print("block  bits   "+"  ".join(f"{n[-5:]:>8s}" for n in names))
for b in (4,8,15,16,25,32,50,64,100,128):
    bits=storage_bits(100,100,block=b)[1]
    print(f"{b:5d} {bits:6.3f} "+"  ".join(f"{yerr(W,n,block=b,policy='ocp'):8.3f}" for n in names))
print("\n== equal-budget: all formats at the SAME number of blocks per row")
for nb,bs in ((4,25),(4,32),(2,50),(2,64)):
    print(f" {nb} blocks/row, block {bs}, {storage_bits(100,100,block=bs)[1]:.2f} bits:", {n[-5:]:round(yerr(W,n,block=bs,policy='ocp'),3) for n in names})
