import numpy as np
from proto_baseline import *
# E4M1 top-code conventions: (a) none reserved vmax=384/emax 8  (b) one NaN (E4M3-style) vmax=256/emax 8  (c) IEEE-like Inf/NaN top exponent vmax=192/emax 7
conv={'a: no NaN/Inf (vmax 384, emax 8)':(dict(m=1,emin=-6,emax=8,vmax=384.0),8),
      'b: one NaN   (vmax 256, emax 8)':(dict(m=1,emin=-6,emax=8,vmax=256.0),8),
      'c: IEEE-like (vmax 192, emax 7)':(dict(m=1,emin=-6,emax=7,vmax=192.0),7)}
def q4(W,f,emax,policy):
    B,c=blocks(W); mx=np.abs(B).max(-1,keepdims=True); se=np.floor(np.log2(np.where(mx>0,mx,1)))-emax
    if policy=='ceil': se=np.where(np.abs(B/2.0**se).max(-1,keepdims=True)>f['vmax'],se+1,se)
    se=np.where(mx==0,0,se); s=2.0**se
    return (fp_round(B/s,f)*s).reshape(100,-1)[:,:c]
for k in 'CE':
    W=d['W_'+k].astype(np.float64); print('Dataset',k,' E4M1 relative Y error %  (ocp | ceil)')
    for n,(f,e) in conv.items():
        print('  ',n,' | '.join(f'{100*metrics(W,q4(W,f,e,p))[1]:.2f}' for p in ('ocp','ceil')))
# BFP variants on C: different exponent rule / rounding, same 5-bit magnitude mantissa, g=32
W=d['W_C'].astype(np.float64)
def bfp(W,g,rule='floor',rnd='rne',e_bits=None):
    B,c=blocks(W,g); mx=np.abs(B).max(-1,keepdims=True)
    E=np.floor(np.log2(np.where(mx>0,mx,1)))
    if rule=='ceil': E=E+1            # mantissa in [0,1): no clipping, one bit wasted
    if e_bits: E=np.clip(E,-2**(e_bits-1),2**(e_bits-1)-1)
    s=2.0**(E-4) if rule=='floor' else 2.0**(E-4)
    z=B/s; z=np.round(z) if rnd=='rne' else np.trunc(z)
    return (np.clip(z,-31,31)*s).reshape(100,-1)[:,:c]
print('\nC, BFP6 variants, rel Y err % (grid of MXINT6 = floor+rne, g=32)')
for g in (16,32):
    for rule in ('floor','ceil'):
        for rnd in ('rne','trunc'):
            print(f'  g={g} exp={rule:5s} round={rnd:5s}  {100*metrics(W,bfp(W,g,rule,rnd))[1]:.3f}')
print('  g=16, 4-bit exp clamp [-8,7], floor/rne:',round(100*metrics(W,bfp(W,16,'floor','rne',4))[1],3))
