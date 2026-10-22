import numpy as np, os
# EXPLORATORY PROTOTYPE (not the submission code). Set CS611_DATA to the path of assignment_data.npz
d=np.load(os.environ.get('CS611_DATA','assignment_data.npz')); X=d['X_cal'].astype(np.float64)
BS=32
# element formats following class notebook parametrisation (bits,m,emin,emax,vmax); E4M1 derived by the same rule (bias=2^(e-1)-1, top exponent code is finite)
FP={'E3M2':dict(m=2,emin=-2,emax=4,vmax=28.0),'E2M3':dict(m=3,emin=0,emax=2,vmax=7.5),'E4M1':dict(m=1,emin=-6,emax=8,vmax=384.0)}
def fp_round(z,f):
    a=np.abs(z); e=np.floor(np.log2(np.where(a>0,a,1))); e=np.maximum(e,f['emin'])
    step=2.0**(e-f['m']); r=np.round(a/step)*step   # np.round = RNE
    return np.sign(z)*np.minimum(r,f['vmax'])
def int_round(z,bits=6):                 # OCP-style MXINT: k*2^-(bits-2), symmetric (-2^(bits-1) code excluded)
    q=2.0**-(bits-2); kmax=2**(bits-1)-1
    return np.clip(np.round(z/q),-kmax,kmax)*q
def bfp_round(z,bits=6):                 # sign-magnitude, (bits-1) magnitude bits, integer mantissa, no implicit bit
    kmax=2**(bits-1)-1; return np.clip(np.round(z),-kmax,kmax)
# (element rounding fn, emax used for scale rule)
FMT={'MXFP E3M2':(lambda z:fp_round(z,FP['E3M2']),4),'MXFP E2M3':(lambda z:fp_round(z,FP['E2M3']),2),'MXFP E4M1':(lambda z:fp_round(z,FP['E4M1']),8),
     'MXINT6':(int_round,0),'BFP6(sm,5b)':(bfp_round,4)}   # BFP: scale 2^(floor(log2 max)-4) -> max in [16,32)
def blocks(W,bs=BS):
    r,c=W.shape; p=(-c)%bs; Wp=np.pad(W,((0,0),(0,p))); return Wp.reshape(r,-1,bs),c
def quant(W,name,policy='ocp',bs=BS,delta=0):
    rnd,emax=FMT[name]; B,c=blocks(W,bs); mx=np.abs(B).max(-1,keepdims=True)
    se=np.floor(np.log2(np.where(mx>0,mx,1)))-emax+delta
    if policy=='ceil': 
        se=np.where(np.abs(B/2.0**se).max(-1,keepdims=True)>fmax(name),se+1,se)
    se=np.where(mx==0,0,se); se=np.clip(se,-127,127); s=2.0**se
    return (rnd(B/s)*s).reshape(W.shape[0],-1)[:,:c]
def fmax(name):
    return {'MXFP E3M2':28,'MXFP E2M3':7.5,'MXFP E4M1':384,'MXINT6':31/16,'BFP6(sm,5b)':31}[name]
def opt_block(W,name,bs=BS,deltas=(-2,-1,0,1)):
    # per-block MSE-optimal exponent among a small search window
    best=None
    for dl in deltas:
        Q=quant(W,name,'ocp',bs,dl); Bq,_=blocks(Q,bs); Bw,_=blocks(W,bs); err=((Bq-Bw)**2).sum(-1)
        best=err if best is None else np.minimum(best,err)
    return best
def metrics(W,Q):
    Y=W@X; Yq=Q@X
    return np.linalg.norm(Q-W)/np.linalg.norm(W), np.linalg.norm(Yq-Y)/np.linalg.norm(Y)
if __name__=='__main__':
    # sanity: self-check on class FP4 example not applicable; check grid
    for k in 'ABCDE':
        W=d['W_'+k].astype(np.float64)
        print(f'\nDataset {k}   (rel err W / rel err Y, in 1e-2)')
        for n in FMT:
            row=[]
            for pol in ('ocp','ceil'):
                Q=quant(W,n,pol); a,b=metrics(W,Q); row.append(f'{pol}: {100*a:6.3f}/{100*b:6.3f}')
            print(f'  {n:12s}',' | '.join(row))
