"""Outward interval validation from the exact initialized reference.
A prefix enclosure does not establish entry at t=3.4.
Uses exact rational dt=1/5000, exact stored float knot states/velocities,
Hermite defect, and adverse training-gate jumps (see enclosure method).
"""
from interval_bounds import *
import time,csv,argparse,multiprocessing as multiprocessing
parser=argparse.ArgumentParser();parser.add_argument('--reference',type=Path,default=Path('inputs/canonical_initial_reference.npz'));parser.add_argument('--output',type=Path,default=Path('.'));parser.add_argument('--radii',type=Path,default=None);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
if args.reference.is_file():
 saved=np.load(args.reference);X=I(saved['X']);th=saved['theta'];fv=saved['velocity'];offset=int(saved['first_index'])
else:
 X=I(np.load(args.reference/'X.npy'));th=np.load(args.reference/'theta.npy',mmap_mode='r');fv=np.load(args.reference/'velocity.npy',mmap_mode='r');offset=0
N=len(X.c);h=th.shape[1]//2
invN=I(1/N,up(UROUND/N));dt=I(1/5000,up(UROUND/5000));half=dt*.5;dtup=scal(dt.hi());halfup=scal(half.hi());invdt=I(5000.);sqrtNlo=np.nextafter(np.sqrt(float(N)),-np.inf)
xn=normup(X,axis=1);cov=(X.T@X)*invN;lam=scal(np.max(sump(cov.absup(),axis=1)));sqrtlam=up(np.sqrt(lam));mp.iv.dps=40

def vstack(a,b):return I(np.vstack([a.c,b.c]),np.vstack([a.r,b.r]))
def scale_norm(x,factor):return mulp(factor,normup(x))
def maxnormA(A):
 # A is h x 2 x 2. Spectral norm <= sqrt(max row sum * max column sum).
 au=A.absup();return up(np.sqrt(mulp(np.max(sump(au,axis=2),axis=1),np.max(sump(au,axis=1),axis=1))))
def adj(W,act,G,q):return vstack(((q@W.T)*G).T@X*invN,act.T@q*invN)
def apply(W,act,G,p):return act@p[h:]+((X@p[:h].T)*G)@W
def Ai_apply(A,p,transpose=False):
 rows=[]
 for j in range(2):
  val=A[:,0,j]*p[:,0]+A[:,1,j]*p[:,1] if transpose else A[:,j,0]*p[:,0]+A[:,j,1]*p[:,1]
  rows.append(val)
 return stack(rows,axis=1)
def endpoint(k,r,w):
 # mp interval evaluation at exact float upper bounds k,r,w and rational dt.
 K=mp.iv.mpf(float(k));D=mp.iv.mpf(1)/5000
 val=mp.iv.exp(K*D)*mp.iv.mpf(float(r))+(mp.iv.exp(K*D)-1)/K*mp.iv.mpf(float(w))
 return up(float(val.b))
def step(task):
 j,Rbar,beta=task
 a0=I(th[j-offset]);a1=I(th[j+1-offset]);f0=I(fv[j-offset]);f1=I(fv[j+1-offset])
 y=(a0+a1)*.5+(f0-f1)*dt*.125
 v=(a1-a0)*1.5*invdt-(f0+f1)*.25
 a=(f1-f0)*invdt
 jerk=(a0-a1)*12.*(invdt*invdt*invdt)+(f0+f1)*6.*(invdt*invdt)
 U=y[:h];W=y[h:];Z=X@U.T
 assert np.all((Z.lo()>0)|(Z.hi()<0))
 G=(Z.lo()>0).astype(float);e=X-(Z*G)@W
 ex=stack([e[:,0]*X[:,0],e[:,0]*X[:,1],e[:,1]*X[:,0],e[:,1]*X[:,1]],axis=1)
 P=normup(y);h2=mulp(halfup,halfup);h3=mulp(h2,halfup);sixth=up(1/6)
 du=positive_sum(mulp(halfup,normup(v[:h],axis=1)),mulp(.5,mulp(h2,normup(a[:h],axis=1))),mulp(sixth,mulp(h3,normup(jerk[:h],axis=1))))
 motion=positive_sum(mulp(halfup,normup(v)),mulp(.5,mulp(h2,normup(a))),mulp(sixth,mulp(h3,normup(jerk))))
 margin=np.maximum(0,np.nextafter(Z.abslow()/xn[:,None],-np.inf))
 total=addp(Rbar,motion);Dtotal=addp(mulp(P,total),mulp(.5,mulp(total,total)))
 near=margin<=addp(du,Rbar)[None,:]
 def clipped(z,positive):
  lo=np.maximum(z.lo(),0) if positive else np.minimum(z.lo(),0)
  hi=np.maximum(z.hi(),0) if positive else np.minimum(z.hi(),0)
  c=(lo+hi)*.5;rad=np.maximum(up(c-lo),up(hi-c));return I(c,rad)
 fixed=I((G.astype(bool)&~near).astype(float)).T@ex
 alo=((fixed+I(near.astype(float)).T@clipped(ex,False))*invN).lo()
 ahi=((fixed+I(near.astype(float)).T@clipped(ex,True))*invN).hi()
 absA=np.maximum(abs(alo),abs(ahi)).reshape(h,2,2)
 signedA=float(np.max(up(np.sqrt(mulp(np.max(sump(absA,axis=2),axis=1),np.max(sump(absA,axis=1),axis=1))))))
 return dict(step=j+1,smooth_rate_upper=float(addp(signedA,mulp(lam,Dtotal))))

if __name__=='__main__':
 with args.radii.open() as f: rows=list(csv.DictReader(f))
 tasks=[(int(q['step']),float(q['radius']),float(q['beta'])) for q in rows]
 with (args.output/'RECOMPUTED_SMOOTH_RATES.csv').open('w') as f:
  writer=csv.DictWriter(f,fieldnames=['step','smooth_rate_upper']);writer.writeheader()
  with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
   for q in pool.imap(step,tasks,chunksize=8):
    writer.writerow(q)
    if q['step']%200==0:f.flush();print(q['step'],flush=True)
