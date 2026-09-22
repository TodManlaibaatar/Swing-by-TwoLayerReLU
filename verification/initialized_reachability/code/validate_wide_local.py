"""Outward interval validation of the LOCAL flow starting at the t=3.4 center.
Does not establish that the initialized trajectory reaches that center.
Uses exact rational dt=1/5000, exact stored float knot states/velocities,
Hermite defect, and adverse training-gate jumps (see enclosure method).
"""
from interval_bounds import *
import time,csv,argparse,multiprocessing as multiprocessing
parser=argparse.ArgumentParser();parser.add_argument('--reference',type=Path,default=Path('work/crossing-closure/canonical-reference'));parser.add_argument('--output',type=Path,default=Path('.'));parser.add_argument('--radii',type=Path,default=None);parser.add_argument('--workers',type=int,default=4);args=parser.parse_args()
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
 j,Rbar=task
 a0=I(th[j-offset]);a1=I(th[j+1-offset]);f0=I(fv[j-offset]);f1=I(fv[j+1-offset])
 y=(a0+a1)*.5+(f0-f1)*dt*.125
 v=(a1-a0)*1.5*invdt-(f0+f1)*.25
 a=(f1-f0)*invdt
 jerk=(a0-a1)*12.*(invdt*invdt*invdt)+(f0+f1)*6.*(invdt*invdt)
 U=y[:h];W=y[h:];Z=X@U.T
 assert np.all((Z.lo()>0)|(Z.hi()<0)),('unresolved midpoint gate',j)
 G=(Z.lo()>0).astype(float);act=Z*G;e=X-act@W;RW=e@W.T
 er=normup(e,axis=1);en=up(normup(e)/sqrtNlo);F=adj(W,act,G,e)
 ex=stack([e[:,0]*X[:,0],e[:,0]*X[:,1],e[:,1]*X[:,0],e[:,1]*X[:,1]],axis=1)
 aa=(I(G).T@ex)*invN;A=I(aa.c.reshape(h,2,2),aa.r.reshape(h,2,2));Amax=float(np.max(maxnormA(A)))
 def DF(p):return vstack(Ai_apply(A,p[h:],True),Ai_apply(A,p[:h]))-adj(W,act,G,apply(W,act,G,p))
 Jv=apply(W,act,G,v);gvx=(X@v[:h].T)*G
 DJvJv=vstack(((Jv@v[h:].T)*G).T@X*invN,gvx.T@Jv*invN)
 Fsecond=DJvJv*(-2.)-adj(W,act,G,(gvx@v[h:])*2.)
 d0=normup(F-v);d1=normup(DF(v)-a);d2=normup(Fsecond+DF(a)-jerk)
 vn=normup(v);an=normup(a);jn=normup(jerk);P=normup(y);wn=normup(W,axis=1)
 h2=mulp(halfup,halfup);h3=mulp(h2,halfup);sixth=up(1/6)
 du=positive_sum(mulp(halfup,normup(v[:h],axis=1)),mulp(.5,mulp(h2,normup(a[:h],axis=1))),mulp(sixth,mulp(h3,normup(jerk[:h],axis=1))))
 dw=positive_sum(mulp(halfup,normup(v[h:],axis=1)),mulp(.5,mulp(h2,normup(a[h:],axis=1))),mulp(sixth,mulp(h3,normup(jerk[h:],axis=1))))
 motion=positive_sum(mulp(halfup,vn),mulp(.5,mulp(h2,an)),mulp(sixth,mulp(h3,jn)))
 Ps=addp(P,motion);speed=positive_sum(vn,mulp(halfup,an),mulp(.5,mulp(h2,jn)));accel=addp(an,mulp(halfup,jn))
 Dref=addp(mulp(P,motion),mulp(.5,mulp(motion,motion)));es=addp(en,mulp(sqrtlam,Dref));L=addp(mulp(lam,mulp(Ps,Ps)),mulp(sqrtlam,es))
 third=positive_sum(mulp(mulp(3.,lam),mulp(mulp(speed,speed),speed)),mulp(mulp(mulp(mulp(9.,lam),Ps),speed),accel),mulp(L,jn))
 eta_smooth=positive_sum(d0,mulp(halfup,d1),mulp(.5,mulp(h2,d2)),mulp(sixth,mulp(h3,third)))
 margin=np.maximum(0,np.nextafter(Z.abslow()/xn[:,None],-np.inf))
 possible=margin<=du[None,:]
 activation=np.maximum(0,up(mulp(du[None,:],xn[:,None])-Z.abslow()))
 ep=addp(er,mulp(Dref,xn));rwref=positive_sum(RW.absup(),mulp(er[:,None],dw[None,:]),mulp(mulp(Dref,xn[:,None]),addp(wn,dw)[None,:]))
 jumpu=mulp(sump(mulp(mulp(possible,rwref),xn[:,None]),axis=0),up(1/N));jumpw=mulp(sump(mulp(activation,ep[:,None]),axis=0),up(1/N))
 outputdefect=dotp(activation,addp(wn,dw))
 eta_gate=addp(up(np.sqrt(addp(sump(mulp(jumpu,jumpu)),sump(mulp(jumpw,jumpw))))),mulp(mulp(sqrtlam,Ps),up(normup(outputdefect)/sqrtNlo)))
 eta=addp(eta_smooth,eta_gate)
 for attempt in range(1):
  if Rbar>.001:raise RuntimeError(('tube radius exceeded',j,Rbar))
  total=addp(Rbar,motion);Dtotal=addp(mulp(P,total),mulp(.5,mulp(total,total)))
  near=margin<=addp(du,Rbar)[None,:]
  Aswitch=np.max(mulp(sump(mulp(mulp(near,er[:,None]),xn[:,None]),axis=0),up(1/N)))
  rate=positive_sum(Amax,mulp(lam,Dtotal),Aswitch)
  rwpos=positive_sum(np.maximum(0,RW.hi()),mulp(er[:,None],addp(dw,Rbar)[None,:]),mulp(mulp(Dtotal,xn[:,None]),addp(addp(wn,dw),Rbar)[None,:]))
  weights=mulp(mulp(rwpos,xn[:,None]),up(1/N));intercepts=np.zeros(h)
  beta=1.
  for i in range(h):
   inds=np.flatnonzero(near[:,i])
   if not len(inds):continue
   ds=np.maximum(0,np.nextafter(margin[inds,i]-du[i],-np.inf));order=np.argsort(ds);ds=ds[order]
   cs=addp(up(np.cumsum(weights[inds[order],i])/np.nextafter(1-gamma(len(inds)),-np.inf)),len(inds)*TINY)
   intercepts[i]=max(0,float(np.max(up(cs-ds))))
  forcing=normup(intercepts)
  return dict(step=j+1,time=(j+1)/5000,rate_upper=float(rate),gate_rate=beta,gate_forcing_upper=float(forcing),eta_upper=float(eta),eta_smooth_upper=float(eta_smooth),eta_gate_upper=float(eta_gate),tube_radius_upper=float(Rbar))

if __name__=='__main__':
 r=up(1e-5);history=[];started=time.monotonic()
 with (args.radii or args.output/'LOCAL_TUBE_RADII.csv').open() as f:
  radii=list(csv.DictReader(f))
 tasks=[(int(row['step']),float(row['radius'])) for row in radii]
 try:
  with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
   for entry in pool.imap(step,tasks,chunksize=4):
    r=endpoint(addp(entry['rate_upper'],entry['gate_rate']),r,addp(entry['eta_upper'],entry['gate_forcing_upper']))
    assert r<entry['tube_radius_upper'],('tube exited',entry['step'],r,entry['tube_radius_upper'])
    entry['radius_upper']=float(r);history.append(entry)
    if entry['step']%100==0:print('step',entry['step'],'radius',r,'elapsed',round(time.monotonic()-started,2),flush=True)
  result=dict(status='LOCAL FLOW ENCLOSED FROM A BALL AT THE t=3.4 NUMERICAL CENTER; INITIALIZED REACHABILITY NOT CERTIFIED',initial_radius=1e-5,initial_reference_index=17000,final_reference_index=19500,dt='1/5000',final_radius_upper=float(r),inside_static_witness_ball=bool(r<5e-5))
 except Exception as err:
  result=dict(status='VALIDATION FAILED',error=repr(err),last_completed_step=history[-1]['step'] if history else None)
 result['runtime_seconds']=time.monotonic()-started
 args.output.mkdir(exist_ok=True,parents=True);(args.output/'WIDE_LOCAL_FLOW_CERTIFICATE.json').write_text(json.dumps(result,indent=2)+'\n')
 if history:
  with (args.output/'WIDE_LOCAL_FLOW_ENCLOSURE.csv').open('w') as f:
   w=csv.DictWriter(f,fieldnames=history[0]);w.writeheader();w.writerows(history)
 print(json.dumps(result,indent=2),flush=True)
