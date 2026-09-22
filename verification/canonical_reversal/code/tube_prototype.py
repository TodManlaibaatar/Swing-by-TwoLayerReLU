"""A posteriori Euclidean tube prototype for a cubic-Hermite reference.
Real-arithmetic majorants include ReLU gate switches. Numerical padding here is
NOT yet a formal directed-rounding implementation; output is a feasibility test.
"""
from pathlib import Path
import numpy as np,json,csv,time
root=Path.cwd();ref=root/'work/crossing-closure/canonical-reference';X=np.load(ref/'X.npy');th=np.load(ref/'theta.npy',mmap_mode='r');fv=np.load(ref/'velocity.npy',mmap_mode='r');N=len(X);h=th.shape[1]//2;dt=.0002;half=dt/2;steps=len(th)-1
xn=np.linalg.norm(X,axis=1);lam=float(np.linalg.eigvalsh(X.T@X/N)[-1])*(1+1e-12);sqrtlam=np.sqrt(lam);betas=np.array([0,.01,.03,.1,.3,1,3,10,30,100,300,1000.])

def norms(q):return np.linalg.norm(q,axis=1)
def adj(W,act,G,y):return np.vstack([(G*(y@W.T)).T@X/N,act.T@y/N])
def apply(W,act,G,p):return act@p[h:]+(G*(X@p[:h].T))@W

def interval(j,r0):
 y=(th[j]+th[j+1])/2+dt*(fv[j]-fv[j+1])/8
 v=1.5*(th[j+1]-th[j])/dt-(fv[j]+fv[j+1])/4
 a=(fv[j+1]-fv[j])/dt
 jerk=12*(th[j]-th[j+1])/dt**3+6*(fv[j]+fv[j+1])/dt**2
 U=y[:h];W=y[h:];Z=X@U.T;G=Z>0;act=np.maximum(Z,0);e=X-act@W;enrow=norms(e);en=np.linalg.norm(e)/np.sqrt(N);RW=e@W.T;F=np.vstack([(G*RW).T@X/N,act.T@e/N])
 Ai=(G.T@np.column_stack([e[:,0]*X[:,0],e[:,0]*X[:,1],e[:,1]*X[:,0],e[:,1]*X[:,1]])/N).reshape(h,2,2)
 absAi=abs(Ai);Anorm=np.sqrt(np.max(absAi.sum(axis=2),axis=1)*np.max(absAi.sum(axis=1),axis=1));Amax=float(Anorm.max())
 def DF(p):return np.vstack([np.einsum('ijk,ij->ik',Ai,p[h:]),np.einsum('ijk,ik->ij',Ai,p[:h])])-adj(W,act,G,apply(W,act,G,p))
 Jv=apply(W,act,G,v);gvx=G*(X@v[:h].T)
 DJvJv=np.vstack([(G*(Jv@v[h:].T)).T@X/N,gvx.T@Jv/N])
 Fsecond=-2*DJvJv-adj(W,act,G,2*gvx@v[h:])
 d0=np.linalg.norm(F-v);d1=np.linalg.norm(DF(v)-a);d2=np.linalg.norm(Fsecond+DF(a)-jerk)
 vn=np.linalg.norm(v);an=np.linalg.norm(a);jn=np.linalg.norm(jerk);P=np.linalg.norm(y);wn=norms(W)
 du=half*norms(v[:h])+.5*half**2*norms(a[:h])+half**3/6*norms(jerk[:h])+1e-12
 dw=half*norms(v[h:])+.5*half**2*norms(a[h:])+half**3/6*norms(jerk[h:])+1e-12
 motion=half*vn+.5*half**2*an+half**3/6*jn+1e-11;Ps=P+motion
 speed=vn+half*an+.5*half**2*jn;accel=an+half*jn
 Dref=P*motion+.5*motion**2;es=en+sqrtlam*Dref
 L=lam*Ps**2+sqrtlam*es
 third=3*lam*speed**3+9*lam*Ps*speed*accel+L*jn
 eta_smooth=d0+half*d1+.5*half**2*d2+half**3/6*third+2e-9
 # Reference path can itself cross gates; bound the discrepancy from the fixed mask extension.
 margin=np.maximum(0,abs(Z)-1e-11)/xn[:,None]
 possible=margin<=du[None,:]
 activation_defect=np.maximum(0,du[None,:]*xn[:,None]-np.maximum(0,abs(Z)-1e-11))
 ep=enrow+Dref*xn
 rwref=abs(RW)+enrow[:,None]*dw[None,:]+Dref*xn[:,None]*(wn+dw)[None,:]
 jump_u=np.mean(possible*rwref*xn[:,None],axis=0)
 jump_w=np.mean(activation_defect*ep[:,None],axis=0)
 output_defect=activation_defect@(wn+dw)
 eta_gate=float(np.sqrt(jump_u@jump_u+jump_w@jump_w)+sqrtlam*Ps*np.linalg.norm(output_defect)/np.sqrt(N))
 eta=eta_smooth+eta_gate
 Rbar=(r0+dt*eta)*np.exp((Amax+1)*dt)*1.03+1e-12
 cache=[]
 for attempt in range(20):
  if Rbar>.03:return None,dict(reason='radius cap exceeded',step=j,time=j*dt,Rbar=Rbar,eta=eta,eta_gate=eta_gate)
  total=Rbar+motion;Dtotal=P*total+.5*total**2
  near=margin<=(du+Rbar)[None,:]
  Aswitch=np.max(np.mean(near*enrow[:,None]*xn[:,None],axis=0))
  rate=Amax+lam*Dtotal+Aswitch+1e-10
  # Upper bound on positive conormal jump coefficients everywhere in the tube.
  rwpos=np.maximum(0,RW)+enrow[:,None]*(dw+Rbar)[None,:]+Dtotal*xn[:,None]*(wn+dw+Rbar)[None,:]
  weights=rwpos*xn[:,None]/N
  intercept=np.zeros((len(betas),h))
  for i in range(h):
   inds=np.flatnonzero(near[:,i])
   if not len(inds):continue
   ds=np.maximum(0,margin[inds,i]-du[i]);order=np.argsort(ds);ds=ds[order];cs=np.cumsum(weights[inds[order],i])
   intercept[:,i]=np.maximum(0,np.max(cs[None,:]-betas[:,None]*ds[None,:],axis=1))+1e-14
  forcing=np.linalg.norm(intercept,axis=1)
  rates=rate+betas
  rnexts=np.exp(rates*dt)*r0+np.expm1(rates*dt)/rates*(eta+forcing)
  ix=np.argmin(rnexts);rnext=float(rnexts[ix])*(1+1e-12)+1e-15
  if rnext<=Rbar:
   return rnext,dict(step=j+1,time=(j+1)*dt,radius=rnext,rate=float(rate),gate_rate=float(betas[ix]),gate_forcing=float(forcing[ix]),eta=eta,eta_smooth=float(eta_smooth),eta_gate=eta_gate,attempts=attempt+1)
  Rbar=rnext*1.06+1e-12
 return None,dict(reason='tube bootstrap did not close',step=j,time=j*dt,Rbar=Rbar,eta=eta)

r=0.;history=[];t0=time.monotonic();result=None
for j in range(steps):
 rnew,entry=interval(j,r)
 if rnew is None:result=dict(status='FAILED FEASIBILITY; not a validated enclosure',failure=entry);print(json.dumps(result),flush=True);break
 r=rnew;history.append(entry)
 if (j+1)%100==0:print('step',j+1,'time',(j+1)*dt,'R',r,'eta',entry['eta'],'gate rate',entry['gate_rate'],'elapsed',round(time.monotonic()-t0,2),flush=True)
if result is None:result=dict(status='FEASIBLE IN FLOATING POINT; directed-rounding audit still required',terminal=steps*dt,final_radius=r)
result['runtime_seconds']=time.monotonic()-t0
out=root/'outputs/crossing-closure';(out/'TUBE_PROTOTYPE_RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
if history:
 with (out/'TUBE_PROTOTYPE_PROFILE.csv').open('w') as f:
  wr=csv.DictWriter(f,fieldnames=history[0]);wr.writeheader();wr.writerows(history)
print(json.dumps(result,indent=2),flush=True)
