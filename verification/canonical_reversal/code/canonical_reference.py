"""Higher-order reference for validation, same archived canonical initialization.
This computation alone is NOT a rigorous enclosure or a new probability result.
"""
from pathlib import Path
import numpy as np,json,time,hashlib
root=Path.cwd();out=root/'work/crossing-closure/canonical-reference';out.mkdir(exist_ok=True)
p=root/'work/appendix-repository/swing_grid_preregistered/preregistration_v1_postspecialization/cells/mu2_2p000__sigma_0p150/training_state.npz';n=np.load(p);rng=np.random.default_rng(0);N=4000;h=200;dt=.0002;steps=20000
X=np.vstack([np.column_stack([3+.15*rng.standard_normal(2000),.15*rng.standard_normal(2000)]),np.column_stack([.15*rng.standard_normal(2000),2+.15*rng.standard_normal(2000)])]);np.save(out/'X.npy',X);theta=np.vstack([n['U'][0],n['W'][0]]).copy()
traj=np.lib.format.open_memmap(out/'theta.npy',mode='w+',dtype=np.float64,shape=(steps+1,2*h,2));vel=np.lib.format.open_memmap(out/'velocity.npy',mode='w+',dtype=np.float64,shape=traj.shape)
def rhs(th):
 U=th[:h];W=th[h:];Z=X@U.T;act=np.maximum(Z,0);err=X-act@W
 return np.vstack([((Z>0)*(err@W.T)).T@X/N,act.T@err/N])
t0=time.monotonic()
for j in range(steps):
 k1=rhs(theta);traj[j]=theta;vel[j]=k1;k2=rhs(theta+dt/2*k1);k3=rhs(theta+dt/2*k2);k4=rhs(theta+dt*k3);theta=theta+dt/6*(k1+2*k2+2*k3+k4)
 if (j+1)%500==0:
  traj.flush();vel.flush();print('step',j+1,'time',(j+1)*dt,'elapsed',round(time.monotonic()-t0,2),flush=True)
traj[-1]=theta;vel[-1]=rhs(theta);traj.flush();vel.flush()
meta=dict(status='numerical reference only; enclosure pending',method='RK4',dt=dt,steps=steps,N=N,h=h,mu1=3,mu2=2,sigma=.15,seed=0,initialization='exact stored float values at t=0 in training_state.npz',dataset='original canonical seeded dataset; float values written to X.npy',source_archive_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),runtime_seconds=time.monotonic()-t0)
(out/'metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta,indent=2),flush=True)
