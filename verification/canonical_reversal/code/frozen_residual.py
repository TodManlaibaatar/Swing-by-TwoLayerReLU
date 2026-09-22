from pathlib import Path
import numpy as np,json,csv
root=Path.cwd();cell=root/'work/appendix-repository/swing_grid_preregistered/preregistration_v1_postspecialization/cells/mu2_2p000__sigma_0p150'
n=np.load(cell/'training_state.npz');I=n['strong_mask'].astype(bool);N=4000;h=200
rng=np.random.default_rng(0)
X=np.vstack([np.column_stack([3+.15*rng.standard_normal(2000),.15*rng.standard_normal(2000)]),np.column_stack([.15*rng.standard_normal(2000),2+.15*rng.standard_normal(2000)])])
r=np.sin(np.pi/36);v0=np.array([r,-1.])
out=[]
for t in [2.9,3.,3.2,3.4,3.5]:
 j=np.argmin(abs(n['time']-t));U=n['U'][j];W=n['W'][j];Z=X@U.T;G=(Z>0).astype(float);act=np.maximum(Z,0);F=act@W;err=X-F;eflat=err.reshape(-1)/np.sqrt(N)
 J=np.zeros((N,2,2*h,2))
 for k in range(2):
  J[:,k,:h,:]=G[:,:,None]*W[None,:,k,None]*X[:,None,:]
  J[:,k,h:,k]=act
 J=J.reshape(2*N,4*h)/np.sqrt(N)
 b=J.T@eflat;M=W[I].T@U[I];ell=np.array([(M[0,0]-1)*r-M[0,1],1.])
 grad=np.zeros((2*h,2));grad[:h][I]=(W[I]@ell)[:,None]*v0;grad[h:][I]=(U[I]@v0)[:,None]*ell
 grad=grad.reshape(-1)
 ev,V=np.linalg.eigh(J.T@J);coef=(grad@V)*(V.T@b)
 b1=J[:4000].T@eflat[:4000];b2=J[4000:].T@eflat[4000:]
 coef1=(grad@V)*(V.T@b1);coef2=(grad@V)*(V.T@b2)
 assert np.max(abs(coef-coef1-coef2))<1e-12
 dt=np.linspace(0,1.5,1501);q=coef@np.exp(-ev[:,None]*dt)
 cross=np.flatnonzero((q[:-1]<0)&(q[1:]>=0));tc=float(t+dt[cross[0]]-q[cross[0]]*(dt[1]-dt[0])/(q[cross[0]+1]-q[cross[0]])) if len(cross) else None
 # A single frozen model is diagnostic, not a validated flow replacement.
 out.append(dict(entry_time=t,initial_Q=float(grad@b),frozen_crossing=tc,loss=float(.5*eflat@eflat),grad_norm=float(np.linalg.norm(grad)),J_norm=float(np.sqrt(ev[-1])),kernel_spectral_max=float(ev[-1]),positive_coeff_mass=float(coef[coef>0].sum()),negative_coeff_mass=float(-coef[coef<0].sum()),positive_mean_rate=float(coef[coef>0]@ev[coef>0]/coef[coef>0].sum()),negative_mean_rate=float(coef[coef<0]@ev[coef<0]/coef[coef<0].sum()),Q_after_half=float(coef@np.exp(-ev*.5)),frozen_initial_slope=float(-coef@ev),strong_initial_residual_half=float(coef1@np.exp(-ev*.5)),weak_initial_residual_half=float(coef2@np.exp(-ev*.5))))
 np.savez(root/f'outputs/crossing-closure/frozen_spectrum_{t:.1f}.npz',rates=ev,coefficients=coef,strong_initial_residual=coef1,weak_initial_residual=coef2,entry=t)
(root/'outputs/crossing-closure/FROZEN_RESIDUAL_DIAGNOSTICS.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
