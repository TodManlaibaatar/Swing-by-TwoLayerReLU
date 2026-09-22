from pathlib import Path
import numpy as np,json,csv
root=Path.cwd();n=np.load(root/'work/appendix-repository/swing_grid_preregistered/preregistration_v1_postspecialization/cells/mu2_2p000__sigma_0p150/training_state.npz');rng=np.random.default_rng(0);N=4000;h=200
X=np.vstack([np.column_stack([3+.15*rng.standard_normal(2000),.15*rng.standard_normal(2000)]),np.column_stack([.15*rng.standard_normal(2000),2+.15*rng.standard_normal(2000)])]);lam=np.linalg.eigvalsh(X.T@X/N)[-1]
def rhs(theta,dir=None):
 U=theta[:h];W=theta[h:];Z=X@U.T;G=Z>0;act=np.maximum(Z,0);e=X-act@W;RW=e@W.T;F=np.vstack([(G*RW).T@X/N,act.T@e/N]);A=(G.T@np.column_stack([e[:,0]*X[:,0],e[:,0]*X[:,1],e[:,1]*X[:,0],e[:,1]*X[:,1]])/N).reshape(h,2,2)
 eig=np.linalg.svd(A,compute_uv=False)[:,0].max();result=(F,float(eig),float(np.sqrt(np.sum(e*e)/N)))
 if dir is not None:
  du=dir[:h];dw=dir[h:];df=act@dw+(G*(X@du.T))@W;Jt=np.vstack([(G*(df@W.T)).T@X/N,act.T@df/N]);He=np.vstack([np.einsum('ijk,ij->ik',A,dw),np.einsum('ijk,ik->ij',A,du)]);return result+(He-Jt,)
 return result
T=np.concatenate([n['U'][:401],n['W'][:401]],axis=1);F=[]
for theta in T:F.append(rhs(theta)[0])
F=np.array(F);out=[]
for j in range(400):
 dt=float(n['time'][j+1]-n['time'][j]);theta=(T[j]+T[j+1])/2+dt*(F[j]-F[j+1])/8;dy=1.5*(T[j+1]-T[j])/dt-(F[j]+F[j+1])/4;ddy=(F[j+1]-F[j])/dt
 fm,a,en,df=rhs(theta,dy);ddd=12*(T[j]-T[j+1])/dt**3+6*(F[j]+F[j+1])/dt**2
 out.append(dict(t=float((n['time'][j]+n['time'][j+1])/2),dt=dt,defect_norm=float(np.linalg.norm(fm-dy)),defect_derivative_norm=float(np.linalg.norm(df-ddy)),smooth_log_norm=a,loss_log_norm=float(np.sqrt(lam)*en),speed=float(np.linalg.norm(dy)),accel=float(np.linalg.norm(ddy)),jerk=float(np.linalg.norm(ddd))))
t=np.array([q['t'] for q in out]);a=np.array([q['smooth_log_norm'] for q in out]);defect=np.array([q['defect_norm'] for q in out]);integ=np.r_[0,np.cumsum((a[1:]+a[:-1])*.005)];R=[]
for k in [339,389,399]:R.append(dict(terminal=float(t[k]),smooth_weighted_midpoint_defect=float(np.trapz(np.exp(integ[k]-integ[:k+1])*defect[:k+1],t[:k+1]))))
summary=dict(status='Feasibility only: sampled midpoint defects, omitted tube growth and gate terms; not an error enclosure.',radii=R,max_defect=float(defect.max()))
(root/'outputs/crossing-closure/HERMITE_DEFECT_DIAGNOSTIC.json').write_text(json.dumps(summary,indent=2)+'\n')
with (root/'outputs/crossing-closure/HERMITE_DEFECT_PROFILE.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=out[0]);w.writeheader();w.writerows(out)
np.savez(root/'work/crossing-closure/hermite_reference.npz',time=n['time'][:401],theta=T,velocity=F)
print(json.dumps(summary,indent=2))
