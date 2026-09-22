from pathlib import Path
import numpy as np,json,csv
root=Path.cwd();n=np.load(root/'work/appendix-repository/swing_grid_preregistered/preregistration_v1_postspecialization/cells/mu2_2p000__sigma_0p150/training_state.npz');I=n['strong_mask'].astype(bool);rng=np.random.default_rng(0);N=4000;h=200
X=np.vstack([np.column_stack([3+.15*rng.standard_normal(2000),.15*rng.standard_normal(2000)]),np.column_stack([.15*rng.standard_normal(2000),2+.15*rng.standard_normal(2000)])]);xn=np.linalg.norm(X,axis=1);lam=np.linalg.eigvalsh(X.T@X/N)[-1];records=[]
for j in range(0,391,2):
 U=n['U'][j];W=n['W'][j];Z=X@U.T;G=Z>0;act=np.maximum(Z,0);F=act@W;e=X-F;RW=e@W.T;dU=(G*RW).T@X/N;dW=act.T@e/N;dq=X@dU.T
 A=G.T@np.column_stack([e[:,0]*X[:,0],e[:,0]*X[:,1],e[:,1]*X[:,0],e[:,1]*X[:,1]])/N;A=A.reshape(h,2,2);an=np.linalg.svd(A,compute_uv=False)[:,0].max();en=np.sqrt(np.sum(e*e)/N)
 row=dict(t=float(n['time'][j]),residual_norm=float(en),smooth_log_norm=float(an),loss_bound_log_norm=float(np.sqrt(lam)*en),parameter_speed=float(np.sqrt(np.sum(dU*dU)+np.sum(dW*dW))))
 for dt in [2e-4,1e-4,1e-5]:
  possible=abs(Z)<=dt*abs(dq)
  fi=np.sum(possible*abs(RW)*xn[:,None],axis=0)/N
  row[f'gate_defect_{dt}']=float(np.linalg.norm(fi));row[f'count_{dt}']=int(possible.sum())
 records.append(row)
t=np.array([r['t'] for r in records]);a=np.array([r['smooth_log_norm'] for r in records]);integ=np.r_[0,np.cumsum(.5*(a[:-1]+a[1:])*np.diff(t))];factors=np.exp(integ[-1]-integ)
summary=dict(status='Feasibility diagnostic only, not an enclosure. Linearized gate sweep and sampled log norm are not rigorous bounds.',smooth_log_norm_integral=float(integ[-1]),smooth_amplification=float(np.exp(integ[-1])))
for dt in [2e-4,1e-4,1e-5]:summary[f'predicted_gate_error_{dt}']=float(np.trapz(factors*np.array([r[f'gate_defect_{dt}'] for r in records]),t))
with (root/'outputs/crossing-closure/CERTIFICATE_FEASIBILITY.csv').open('w') as f:
 wr=csv.DictWriter(f,fieldnames=records[0]);wr.writeheader();wr.writerows(records)
(root/'outputs/crossing-closure/CERTIFICATE_FEASIBILITY.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
