from pathlib import Path
import numpy as np,json,csv,hashlib
root=Path.cwd();repo=root/'work/appendix-repository';cell=repo/'swing_grid_preregistered/preregistration_v1_postspecialization/cells/mu2_2p000__sigma_0p150'
n=np.load(cell/'training_state.npz');I=n['strong_mask'].astype(bool);N=4000;npcl=2000
rng=np.random.default_rng(0)
X=np.vstack([np.column_stack([3+.15*rng.standard_normal(npcl),.15*rng.standard_normal(npcl)]),np.column_stack([.15*rng.standard_normal(npcl),2+.15*rng.standard_normal(npcl)])])
rows=[]
for t in [2.9,3.,3.4,3.6,3.7,4.,4.5]:
 j=np.argmin(abs(n['time']-t));U=n['U'][j];W=n['W'][j];Z=X@U.T;G=(Z>0).astype(float);act=np.maximum(Z,0);F=act@W;err=X-F
 dW=act.T@err/N;dU=(G*(err@W.T)).T@X/N
 A=np.einsum('ni,nj,nk->ijk',G,err,X)/N
 A2=np.einsum('ni,nj,nk->ijk',G[npcl:],err[npcl:],X[npcl:])/N
 M=W[I].T@U[I];dM=dW[I].T@U[I]+W[I].T@dU[I]
 r=np.sin(np.pi/36);Y=(M[0,0]-1)*r-M[0,1];Q=Y*(r*dM[0,0]-dM[0,1])+r*dM[1,0]-dM[1,1]
 x=X[npcl:];g=G[npcl:,I];a,b=U[I].T;c,d=W[I].T;F_I=act[npcl:,I]@c;F_O=F[npcl:,0]-F_I
 h=x[:,1]*(g@(c*d));lag=x[:,1]*(g@(c*(d-b)));noise=-x[:,0]*(g@(c*a))
 assert np.max(abs(h-F_I-lag-noise))<1e-12
 S2=-np.sum(A2[I,0,1]*c*d)
 sq=.5*np.mean(F_I**2);comp=.5*np.mean(F_O*F_I);target=-.5*np.mean(x[:,0]*F_I);lagterm=.5*np.mean((F[npcl:,0]-x[:,0])*lag);noiseterm=.5*np.mean((F[npcl:,0]-x[:,0])*noise)
 assert abs(S2-sq-comp-target-lagterm-noiseterm)<1e-12
 # Alternative square: output Gram h is the feature, and error is aligned with it.
 v=F[npcl:,0]-x[:,0];hv=np.mean(h*v);hh=np.mean(h*h);vv=np.mean(v*v)
 # Clusterwise own-feature correct residual and diagonal opposition.
 d22diag=np.sum(A2[I,1,1]*(b*b+d*d))
 rows.append(dict(t=float(n['time'][j]),Q=Q,S2=S2,O=S2-Q,positive_square=sq,complement=comp,target=target,lag=lagterm,noise=noiseterm,output_feature_norm=np.sqrt(hh),wrong_residual_norm=np.sqrt(vv),alignment=hv/np.sqrt(hh*vv),feature_projection=hv/hh,weak_diagonal_opposition=d22diag,weak_e2_norm=np.sqrt(np.mean(err[npcl:,1]**2)),active_cd_mean=float(np.mean(g@(c*d))),d_min=float(d.min()),d_max=float(d.max())))
path=root/'outputs/crossing-closure/SIGNED_SOURCE_DIAGNOSTICS.csv'
with path.open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
print(json.dumps(rows,indent=2))
