from pathlib import Path
import numpy as np,json
root=Path.cwd();ref=root/'work/crossing-closure/canonical-reference';th=np.load(ref/'theta.npy',mmap_mode='r');F=np.load(ref/'velocity.npy',mmap_mode='r');arc=np.load(root/'work/appendix-repository/swing_grid_preregistered/preregistration_v1_postspecialization/cells/mu2_2p000__sigma_0p150/training_state.npz');I=arc['strong_mask'].astype(bool);r=np.sin(np.pi/36);xi=np.array([r,-np.sqrt(1-r*r)]);v=np.array([r,-1]);records=[]
for j in range(0,len(th),50):
 U=th[j,:200];W=th[j,200:];du=F[j,:200];dw=F[j,200:];M=W[I].T@U[I];dM=dw[I].T@U[I]+W[I].T@du[I];Y=(M[0,0]-1)*r-M[0,1];Qg=float(Y*(dM[0]@v)+dM[1]@v);q=U@xi;act=q>0;fp=np.maximum(q,0)@W;df=np.maximum(q,0)@dw+((du@xi)*act)@W
 records.append(dict(t=j*.0002,Q_G=Qg,full_rate=float((fp-xi)@df),full_error=float(.5*np.sum((fp-xi)**2)),m11=float(M[0,0]),core_mass=float(np.sum(U[I,0]*W[I,0])),max_input_angle_deg=float(np.degrees(np.max(abs(np.arctan2(U[I,1],U[I,0])))))))
def cross(key):
 for a,b in zip(records,records[1:]):
  if a['t']>=2.5 and a[key]<0<=b[key]:return a['t']-a[key]*(b['t']-a['t'])/(b[key]-a[key])
result=dict(status='Higher-order numerical reference only; not validated gradient-flow witnesses.',Q_G_interpolated_crossing=cross('Q_G'),full_rate_interpolated_crossing=cross('full_rate'),snapshots=[min(records,key=lambda a:abs(a['t']-t)) for t in [3.4,3.6,3.9,4.]])
(root/'outputs/crossing-closure/RK4_REFERENCE_WITNESSES.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
