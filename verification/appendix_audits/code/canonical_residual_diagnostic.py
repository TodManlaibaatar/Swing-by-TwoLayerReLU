from pathlib import Path
import sys,json,csv,math,subprocess
import numpy as np
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--repository',required=True);parser.add_argument('--output',default='.');args=parser.parse_args()
root=Path(args.repository).resolve();sys.path.insert(0,str(root))
import theory_guided_swing_by_phase_diagram as exp
from full_flow_diagnostic import Config
from dataclasses import replace
cfg=replace(Config(),max_time=10.)
X,_=exp.make_data(cfg)
a=np.load(root/'swing_grid_preregistered/preregistration_v1_postspecialization/cells/mu2_2p000__sigma_0p150/training_state.npz')
t=a['time']; v=np.array([np.cos(np.deg2rad(-85)),np.sin(np.deg2rad(-85))]);I=a['strong_mask'].astype(bool);J=a['weak_mask'].astype(bool);O=a['remainder_mask'].astype(bool)
rows=[]
for k in np.flatnonzero((t>=2.5)&(t<=5)):
 U,W=a['U'][k],a['W'][k];dU,dW,_=exp.relu_rhs(U,W,X);q=U@v;act=q>0;f=np.maximum(q,0)@W;df=np.maximum(q,0)@dW+(act*(dU@v))@W
 M=W[I].T@U[I];dM=dW[I].T@U[I]+W[I].T@dU[I]
 R=f-M@v;Rd=df-dM@v;er=(M@v-v)@(dM@v);ef=(f-v)@df
 G=((M[0,0]-1)*v[0]-M[0,1])*(dM[0,0]*v[0]-dM[0,1])+v[0]*dM[1,0]-dM[1,1]
 rows.append(dict(time=float(t[k]),R=float(np.linalg.norm(R)),Rdot=float(np.linalg.norm(Rd)),relative_R=float(np.linalg.norm(R)/np.linalg.norm(f-v)),G=float(G),edot_reduced=float(er),edot_full=float(ef),inactive_strong=int((I&~act).sum()),active_weak=int((J&act).sum()),active_remainder=int((O&act).sum()),remainder_norm=float(np.linalg.norm(np.maximum(q[O],0)@W[O])),strong_mismatch_norm=float(np.linalg.norm(np.maximum(-q[I],0)@W[I]))))
# Recompute measurements from saved trajectories; do not integrate or train.
def cross(col):
 out=[]
 for x,y in zip(rows,rows[1:]):
  if x[col]<=0<y[col]:out.append(x['time']-x[col]*(y['time']-x['time'])/(y[col]-x[col]))
 return out
phase=list(csv.DictReader((root/'theory_guided_swing_phase_diagram_v3/phase_diagram.csv').open()))
incone=lambda r: float(r['x1'])>=-1e-12 and float(r['x2'])>=-1e-12
counts={'total':len(phase),'inside_closed_positive_cone':sum(map(incone,phase)),'outside':sum(not incone(r) for r in phase),'swing_above_threshold':sum(float(r['actual_persistent_swing'])>1e-4 for r in phase)}
switch=[r for r in phase if int(r['selector_switch_count']) and float(r['actual_persistent_swing'])>1e-4]
switch.sort(key=lambda r:-float(r['actual_persistent_swing']))
# Compute full-horizon compositional output from stored states.
vp=np.array([3.,2.])/math.sqrt(13);F=np.einsum('thi,th->ti',a['W'],np.maximum(a['U']@vp,0));E=((F-vp)**2).sum(1)/2
crosses={p.parent.name:len(json.loads(p.read_text())['predicted_p1_G_timing']['all_G_zero_crossings_fd']) for p in (root/'swing_grid_preregistered/preregistration_v1_postspecialization/cells').glob('*/preregistration.json')}
result={'repository_commit':subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip(),'method':'Read saved states and evaluate RHS only; no training or integration.','residual':{k:max(r[k] for r in rows) for k in ['R','Rdot','relative_R','inactive_strong','active_weak','active_remainder','remainder_norm','strong_mismatch_norm']},'strong_count':int(I.sum()),'G_reduced_max_discrepancy':max(abs(r['G']-r['edot_reduced']) for r in rows),'G_reduced_correlation':float(np.corrcoef([r['G'] for r in rows],[r['edot_reduced'] for r in rows])[0,1]),'crossings':{k:cross(k) for k in ['G','edot_reduced','edot_full']},'phase_counts':counts,'compositional':{'angle_deg':float(np.degrees(np.arctan2(2,3))),'max_successive_error_increment':float(np.diff(E).max()),'E0':float(E[0]),'E10':float(E[-1])},'grid_G_crossing_counts':crosses,'switch_rows_count':len(switch),'top26':{'max_absolute_timing_difference':max(abs(float(r['t_reversal_minus_nearest_switch'])) for r in switch[:26]),'positive_predicted_delta':sum(float(r['nearest_switch_predicted_delta'])>0 for r in switch[:26])}}
out=Path(args.output);out.mkdir(parents=True,exist_ok=True);(out/'EMPIRICAL_VERIFICATION.json').write_text(json.dumps(result,indent=2));
with (out/'CANONICAL_RESIDUAL_DIAGNOSTIC.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps(result,indent=2))
