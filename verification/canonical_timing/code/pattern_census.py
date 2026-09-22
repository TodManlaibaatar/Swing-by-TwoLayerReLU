"""Deterministic reference census only; no trajectory/probability certification."""
from pathlib import Path
import numpy as np,csv,json,hashlib,time
p=Path('work/crossing-closure/canonical-reference');out=Path('outputs/canonical-timing');X=np.load(p/'X.npy');T=np.load(p/'theta.npy',mmap_mode='r');V=np.load(p/'velocity.npy',mmap_mode='r');N=len(X);h=200;xn=np.linalg.norm(X,axis=1);previous=None;rows=[];switchpairs=0;switchneurons=0;start=time.monotonic()
with (out/'TRAINING_PATTERN_CENSUS.csv').open('w') as f:
 fields=['step','time','distinct_patterns','nondead_patterns','dead_neurons','largest_group','largest_live_group','changed_neurons_since_previous_knot','changed_pairs_since_previous_knot','near_boundary_neurons_linear_step','group_sizes'];writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
 for j in range(len(T)):
  U=T[j,:h];q=X@U.T;mask=q>0;packed=np.packbits(mask.T,axis=1);_,counts=np.unique(packed,axis=0,return_counts=True);dead=int((~mask.any(axis=0)).sum());change=0 if previous is None else int((mask!=previous).any(axis=0).sum());pairs=0 if previous is None else int((mask!=previous).sum());previous=mask
  near=int((np.min(abs(q)/xn[:,None],axis=0)<=np.linalg.norm(V[j,:h],axis=1)/5000).sum());livecounts=sorted(counts.tolist());
  if dead:livecounts.remove(dead)
  row=dict(step=j,time=j/5000,distinct_patterns=len(counts),nondead_patterns=len(counts)-int(dead>0),dead_neurons=dead,largest_group=int(max(counts)),largest_live_group=max(livecounts,default=0),changed_neurons_since_previous_knot=change,changed_pairs_since_previous_knot=pairs,near_boundary_neurons_linear_step=near,group_sizes=json.dumps(sorted(counts.tolist(),reverse=True),separators=(',',':')));writer.writerow(row);switchpairs+=pairs;switchneurons+=change
  if j%1000==0:f.flush();rows.append(row);print(j,flush=True)
report=dict(status='DIAGNOSTIC ONLY',reference_knot_count=len(T),scope='Counts are sampled-reference geometry, not validated crossing counts or certified actual-flow patterns.',total_adjacent_knot_changed_pairs=switchpairs,total_adjacent_knot_changed_neurons=switchneurons,coarse_checkpoints=rows,runtime_seconds=time.monotonic()-start,inputs={n:hashlib.sha256((p/n).read_bytes()).hexdigest() for n in ['X.npy','theta.npy','velocity.npy']})
(out/'TRAINING_PATTERN_CENSUS.json').write_text(json.dumps(report,indent=2)+'\n')
