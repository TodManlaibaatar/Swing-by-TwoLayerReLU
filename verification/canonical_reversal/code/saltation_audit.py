from pathlib import Path
import numpy as np,json,time
root=Path.cwd();ref=root/'work/crossing-closure/canonical-reference';X=np.load(ref/'X.npy');th=np.load(ref/'theta.npy',mmap_mode='r');fv=np.load(ref/'velocity.npy',mmap_mode='r');h=200;N=len(X);old=X@th[0,:h].T;logs=np.zeros(h);count=0;adverse=0;ratio_max=1.;ambiguous=0;t0=time.monotonic()
for j in range(len(th)-1):
 new=X@th[j+1,:h].T;nn,ii=np.where((old>0)!=(new>0))
 for n,i in zip(nn,ii):
  s=float(-old[n,i]/(new[n,i]-old[n,i]));theta=(1-s)*th[j]+s*th[j+1];u=theta[:h];w=theta[h:];x=X[n];ex=x-np.maximum(u@x,0)@w;da=1. if new[n,i]>0 else -1.;dv=da*float(w[i]@ex)*float(x@x)/N
  vl=float(x@fv[j,i]);vr=float(x@fv[j+1,i]);vm=(1-s)*vl+s*vr-s*dv;vp=vm+dv
  count+=1
  if da*vm<=0 or da*vp<=0:ambiguous+=1;continue
  ratio=vp/vm
  if ratio>1:logs[i]+=np.log(ratio);adverse+=1
  ratio_max=max(ratio_max,ratio)
 old=new
 if (j+1)%4000==0:print('time',(j+1)*.0002,'events',count,'elapsed',round(time.monotonic()-t0,2),flush=True)
result=dict(status='Retrospective numerical saltation diagnostic, not a validated event enclosure. Crossing state and side velocities are interpolated.',events=count,adverse_events=adverse,ambiguous_interpolated_events=ambiguous,max_single_event_ratio=ratio_max,max_neuron_sum_positive_log_ratio=float(logs.max()),sum_over_all_neurons_positive_log_ratio=float(logs.sum()),runtime_seconds=time.monotonic()-t0)
(root/'outputs/crossing-closure/SALTATION_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
