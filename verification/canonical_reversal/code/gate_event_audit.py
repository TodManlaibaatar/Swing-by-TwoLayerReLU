from pathlib import Path
import numpy as np,json,time
root=Path.cwd();ref=root/'work/crossing-closure/canonical-reference';X=np.load(ref/'X.npy');th=np.load(ref/'theta.npy',mmap_mode='r');fv=np.load(ref/'velocity.npy',mmap_mode='r');h=200;dt=.0002;t0=time.monotonic();old=X@th[0,:h].T;count=0;rev=[];min_speed=1e99;first=[]
for j in range(len(th)-1):
 new=X@th[j+1,:h].T;nn,ii=np.where((old>0)!=(new>0));count+=len(ii)
 if len(ii):
  vl=np.einsum('ij,ij->i',X[nn],fv[j,ii]);vr=np.einsum('ij,ij->i',X[nn],fv[j+1,ii]);direction=np.where(new[nn,ii]>0,1.,-1.);speed=np.minimum(direction*vl,direction*vr)
  if len(first)<5:
   for k in range(min(5-len(first),len(ii))):first.append(dict(step=j,sample=int(nn[k]),neuron=int(ii[k]),q0=float(old[nn[k],ii[k]]),q1=float(new[nn[k],ii[k]]),v0=float(vl[k]),v1=float(vr[k])))
  for k in np.flatnonzero(speed<=0):
   rev.append(dict(step=j,time=j*dt,sample=int(nn[k]),neuron=int(ii[k]),q0=float(old[nn[k],ii[k]]),q1=float(new[nn[k],ii[k]]),v0=float(vl[k]),v1=float(vr[k])))
  min_speed=min(min_speed,float(speed.min()))
 old=new
 if (j+1)%2000==0:print('time',(j+1)*dt,'crossings',count,'reversing',len(rev),'elapsed',round(time.monotonic()-t0,2),flush=True)
result=dict(status='Numerical event audit only; endpoint speed reversals are candidate sliding/chattering events, not validated classification.',total_step_gate_sign_changes=count,reversing_endpoint_speed_events=len(rev),minimum_oriented_endpoint_speed=min_speed,first_events=first,first_reversing_events=rev[:25],runtime_seconds=time.monotonic()-t0)
(root/'outputs/crossing-closure/GATE_EVENT_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
