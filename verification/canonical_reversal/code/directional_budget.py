from pathlib import Path
import numpy as np,json
root=Path.cwd();n=np.load(root/'work/appendix-repository/swing_grid_preregistered/preregistration_v1_postspecialization/cells/mu2_2p000__sigma_0p150/training_state.npz');I=n['strong_mask'].astype(bool);rng=np.random.default_rng(0);N=4000;h=200
X=np.vstack([np.column_stack([3+.15*rng.standard_normal(2000),.15*rng.standard_normal(2000)]),np.column_stack([.15*rng.standard_normal(2000),2+.15*rng.standard_normal(2000)])]);r=np.sin(np.pi/36);v=np.array([r,-1.]);entry=3.4;H=.5

def init(U,W):
 Z=X@U.T;return np.maximum(Z,0),Z>0

def apply(U,W,act,G,d):
 du=d[:2*h].reshape(h,2);dw=d[2*h:].reshape(h,2)
 return act@dw+(G*(X@du.T))@W

def adj(U,W,act,G,f):
 return np.concatenate([((G*(f@W.T)).T@X/N).reshape(-1),(act.T@f/N).reshape(-1)])

def grad(U,W):
 M=W[I].T@U[I];ell=np.array([M[0]@v-r,1]);g=np.zeros((2*h,2));g[:h][I]=(W[I]@ell)[:,None]*v;g[h:][I]=(U[I]@v)[:,None]*ell;return g.reshape(-1)

def norm(f):return np.sqrt(np.sum(f*f)/N)

ia=np.argmin(abs(n['time']-entry));Ua=n['U'][ia];Wa=n['W'][ia];acta,Ga=init(Ua,Wa);ea=X-acta@Wa;ga=grad(Ua,Wa)
J=np.zeros((N,2,2*h,2))
for k in range(2):
 J[:,k,:h,:]=Ga[:,:,None]*Wa[None,:,k,None]*X[:,None,:];J[:,k,h:,k]=acta
J=J.reshape(2*N,4*h)/np.sqrt(N);ev,V=np.linalg.eigh(J.T@J);ba=adj(Ua,Wa,acta,Ga,ea);psi=float((ga@V)*np.exp(-ev*H)@(V.T@ba))
records=[]
for i in range(ia,ia+51):
 U=n['U'][i];W=n['W'][i];t=float(n['time'][i]-entry);act,G=init(U,W);e=X-act@W;bt=adj(U,W,act,G,e)
 dualcoeff=V@(np.exp(-ev*(H-t))*(V.T@ga));z=apply(Ua,Wa,acta,Ga,dualcoeff)
 Djz=adj(U,W,act,G,z)-adj(Ua,Wa,acta,Ga,z);Dje=bt-adj(Ua,Wa,acta,Ga,e);jaz=adj(Ua,Wa,acta,Ga,z)
 records.append(dict(t=t,first=float(Djz@bt),second=float(jaz@Dje),bound1=float(np.linalg.norm(Djz)*np.linalg.norm(bt)),bound2=float(np.linalg.norm(jaz)*np.linalg.norm(Dje)),Djz2=float(Djz@Djz),speed2=float(bt@bt)))
end1=float((grad(U,W)-ga)@bt);end2=float(ga@Dje);tt=np.array([a['t'] for a in records]);integ=lambda key:float(np.trapz([a[key] for a in records],tt));actualq=float(grad(U,W)@bt);error=actualq-psi
result=dict(entry=entry,horizon=H,psi=psi,actual_Q_at_endpoint=actualq,actual_minus_reference=error,endpoint_gradient_drift=end1,endpoint_J_drift=end2,integral1_trapezoid=integ('first'),integral2_trapezoid=integ('second'),signed_reconstruction=end1+end2-integ('first')-integ('second'),quadrature_and_discrete_trajectory_mismatch=error-(end1+end2-integ('first')-integ('second')),absolute_pointwise_integrated_budget=abs(end1)+abs(end2)+integ('bound1')+integ('bound2'),energy_Cauchy_budget=abs(end1)+abs(end2)+np.sqrt(integ('Djz2')*integ('speed2'))+integ('bound2'),late_positive_margin_after_pointwise_budget=psi-(abs(end1)+abs(end2)+integ('bound1')+integ('bound2')),status='Retrospective sampled diagnostics only. Trapezoidal sums on Euler states are not analytic upper bounds or validated quadrature.')
(root/'outputs/crossing-closure/DIRECTIONAL_BUDGET_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
