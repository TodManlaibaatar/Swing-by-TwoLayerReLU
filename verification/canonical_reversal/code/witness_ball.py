from pathlib import Path
import numpy as np,json
root=Path.cwd();ref=root/'work/crossing-closure/canonical-reference';X=np.load(ref/'X.npy');th=np.load(ref/'theta.npy',mmap_mode='r');fv=np.load(ref/'velocity.npy',mmap_mode='r');ar=np.load(root/'work/appendix-repository/swing_grid_preregistered/preregistration_v1_postspecialization/cells/mu2_2p000__sigma_0p150/training_state.npz');strong=ar['strong_mask'].astype(bool);N=4000;h=200;xn=np.linalg.norm(X,axis=1);lam=np.linalg.eigvalsh(X.T@X/N)[-1];R=3e-5;r=np.sin(np.pi/36);xi=np.array([r,-np.sqrt(1-r*r)]);out=[]
for t in [3.4,3.9]:
 j=round(t/.0002);theta=th[j];field=fv[j];U=theta[:h];W=theta[h:];Z=X@U.T;G=Z>0;act=np.maximum(Z,0);e=X-act@W;en=np.linalg.norm(e)/np.sqrt(N);er=np.linalg.norm(e,axis=1);wn=np.linalg.norm(W,axis=1);P=np.linalg.norm(theta);D=P*R+.5*R*R;Pb=P+R;eb=en+np.sqrt(lam)*D;L=lam*Pb*Pb+np.sqrt(lam)*eb
 possible=abs(Z)<=R*xn[:,None];da=np.maximum(0,R*xn[:,None]-abs(Z));rwbound=abs(e@W.T)+R*er[:,None]+D*xn[:,None]*(wn+R)[None,:]
 jumpu=np.mean(possible*rwbound*xn[:,None],axis=0);jumpw=np.mean(da*(er+D*xn)[:,None],axis=0);fchange=da@(wn+R);gateF=np.sqrt(jumpu@jumpu+jumpw@jumpw)+np.sqrt(lam)*Pb*np.linalg.norm(fchange)/np.sqrt(N)
 for kind in ['Q_G','full_rate']:
  if kind=='Q_G':
   I=strong;v=np.array([r,-1.]);M=W[I].T@U[I];ell=np.array([M[0]@v-r,1.]);probe_margin=None
  else:
   scores=U@xi;I=scores>0;v=xi;M=W[I].T@U[I];ell=M@v-xi;probe_margin=float(min(abs(scores)));assert probe_margin>R
  V=np.linalg.norm(v);PI=np.sqrt(np.sum(U[I]**2)+np.sum(W[I]**2));Dm=PI*R+.5*R*R;ellb=np.linalg.norm(ell)+V*Dm;Hg=V*V*(PI+R)**2+V*ellb
  g=np.zeros_like(theta);g[:h][I]=(W[I]@ell)[:,None]*v;g[h:][I]=(U[I]@v)[:,None]*ell;gn=np.linalg.norm(g);q=float(np.sum(g*field));smooth=(Hg*(np.linalg.norm(field)+L*R)+gn*L)*R;gate=(gn+Hg*R)*gateF;budget=float(smooth+gate)
  out.append(dict(time=t,kind=kind,radius=R,q_center=q,smooth_budget=float(smooth),training_gate_budget=float(gate),total_budget=budget,lower=q-budget,upper=q+budget,probe_margin=probe_margin))
result=dict(status='Floating-point evaluation of conservative static ball inequalities; not yet directed-rounded or linked to initialized reachability.',witnesses=out)
(root/'outputs/crossing-closure/WITNESS_BALL_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
