from pathlib import Path
import numpy as np,json
root=Path.cwd();n=np.load(root/'work/appendix-repository/swing_grid_preregistered/preregistration_v1_postspecialization/cells/mu2_2p000__sigma_0p150/training_state.npz');I=n['strong_mask'].astype(bool);rng=np.random.default_rng(0);N=4000;h=200
X=np.vstack([np.column_stack([3+.15*rng.standard_normal(2000),.15*rng.standard_normal(2000)]),np.column_stack([.15*rng.standard_normal(2000),2+.15*rng.standard_normal(2000)])]);xn=np.linalg.norm(X,axis=1)
j=np.argmin(abs(n['time']-3.4));U=n['U'][j];W=n['W'][j];Z=X@U.T;A=np.maximum(Z,0);G=Z>0;F=A@W;e=(X-F).reshape(-1)/np.sqrt(N)
J=np.zeros((N,2,2*h,2))
for k in range(2):
 J[:,k,:h,:]=G[:,:,None]*W[None,:,k,None]*X[:,None,:];J[:,k,h:,k]=A
J=J.reshape(2*N,4*h)/np.sqrt(N)
r=np.sin(np.pi/36);v=np.array([r,-1]);M=W[I].T@U[I];ell=np.array([M[0]@v-r,1]);g=np.zeros((2*h,2));g[:h][I]=(W[I]@ell)[:,None]*v;g[h:][I]=(U[I]@v)[:,None]*ell;g=g.reshape(-1)
H=.5;loss=.5*e@e;R=np.sqrt(H*loss);P=np.sqrt(np.sum(U*U)+np.sum(W*W));Xmax=xn.max();D_M=P*R+R*R/2;Dg=np.linalg.norm(v)*np.linalg.norm(ell)*R+np.linalg.norm(v)**2*D_M*(P+R)
boundary=abs(Z)<=R*xn[:,None];Gamma=np.sqrt(np.mean(np.sum(boundary*(np.sum(W*W,axis=1)[None,:]),axis=1)*xn*xn));D_J=np.sqrt(2)*Xmax*R+Gamma
jn=np.linalg.norm(J,ord=2);cn=np.linalg.norm(J@g);D_C=D_J*np.linalg.norm(g)+(jn+D_J)*Dg;D_K=(2*jn+D_J)*D_J;rho=np.linalg.norm(e)*(D_C+cn*H*D_K)
z=np.load(root/'outputs/crossing-closure/frozen_spectrum_3.4.npz');psi=float(z['coefficients']@np.exp(-z['rates']*H))
result=dict(entry=3.4,horizon=H,loss=float(loss),parameter_displacement_bound=float(R),J_norm=float(jn),parameter_norm=float(P),boundary_term=float(Gamma),J_drift_bound=float(D_J),g_drift_bound=float(Dg),rho_bound=float(rho),frozen_positive_witness=psi,required_inequality='rho_bound < frozen_positive_witness',passed=bool(rho<psi),interpretation='Diagnostic substitution of saved Euler state into conservative loss/margin bound; not a validated entry-state certificate or proof of impossibility.')
(root/'outputs/crossing-closure/LOCAL_BUDGET_DIAGNOSTIC.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
