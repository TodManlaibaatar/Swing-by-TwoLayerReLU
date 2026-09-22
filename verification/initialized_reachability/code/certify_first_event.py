"""Nonlinear enclosure of the first training-gate event from archived theta(0).
No numerical trajectory or sampled event list is used as a premise.
"""
from interval_bounds import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('inputs/canonical_witness_states.npz'));ap.add_argument('--output',type=Path,default=Path('FIRST_EVENT_CERTIFICATE.json'));args=ap.parse_args();data=np.load(args.data);X=I(data['X']);theta=I(data['initial_theta']);N=4000;h=200;U=theta[:h];W=theta[h:];q=X@U.T
assert np.all((q.lo()>0)|(q.hi()<0));G=(q.lo()>0).astype(float);act=q*G;invN=I(1/N,up(UROUND/N));e=X-act@W

def vst(a,b):return I(np.vstack([a.c,b.c]),np.vstack([a.r,b.r]))
def adj(Y):return vst(((Y@W.T)*G).T@X*invN,act.T@Y*invN)
F=adj(e);du,dw=F[:h],F[h:];Jv=act@dw+((X@du.T)*G)@W
# Residual Hessian action is computed directly without a sampled derivative.
HF=vst(((e@dw.T)*G).T@X*invN,((X@du.T)*G).T@e*invN)
acc=HF-adj(Jv);dq=X@du.T;ddq=X@acc[:h].T
cov=X.T@X*invN;lam=float(np.max(sump(cov.absup(),axis=1)));sl=up(np.sqrt(lam));P=normup(theta);snlo=np.nextafter(np.sqrt(N),-np.inf);en=up(normup(e)/snlo);xn=normup(X,axis=1)
R=up(1e-6);Pb=addp(P,R);D=addp(mulp(P,R),mulp(.5,mulp(R,R)));eb=addp(en,mulp(sl,D));speed=mulp(mulp(sl,Pb),eb);L=addp(mulp(lam,mulp(Pb,Pb)),mulp(sl,eb));third=addp(mulp(mulp(mulp(3.,lam),Pb),mulp(speed,speed)),mulp(mulp(L,L),speed))
n,i=3971,36
# These times are proposed rational endpoints, checked with intervals below.
a=I(35888/10**10,up(UROUND*35888/10**10));b=I(35891/10**10,up(UROUND*35891/10**10));H=float(b.hi());assert mulp(speed,H)<R

def score(t):
 T=float(t.hi());err=mulp(mulp(mulp(mulp(third,T),T),T),up(1/6));v=q[n,i]+dq[n,i]*t+ddq[n,i]*t*t*.5
 rad=mulp(xn[n],err);return I(v.c,addp(v.r,rad))
qa=score(a);qb=score(b);assert qa.hi()<0 and qb.lo()>0
# A whole-interval second-order envelope excludes every competing event.
signs=2*G-1;oriented=q*signs;oriented_v=dq*signs
rem=mulp(.5,mulp(mulp(L,speed),mulp(H,H)))
drift_lower=np.nextafter(np.minimum(0,oriented_v.lo())*H,-np.inf)
other_lower=np.nextafter(oriented.lo()+drift_lower,-np.inf)-mulp(xn[:,None],rem)
other_lower=np.nextafter(other_lower,-np.inf);other_lower[n,i]=np.inf;assert np.min(other_lower)>0
slope_lo=np.nextafter(dq[n,i].lo()-mulp(xn[n],mulp(mulp(L,speed),H)),-np.inf);assert slope_lo>0
# Bound the jump in normal velocity uniformly over the same ball.
x=X[n];w0=W[i];e0=e[n];we=(w0*e0).sum();er=normup(e0);wn=normup(w0)
we_err=addp(mulp(R,er),mulp(addp(wn,R),mulp(D,xn[n])))
weI=I(we.c,addp(we.r,we_err));xnorm2=(x*x).sum();jump=weI*xnorm2*invN
postlo=np.nextafter(slope_lo+min(0,float(jump.lo())),-np.inf);assert postlo>0
result=dict(status='FIRST TRAINING-GATE EVENT CERTIFIED FROM EXACT ARCHIVED INITIAL PARAMETERS',indexing='zero based',sample=n,neuron=i,transition='inactive to active',time_bracket_rationals=['35888/10^10','35891/10^10'],q_at_lower_upper=float(qa.hi()),q_at_upper_lower=float(qb.lo()),other_gates_oriented_margin_lower=float(np.min(other_lower)),pre_event_normal_velocity_lower=float(slope_lo),post_event_normal_velocity_lower=float(postlo),smooth_reference_parameter_radius=float(R),whole_interval_speed_upper=float(speed),polynomial_third_derivative_upper=float(third),input_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(),scope='A nonlinear event existence, isolation and two-sided transversality certificate. It does not continue the full trajectory to time 3.4.')
args.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
