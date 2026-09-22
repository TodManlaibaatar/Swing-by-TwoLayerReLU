"""Outward static witness enclosures. Does NOT certify initialized reachability.
Run: python3 code/certify_witness_balls.py --data canonical_witness_states.npz
The exact mathematical dataset is the binary-float dataset in --data.
"""
from interval_bounds import *
import argparse,platform
parser=argparse.ArgumentParser()
parser.add_argument('--data',type=Path,default=Path('outputs/crossing-closure/canonical_witness_states.npz'))
parser.add_argument('--output',type=Path,default=None)
args=parser.parse_args();data=np.load(args.data);XX=data['X'];strong=data['strong_mask'];N=len(XX);h=data['theta'].shape[1]//2
X=I(XX);invN=I(1/N,up(abs(1/N)*UROUND));cov=(X.T@X)*invN
lam=scal(np.max(sump(cov.absup(),axis=1)));sqrtlam=up(np.sqrt(lam));xn=normup(X,axis=1)
sqrtNlo=np.nextafter(np.sqrt(float(N)),-np.inf)
mp.iv.dps=60
def trigI(v):
 lo=np.nextafter(float(v.a),-np.inf);hi=np.nextafter(float(v.b),np.inf)
 c=(lo+hi)/2;r=up(max(c-lo,hi-c));return I(c,r)
r=trigI(mp.iv.sin(mp.iv.pi/36));k=trigI(mp.iv.cos(mp.iv.pi/36));xi=stack([r,-k]);R=up(3.5e-5);results=[];geometry=[]
for ti,time in enumerate(data['times']):
 theta=I(data['theta'][ti]);U=theta[:h];W=theta[h:];Z=X@U.T
 assert np.all((Z.lo()>0)|(Z.hi()<0)), 'Training mask unresolved'
 G=(Z.lo()>0).astype(float);act=Z*G;e=X-act@W;RW=e@W.T
 er=normup(e,axis=1);wn=normup(W,axis=1);P=normup(theta)
 D=positive_sum(mulp(P,R),mulp(.5,mulp(R,R)));Pb=addp(P,R)
 possible=Z.abslow()<=mulp(R,xn[:,None]);da=np.maximum(0,up(mulp(R,xn[:,None])-Z.abslow()))
 rw=positive_sum(RW.absup(),mulp(R,er[:,None]),mulp(mulp(D,xn[:,None]),addp(wn,R)[None,:]))
 # Geometry: angle bounds via cone inequalities, avoiding inverse trig.
 PI=up(np.sqrt(addp(sump(mulp(U[strong].absup(),U[strong].absup())),sump(mulp(W[strong].absup(),W[strong].absup())))))
 mass=(U[strong,0]*W[strong,0]).sum();masserr=addp(mulp(PI,R),mulp(.5,mulp(R,R)))
 masslo=np.nextafter(mass.lo()-masserr,-np.inf)
 coneflags={}
 for name,VV,degrees in [('input',U,15),('output',W,20)]:
  tanlo=scal(trigI(mp.iv.tan(mp.iv.pi*degrees/180)).lo())
  xlo=np.nextafter(VV[strong,0].lo()-R,-np.inf)
  yup=addp(VV[strong,1].absup(),R)
  coneflags[name]=bool(np.all(xlo>0) and np.all(yup<np.nextafter(xlo*tanlo,-np.inf)))
  assert coneflags[name]
 # This speed bound is valid on the whole ball for the actual ReLU field.
 en=up(normup(e)/sqrtNlo);speed=mulp(mulp(sqrtlam,Pb),addp(en,mulp(sqrtlam,D)))
 geometry.append(dict(time=float(time),strong_count=int(sum(strong)),m11_ball_lower=scal(masslo),input_cone_15deg=coneflags['input'],output_cone_20deg=coneflags['output'],actual_field_norm_upper=scal(speed),parameter_norm_ball_upper=scal(Pb)))
 for cluster,sl in [('all',slice(None)),('strong',slice(0,N//2)),('weak',slice(N//2,N))]:
  # All cluster contributions retain 1/N normalization and the FULL residual.
  Xc=X[sl];ec=e[sl];ac=act[sl];Gc=G[sl]
  field=stack([((RW[sl]*Gc).T@Xc)*invN,(ac.T@ec)*invN],axis=0)
  field=I(field.c.reshape(2*h,2),field.r.reshape(2*h,2))
  covc=(Xc.T@Xc)*invN;lc=scal(np.max(sump(covc.absup(),axis=1)));slc=up(np.sqrt(lc))
  enc=up(normup(ec)/sqrtNlo);ebc=addp(enc,mulp(slc,D))
  L=addp(mulp(lc,mulp(Pb,Pb)),mulp(slc,ebc))
  jumpu=mulp(sump(mulp(mulp(possible[sl],rw[sl]),xn[sl,None]),axis=0),up(1/N))
  jumpw=mulp(sump(mulp(da[sl],addp(er[sl],mulp(D,xn[sl]))[:,None]),axis=0),up(1/N))
  fchange=dotp(da[sl],addp(wn,R))
  gateF=addp(up(np.sqrt(addp(sump(mulp(jumpu,jumpu)),sump(mulp(jumpw,jumpw))))),mulp(mulp(slc,Pb),up(normup(fchange)/sqrtNlo)))
  for kind in ['Q_G','full_rate']:
   if kind=='Q_G':
    mask=strong;v=stack([r,I(-1.)]);M=W[mask].T@U[mask];ell=stack([(M[0]@v)-r,I(1.)]);probe_margin=None
   else:
    scores=U@xi;assert np.all((scores.lo()>0)|(scores.hi()<0));mask=scores.lo()>0;v=xi;M=W[mask].T@U[mask];ell=M@v-xi
    probe_margin=scal(np.min(scores.abslow()));assert probe_margin>mulp(R,normup(v))
   V=normup(v);PI=up(np.sqrt(addp(sump(mulp(U[mask].absup(),U[mask].absup())),sump(mulp(W[mask].absup(),W[mask].absup())))))
   Dm=addp(mulp(PI,R),mulp(.5,mulp(R,R)));ellb=addp(normup(ell),mulp(V,Dm))
   Hg=addp(mulp(mulp(V,V),mulp(addp(PI,R),addp(PI,R))),mulp(V,ellb))
   gu=(W[mask]@ell)[:,None]*v;gw=(U[mask]@v)[:,None]*ell
   gc=np.zeros((2*h,2));gr=np.zeros_like(gc);gc[:h][mask]=gu.c;gr[:h][mask]=gu.r;gc[h:][mask]=gw.c;gr[h:][mask]=gw.r;g=I(gc,gr);gn=normup(g);q=(g*field).sum()
   smooth=mulp(addp(mulp(Hg,addp(normup(field),mulp(L,R))),mulp(gn,L)),R)
   gate=mulp(addp(gn,mulp(Hg,R)),gateF);budget=addp(smooth,gate)
   lo=np.nextafter(q.lo()-budget,-np.inf);hi=np.nextafter(q.hi()+budget,np.inf)
   if cluster=='all':assert hi<0 if time<3.5 else lo>0
   elif cluster=='strong':assert hi<0
   else:assert lo>0
   results.append(dict(time=float(time),cluster=cluster,quantity=kind,center_lower=scal(q.lo()),center_upper=scal(q.hi()),smooth_budget_upper=scal(smooth),gate_budget_upper=scal(gate),ball_lower=scal(lo),ball_upper=scal(hi),probe_margin_lower=probe_margin))
result=dict(status='STATIC WITNESS BALLS ENCLOSED; INITIALIZED TRAJECTORY REACHABILITY NOT CERTIFIED',arithmetic='Float64 midpoint-radius intervals; dot gamma_k bounds; outward nextafter sums/products; interval trigonometry at 60 decimal digits. Assumes standard IEEE round-to-nearest basic arithmetic, sqrt, and dot accumulation without unsafe fast-math.',static_radius='0.000035',radius_used_upper_float=float(R),results=results,geometry=geometry,inputs={args.data.name:hashlib.sha256(args.data.read_bytes()).hexdigest()},versions=dict(python=platform.python_version(),numpy=np.__version__,mpmath=mp.__version__),not_claimed=['A true initialized flow reaches either ball','A gradient-flow crossing has been certified','High-probability iid initialization guarantee'])
output=args.output or args.data.parent/'STATIC_WITNESS_CERTIFICATE.json';output.write_text(json.dumps(result,indent=2)+'\n')
for q in results:print(q['time'],q['cluster'],q['quantity'],q['ball_lower'],q['ball_upper'])
print(json.dumps(geometry,indent=2))
