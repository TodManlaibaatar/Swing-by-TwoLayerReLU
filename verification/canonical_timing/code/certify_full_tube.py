"""Full-tube static timing audit. Reuses, and does not alter, the flow certificate.
Every bound covers all compatible training selectors. Probe gates are certified
on the reference tube (not assumed from a sampled trajectory).
"""
from interval_bounds import *
import argparse,csv,time,multiprocessing,platform
ap=argparse.ArgumentParser();ap.add_argument('--inputs',type=Path,default=Path(__file__).resolve().parent.parent/'prior');ap.add_argument('--output',type=Path,default=Path(__file__).resolve().parent.parent);ap.add_argument('--workers',type=int,default=3);ap.add_argument('--start',type=int,default=0);ap.add_argument('--stop',type=int,default=2500);args=ap.parse_args()
p=args.inputs;z=np.load(p/'inputs/canonical_late_reference.npz');w=np.load(p/'inputs/canonical_witness_states.npz');X=I(z['X']);T=z['theta'];V=z['velocity'];strong=w['strong_mask'];N=len(X.c);h=T.shape[1]//2
assert np.array_equal(z['X'],w['X']);assert int(sum(strong))==71
with (p/'WIDE_LOCAL_FLOW_ENCLOSURE.csv').open() as f:ledger=list(csv.DictReader(f))
assert len(ledger)==2500
mp.iv.dps=60
invN=I(1/N,up(UROUND/N));dt=I(1/5000,up(UROUND/5000));halfup=float((dt*.5).hi());invdt=I(5000.)
xn=normup(X,axis=1);cov=(X.T@X)*invN;lam=float(np.max(sump(cov.absup(),axis=1)));sl=up(np.sqrt(lam));sqrtNlo=np.nextafter(np.sqrt(N),-np.inf)
def trig(v):
 lo=np.nextafter(float(v.a),-np.inf);hi=up(float(v.b));c=(lo+hi)/2;return I(c,up(max(c-lo,hi-c)))
r=trig(mp.iv.sin(mp.iv.pi/36));k=trig(mp.iv.cos(mp.iv.pi/36));xi=stack([r,-k]);vg=stack([r,I(-1.)]);width=up(1e-6);vnxi=normup(xi)
def bounds(theta,R):
 U=theta[:h];W=theta[h:];Z=X@U.T
 if not np.all((Z.lo()>0)|(Z.hi()<0)):return None
 G=(Z.lo()>0).astype(float);act=Z*G;e=X-act@W;RW=e@W.T
 field=stack([((RW*G).T@X)*invN,(act.T@e)*invN]);F=I(field.c.reshape(2*h,2),field.r.reshape(2*h,2));Fn=normup(F)
 er=normup(e,axis=1);wn=normup(W,axis=1);P=normup(theta);Pb=addp(P,R);D=addp(mulp(P,R),mulp(.5,mulp(R,R)));en=up(normup(e)/sqrtNlo)
 L=addp(mulp(lam,mulp(Pb,Pb)),mulp(sl,addp(en,mulp(sl,D))))
 possible=Z.abslow()<=mulp(R,xn[:,None]);da=np.maximum(0,up(mulp(R,xn[:,None])-Z.abslow()))
 rw=positive_sum(RW.absup(),mulp(R,er[:,None]),mulp(mulp(D,xn[:,None]),addp(wn,R)[None,:]))
 ju=mulp(sump(mulp(mulp(possible,rw),xn[:,None]),axis=0),up(1/N));jw=mulp(sump(mulp(da,addp(er,mulp(D,xn))[:,None]),axis=0),up(1/N))
 fchange=dotp(da,addp(wn,R));gate=addp(up(np.sqrt(addp(sump(mulp(ju,ju)),sump(mulp(jw,jw))))),mulp(mulp(sl,Pb),up(normup(fchange)/sqrtNlo)))
 speed=mulp(mulp(sl,Pb),addp(en,mulp(sl,D)))
 def grad(mask,v,kind):
  M=W[mask].T@U[mask];ell=stack([M[0]@v-r,I(1.)]) if kind=='G' else M@v-xi
  PI=up(np.sqrt(addp(sump(mulp(U[mask].absup(),U[mask].absup())),sump(mulp(W[mask].absup(),W[mask].absup())))))
  vn=normup(v);Dm=addp(mulp(PI,R),mulp(.5,mulp(R,R)));Hg=addp(mulp(mulp(vn,vn),mulp(addp(PI,R),addp(PI,R))),mulp(vn,addp(normup(ell),mulp(vn,Dm))))
  gu=(W[mask]@ell)[:,None]*v;gw=(U[mask]@v)[:,None]*ell;gc=np.zeros((2*h,2));gr=np.zeros_like(gc);gc[:h][mask]=gu.c;gr[:h][mask]=gu.r;gc[h:][mask]=gw.c;gr[h:][mask]=gw.r
  return I(gc,gr),Hg
 scores=U@xi
 if not np.all((scores.lo()>0)|(scores.hi()<0)):return None
 active=scores.lo()>0;gG,HG=grad(strong,vg,'G');gE,HE=grad(active,xi,'E')
 result=dict(training_ambiguous_pairs=int(possible.sum()),training_gate_assignment_status='UNRESOLVED_BUT_ALL_SELECTORS_ENCLOSED' if np.any(possible) else 'RESOLVED',speed_upper=float(speed),parameter_norm_upper=float(Pb),field_lipschitz_upper=float(L),training_gate_correction_upper=float(gate))
 for name,g,Hg in [('G',gG,HG),('E',gE,HE),('Xi',gE-gG,addp(HE,HG))]:
  gn=normup(g);q=(g*F).sum();smooth=mulp(addp(mulp(Hg,addp(Fn,mulp(L,R))),mulp(gn,L)),R);jump=mulp(addp(gn,mulp(Hg,R)),gate);budget=addp(smooth,jump)
  result[name+'_center_lower']=float(q.lo());result[name+'_center_upper']=float(q.hi());result[name+'_budget_upper']=float(budget);result[name+'_lower']=float(np.nextafter(q.lo()-budget,-np.inf));result[name+'_upper']=float(up(q.hi()+budget))
 # Fixed probe pattern on the tube suffices; the extension is polynomial on the larger ball.
 angular=mulp(mulp(mulp(mulp(2.,addp(1.,mulp(.5,mulp(Pb,Pb)))),Pb),speed),width)
 result['sector_rate_cost_upper']=float(angular)
 for name in ['E','Xi']:
  result[name+'_sector_lower']=float(np.nextafter(result[name+'_lower']-angular,-np.inf));result[name+'_sector_upper']=float(up(result[name+'_upper']+angular))
 return result

def step(j):
 row=ledger[j];assert int(row['step'])==17001+j
 Rbar=float(row['radius_upper']);assert Rbar<float(row['tube_radius_upper']);a0=I(T[j]);a1=I(T[j+1]);f0=I(V[j]);f1=I(V[j+1]);y=(a0+a1)*.5+(f0-f1)*dt*.125
 vel=(a1-a0)*1.5*invdt-(f0+f1)*.25;acc=(f1-f0)*invdt;jerk=(a0-a1)*12.*(invdt*invdt*invdt)+(f0+f1)*6.*(invdt*invdt)
 h2=mulp(halfup,halfup);h3=mulp(h2,halfup)
 motion=positive_sum(mulp(halfup,normup(vel)),mulp(.5,mulp(h2,normup(acc))),mulp(up(1/6),mulp(h3,normup(jerk))))
 rounding=normup(I(np.zeros_like(y.c),y.r));R=positive_sum(Rbar,motion,rounding)
 theta=I(y.c)
 # Bernstein bounds certify probe signs for the entire reference segment.
 b0=a0[:h]@xi;b3=a1[:h]@xi;third=I(1/3,up(UROUND/3));b1=b0+(f0[:h]@xi)*dt*third;b2=b3-(f1[:h]@xi)*dt*third
 bs=stack([b0,b1,b2,b3]);lo=np.min(bs.lo(),axis=0);hi=np.max(bs.hi(),axis=0);margin=np.maximum(lo,-hi)
 # Input norm along the cubic is bounded by endpoint Bernstein control vectors.
 Ucontrol=stack([a0[:h],a0[:h]+f0[:h]*dt*third,a1[:h]-f1[:h]*dt*third,a1[:h]])
 un=np.max(normup(Ucontrol,axis=2),axis=0)
 centralmargin=np.nextafter(margin-mulp(Rbar,vnxi),-np.inf)
 sectorcost=mulp(addp(un,Rbar),width);sectormargin=np.nextafter(centralmargin-sectorcost,-np.inf)
 out=dict(step=17001+j,t_left=(17000+j)/5000,t_right=(17001+j)/5000,flow_tube_radius=Rbar,prescribed_guard_radius=float(row['tube_radius_upper']),reference_motion_upper=float(motion),midpoint_rounding_upper=float(rounding),static_radius=float(R),probe_central_margin_lower=float(np.min(centralmargin)),probe_sector_margin_lower=float(np.min(sectormargin)))
 result=bounds(theta,R)
 out['status']='ENCLOSED' if result is not None and np.all(sectormargin>0) else 'UNRESOLVED'
 if result is not None:out.update(result)
 return out
if __name__=='__main__':
 args.output.mkdir(parents=True,exist_ok=True);started=time.monotonic();rows=[]
 path=args.output/('TIMING_FULL_TUBE_CERTIFICATE.csv' if args.start==0 and args.stop==2500 else f'timing_{args.start}_{args.stop}.csv')
 with path.open('w') as f:
  with multiprocessing.get_context('spawn').Pool(args.workers) as pool:
   for row in pool.imap(step,range(args.start,args.stop),chunksize=4):
    if not rows:writer=csv.DictWriter(f,fieldnames=row);writer.writeheader()
    rows.append(row);writer.writerow(row)
    if len(rows)%100==0:f.flush();print('steps',len(rows),'elapsed',round(time.monotonic()-started,2),flush=True)
 report=dict(status='ENCLOSURES COMPUTED; SIGN BRACKETS REQUIRE SEPARATE REPLAY',count=len(rows),unresolved_steps=[q['step'] for q in rows if q['status']!='ENCLOSED'],seconds=time.monotonic()-started,inputs={str(q.relative_to(p)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [p/'inputs/canonical_late_reference.npz',p/'inputs/canonical_witness_states.npz',p/'WIDE_LOCAL_FLOW_ENCLOSURE.csv',p/'WIDE_LOCAL_FLOW_CERTIFICATE.json']},versions=dict(python=platform.python_version(),numpy=np.__version__,mpmath=mp.__version__))
 (args.output/('TIMING_ENCLOSURE_RUN.json' if args.start==0 and args.stop==2500 else f'TIMING_RUN_{args.start}_{args.stop}.json')).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
