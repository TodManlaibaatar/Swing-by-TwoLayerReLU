"""Outward OOD-error enclosures for three static parameter balls."""
from interval_bounds import *
import argparse
ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('inputs/canonical_snapshot_states.npz'));ap.add_argument('--output',type=Path,default=Path('WIDE_SNAPSHOT_CERTIFICATE.json'));a=ap.parse_args();z=np.load(a.data);mp.iv.dps=60

def mi(z):
 lo=np.nextafter(float(z.a),-np.inf);hi=up(float(z.b));c=(lo+hi)/2;return I(c,up(max(c-lo,hi-c)))
xi=stack([mi(mp.iv.sin(mp.iv.pi/36)),-mi(mp.iv.cos(mp.iv.pi/36))]);V=normup(xi);R=up(3.5e-5);rows=[]
for t,th in zip(z['times'],z['theta']):
 R=up(2e-5 if t<3.5 else (3.5e-5 if t<3.8 else 4.6e-5))
 U=I(th[:200]);W=I(th[200:]);q=U@xi;assert np.all((q.lo()>0)|(q.hi()<0));assert min(q.abslow())>mulp(R,V)
 active=q.lo()>0;M=W[active].T@U[active];e=M@xi-xi;E=(e*e).sum()*.5
 gu=(W[active]@e)[:,None]*xi;gw=(U[active]@xi)[:,None]*e
 G=up(np.sqrt(addp(sump(mulp(gu.absup(),gu.absup())),sump(mulp(gw.absup(),gw.absup())))))
 PI=up(np.sqrt(addp(sump(mulp(U[active].absup(),U[active].absup())),sump(mulp(W[active].absup(),W[active].absup())))))
 D=addp(mulp(PI,R),mulp(.5,mulp(R,R)));Hb=addp(mulp(mulp(V,V),mulp(addp(PI,R),addp(PI,R))),mulp(V,addp(normup(e),mulp(V,D))))
 err=addp(mulp(G,R),mulp(.5,mulp(Hb,mulp(R,R))));lo=np.nextafter(E.lo()-err,-np.inf);hi=up(E.hi()+err)
 P=addp(normup(I(th)),R);L=mulp(.5,mulp(P,P));K=mulp(addp(1,L),addp(1,L))
 rows.append(dict(time=float(t),radius_upper=float(R),E_center_lower=scal(E.lo()),E_center_upper=scal(E.hi()),E_ball_lower=scal(lo),E_ball_upper=scal(hi),parameter_norm_ball_upper=scal(P),probe_error_Lipschitz_upper=scal(K)))
gap1=np.nextafter(rows[0]['E_ball_lower']-rows[1]['E_ball_upper'],-np.inf);gap2=np.nextafter(rows[2]['E_ball_lower']-rows[1]['E_ball_upper'],-np.inf)
width=1e-6;K=max(q['probe_error_Lipschitz_upper'] for q in rows);transport=mulp(mulp(2.,K),up(width));a1=np.nextafter(gap1-transport,-np.inf);a2=np.nextafter(gap2-transport,-np.inf)
assert a1>1.2e-4 and a2>4e-5
result=dict(status='STATIC SNAPSHOT BALLS ENCLOSED; TRAJECTORY REACHABILITY SEPARATE',radius_by_time={'3.4':'0.00002','3.7':'0.000035','3.9':'0.000046'},snapshots=rows,central_first_gap_lower=float(gap1),central_rebound_lower=float(gap2),sector_angular_halfwidth='0.000001',sector_first_gap_lower=float(a1),sector_rebound_lower=float(a2),input_sha256=hashlib.sha256(a.data.read_bytes()).hexdigest())
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
