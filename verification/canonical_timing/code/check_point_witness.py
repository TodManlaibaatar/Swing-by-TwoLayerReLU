from pathlib import Path
import sys,json,importlib.util,numpy as np
p=Path(__file__).resolve().parent.parent;sys.argv=['static-check']
spec=importlib.util.spec_from_file_location('timing',p/'code/certify_full_tube.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
R=m.up(2.6457445093e-5);result=m.bounds(m.I(m.T[1200]),R);r0=float(m.ledger[1199]['radius_upper']);remaining=np.nextafter(R-r0,-np.inf);travel=m.mulp(result['speed_upper'],m.up(6e-6));assert travel<remaining
assert result['G_lower']>0 and result['E_upper']<0 and result['Xi_upper']<0
scores=m.I(m.T[1200,:200])@m.xi;assert np.min(scores.abslow())>m.mulp(R,m.vnxi)
result.update(radius=float(R),flow_radius_at_3p64=r0,extension_rational='6/1000000',remaining_radius_lower=float(remaining),travel_upper=float(travel),interval=['3.64','3.640006'],status='CONDITIONAL INTERVAL WITNESS VERIFIED')
(p/'POINT_TIMING_WITNESS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
