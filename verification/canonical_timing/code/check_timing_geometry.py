from pathlib import Path
import sys,csv,json,importlib.util,numpy as np
p=Path(__file__).resolve().parent.parent;sys.argv=['geometry-audit'];sp=importlib.util.spec_from_file_location('timing',p/'code/certify_full_tube.py');m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
rows=list(csv.DictReader((p/'TIMING_FULL_TUBE_CERTIFICATE.csv').open()));third=m.I(1/3,m.up(m.UROUND/3));min_margin=float('inf')
for j,row in enumerate(rows):
 a0=m.I(m.T[j]);a1=m.I(m.T[j+1]);f0=m.I(m.V[j]);f1=m.I(m.V[j+1]);y=(a0+a1)*.5+(f0-f1)*m.dt*.125
 b0=a0[:m.h]@m.xi;b3=a1[:m.h]@m.xi;bs=m.stack([b0,b0+(f0[:m.h]@m.xi)*m.dt*third,b3-(f1[:m.h]@m.xi)*m.dt*third,b3]);lo=np.min(bs.lo(),axis=0);hi=np.max(bs.hi(),axis=0)
 mid=m.I(y.c[:m.h])@m.xi
 assert np.all((mid.lo()>0)|(mid.hi()<0));assert np.array_equal(mid.lo()>0,lo>0);assert np.all((lo>0)|(hi<0))
 rbar=float(row['flow_tube_radius']);ctl=m.stack([a0[:m.h],a0[:m.h]+f0[:m.h]*m.dt*third,a1[:m.h]-f1[:m.h]*m.dt*third,a1[:m.h]]);un=np.max(m.normup(ctl,axis=2),axis=0)
 margin=np.nextafter(np.maximum(lo,-hi)-m.mulp(rbar,m.vnxi),-np.inf);sector=np.nextafter(margin-m.mulp(m.addp(un,rbar),m.width),-np.inf)
 assert float(np.min(sector))==float(row['probe_sector_margin_lower']);assert np.all(sector>0);min_margin=min(min_margin,float(np.min(sector)))
# Recompute representative complete steps with the delivered portable verifier.
for j in [0,1199,1200,2499]:
 out=m.step(j)
 for key in ['G_lower','G_upper','E_lower','E_upper','Xi_lower','Xi_upper','E_sector_lower','E_sector_upper']:
  assert out[key]==float(rows[j][key]),(j,key)
result=dict(status='PASS',whole_step_probe_pattern_checks=2500,nominal_patterns_match_Bernstein_patterns=True,minimum_sector_margin=min_margin,full_steps_recomputed=[1,1200,1201,2500]);(p/'TIMING_GEOMETRY_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
