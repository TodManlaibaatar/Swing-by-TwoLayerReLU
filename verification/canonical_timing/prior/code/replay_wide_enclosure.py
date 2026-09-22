"""Independent scalar replay with outward rounding of decimal entry radius."""
from pathlib import Path
import json,csv,argparse,numpy as np,mpmath as mp
ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,default=Path('.'));args=ap.parse_args();p=args.directory
mp.iv.dps=60
with (p/'WIDE_LOCAL_FLOW_ENCLOSURE.csv').open() as f:rows=list(csv.DictReader(f))
r=np.nextafter(1.2e-5,np.inf);initial=float(r);margin=float('inf')
for expected,row in zip(range(17001,19501),rows):
 assert int(row['step'])==expected
 K=mp.iv.mpf(float(row['rate_upper']))+mp.iv.mpf(float(row['gate_rate']))
 W=mp.iv.mpf(float(row['eta_upper']))+mp.iv.mpf(float(row['gate_forcing_upper']))
 D=mp.iv.mpf(1)/5000
 r=np.nextafter(float((mp.iv.exp(K*D)*mp.iv.mpf(float(r))+(mp.iv.exp(K*D)-1)/K*W).b),np.inf)
 assert r<float(row['tube_radius_upper'])
 row['radius_upper']=float(r);margin=min(margin,float(row['tube_radius_upper'])-r)
assert len(rows)==2500 and r<4.29e-5
assert float(rows[1499]['radius_upper'])<3.5e-5
with (p/'WIDE_LOCAL_FLOW_ENCLOSURE.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
report=json.loads((p/'WIDE_LOCAL_FLOW_CERTIFICATE.json').read_text());assert report['status'].startswith('LOCAL FLOW ENCLOSED')
report.update(initial_radius=1.2e-5,initial_radius_decimal='0.000012',initial_radius_float_upper=initial,final_radius_upper=float(r),scalar_replay='PASS: 60-digit interval exponential; exact rational step; upward initial decimal radius',minimum_step_slack_diagnostic=margin,snapshot_radius_at_3p7=float(rows[1499]['radius_upper']))
(p/'WIDE_LOCAL_FLOW_CERTIFICATE.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
