"""Replay interval-derived initialization bounds after independent smooth-rate recomputation.
Consumes the defect/gate coefficient ledger and independently recomputed signed smooth rates.
Never uses the superseded accumulated radius as an input.
"""
from pathlib import Path
import csv,json,numpy as np,mpmath as mp,argparse
ap=argparse.ArgumentParser();ap.add_argument('--directory',type=Path,default=Path('.'));a=ap.parse_args();p=a.directory
mp.iv.dps=60
with (p/'INITIAL_COEFFICIENTS.csv').open() as f:raw=list(csv.DictReader(f))
with (p/'RECOMPUTED_SMOOTH_RATES.csv').open() as f:rates={int(r['step']):float(r['smooth_rate_upper']) for r in csv.DictReader(f)}
r=0.;rows=[];target=mp.iv.mpf('0.000012');previous=0.;status='PREFIX VALIDATED; TARGET NOT YET EXCEEDED'
for j,old in enumerate(raw,1):
 assert int(old['step'])==j and j in rates
 row={k:float(v) for k,v in old.items()};row['step']=j
 row['smooth_rate_upper']=rates[j]
 row['rate_upper']=np.nextafter(rates[j]+row['perturbation_rate_upper'],np.inf)
 K=mp.iv.mpf(row['rate_upper'])+mp.iv.mpf(row['gate_rate'])
 W=mp.iv.mpf(row['eta_upper'])+mp.iv.mpf(row['gate_forcing_upper'])
 D=mp.iv.mpf(1)/5000;previous=r
 r=float(np.nextafter(float((mp.iv.exp(K*D)*mp.iv.mpf(r)+(mp.iv.exp(K*D)-1)/K*W).b),np.inf))
 row['radius_upper']=r;row['previous_radius_upper']=previous
 row['step_tube_slack_lower']=float(np.nextafter(row['tube_radius_upper']-r,-np.inf))
 if not r<row['tube_radius_upper']:
  status='FIRST FAILED TUBE INEQUALITY';failure=row;break
 rows.append(row)
 if mp.iv.mpf(r).a>target.b:
  status='PREFIX VALIDATED; NONDECREASING COMPARISON EXCEEDS ENTRY BUDGET';failure=row;break
else:failure=rows[-1]
report=dict(status=status,initial_radius=0,initial_reference_index=0,last_validated_reference_index=rows[-1]['step'] if rows else 0,first_failed_or_budget_step=failure['step'],target_entry_radius_exact='0.000012',failure_comparison=failure,arithmetic='60-digit interval exponential; exact rational dt=1/5000; nextafter upward replay; smooth rates independently recomputed with explicit single normalization.',scope='Failure of this sufficient scalar enclosure only. It is not a lower bound on the true error and is not an obstruction to initialized reversal.')
(p/'INITIAL_FLOW_CERTIFICATE.json').write_text(json.dumps(report,indent=2)+'\n')
with (p/'INITIAL_FLOW_ENCLOSURE.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
print(json.dumps(report,indent=2))
