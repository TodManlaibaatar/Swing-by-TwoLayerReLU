"""Independent full-step checks against the delivered coefficient ledger."""
from pathlib import Path
import sys,csv,json,importlib.util,numpy as np
p=Path(__file__).resolve().parent.parent
sys.argv=['validator','--reference',str(p/'inputs/canonical_initial_reference.npz'),'--output',str(p)]
spec=importlib.util.spec_from_file_location('initial_validator',p/'code/validate_initial_segment.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
with (p/'INITIAL_FLOW_ENCLOSURE.csv').open() as f:ledger={int(r['step']):r for r in csv.DictReader(f)}
with (p/'INITIAL_SEGMENT_TUBES.csv').open() as f:tubes={int(r['step']):r for r in csv.DictReader(f)}
checks=[]
for j in [0,2486,4973]:
 t=tubes[j];r=v.step((j,float(t['radius']),float(t['beta'])));old=ledger[j+1]
 for key in ['eta_upper','eta_smooth_upper','eta_gate_upper','gate_forcing_upper','perturbation_rate_upper','smooth_rate_upper']:
  assert r[key]==float(old[key]),(j,key,r[key],old[key])
 checks.append(j+1)
report=dict(status='PASS',full_steps_exactly_recomputed=checks,scope='Implementation regression; all other intervals covered by delivered validated ledger and deterministic proof.')
(p/'FULL_STEP_RECOMPUTATION_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
