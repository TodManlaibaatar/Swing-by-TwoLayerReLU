from pathlib import Path
import csv,json,hashlib,mpmath as mp
p=Path(__file__).resolve().parent.parent;f=p/'TIMING_FULL_TUBE_CERTIFICATE.csv';rows=list(csv.DictReader(f.open()));assert len(rows)==2500
for j,q in enumerate(rows):assert int(q['step'])==17001+j
unresolved=[int(q['step']) for q in rows if q['status']!='ENCLOSED'];assert not unresolved
mp.iv.dps=60
reports={}
for name,lo,hi in [('G','G_lower','G_upper'),('E_central','E_lower','E_upper'),('E_sector','E_sector_lower','E_sector_upper')]:
 first=next(j for j,q in enumerate(rows) if float(q[hi])>=0);last=max(j for j,q in enumerate(rows) if float(q[lo])<=0)
 assert all(float(q[hi])<0 for q in rows[:first]);assert all(float(q[lo])>0 for q in rows[last+1:])
 reports[name]=dict(left_index=17000+first,right_index=17001+last,denominator=5000,left=(17000+first)/5000,right=(17001+last)/5000,unresolved_sign_steps=[int(q['step']) for q in rows if float(q[lo])<=0<=float(q[hi])],negative_prefix_slack=min(-float(q[hi]) for q in rows[:first]),positive_suffix_slack=min(float(q[lo]) for q in rows[last+1:]),last_negative_step_bounds={k:float(rows[first-1][k]) for k in [lo,hi]},first_positive_step_bounds={k:float(rows[last+1][k]) for k in [lo,hi]})
gap_num=reports['E_sector']['left_index']-reports['G']['right_index'];assert gap_num>0
lag=[q for q in rows if int(q['step'])>reports['G']['right_index'] and int(q['step'])<=reports['E_sector']['left_index']]
assert lag and all(float(q['G_lower'])>0 and float(q['E_sector_upper'])<0 for q in lag)
# Direct discrepancy bound, not subtraction of the two independently widened rates.
assert all(float(q['Xi_sector_upper'])<0 for q in lag)
report=dict(status='CLOSED: FULL-TUBE CONDITIONAL SIGN-TIMING AND MINIMIZER LOCALIZATION',condition='Distance at t=3.4 to the archived reference <= 0.000012; no initialized reachability claim',brackets=reports,sector_half_width='0.000001',gap_exact=f'{gap_num}/5000',gap=float(gap_num/5000),lag_interval=[reports['G']['right'],reports['E_sector']['left']],lag_step_count=len(lag),lag_G_lower=min(float(q['G_lower']) for q in lag),lag_E_sector_upper=max(float(q['E_sector_upper']) for q in lag),lag_Xi_direct_lower=min(float(q['Xi_lower']) for q in lag),lag_Xi_direct_upper=max(float(q['Xi_upper']) for q in lag),lag_Xi_sector_upper=max(float(q['Xi_sector_upper']) for q in lag),minimum_probe_sector_margin_lower=min(float(q['probe_sector_margin_lower']) for q in rows),unresolved_domain_steps=unresolved,training_ambiguous_steps=[int(q['step']) for q in rows if int(q['training_ambiguous_pairs'])],training_ambiguity_semantics='Each ambiguous training selector is enclosed, never frozen or assumed. These steps have rigorous rate bounds; only unresolved signs are bracketed.',csv_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),input_audit=json.loads((p/'TIMING_ENCLOSURE_RUN.json').read_text()),not_claimed=['Unique or continuous G zero','Unique OOD minimizer','Equality to empirically interpolated timing values','Initialized entry or high probability'])
(p/'TIMING_FULL_TUBE_CERTIFICATE.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['input_audit','training_ambiguous_steps']},indent=2))
