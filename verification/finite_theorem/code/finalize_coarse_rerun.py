from pathlib import Path
import csv,datetime,hashlib,json,shutil,zipfile
root=Path.cwd();out=root/'outputs'
backup=out/'simulation-provenance/before-coarse-rerun-2026-09-20'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(tag):return list(csv.DictReader((out/f'SWINGBY_SIM_{tag}.csv').open()))
before=json.loads((backup/'before_hashes.json').read_text())
unchanged=['work/sim_sbstrength.cpp','work/analyze_sbstrength_sim.py','outputs/SWINGBY_SIM_fine.csv','outputs/SWINGBY_SIM_fine_initial.txt','outputs/SWINGBY_SIM_upper.csv','outputs/SWINGBY_SIM_upper_initial.txt']
for name in unchanged:assert sha(root/name)==before[name],name
c,f,u=map(rows,['coarse','fine','upper'])
assert len(c)==492,len(c)
assert [r['t'] for r in c]==[r['t'] for r in u], 'Coarse and upper grids differ'
assert 'Finished 491 steps' in (out/'SWINGBY_SIM_coarse_rerun.log').read_text()
violations=[]
for a,b in zip(c,c[1:]):
 t=float(a['t']);dt=float(b['t'])-t
 cap=1e-5 if t<.00017-1e-12 else (.001 if t<.1 else .02)
 if not(0<dt<=cap+1e-12):violations.append((t,dt,cap))
assert not violations,violations
j=json.loads((out/'SWINGBY_STRENGTHENING_SIMULATION.json').read_text())
assert len(j['instrumented_checks'])==22 and all(j['instrumented_checks'].values())
for name,digest in j['files_sha256'].items():
 p=root/'work'/name if name.endswith('.cpp') else out/name
 assert sha(p)==digest,(name,'stale analysis hash')
provpath=out/'SWINGBY_SIM_coarse_rerun_provenance.json';p=json.loads(provpath.read_text())
p.update(status='complete',completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),old_coarse_rows=407,new_coarse_rows=len(c),fine_rows=len(f),upper_rows=len(u),step_caps_verified=True,coarse_grid_equals_upper=True,unchanged_files_verified=unchanged,common_snapshots=j['coarse_fine_common_snapshots'],max_discrepancies=j['coarse_fine_max_discrepancy'],instrumented_assertions_passed=22,instrumented_assertion_scope='Original analyzer: fine and upper trajectories; coarse used for common-snapshot refinement comparison.')
provpath.write_text(json.dumps(p,indent=2)+'\n')
report=out/'SWINGBY_COARSE_RERUN_REPORT.md'
lines=['# Coarse trajectory provenance repair','',f"Regenerated only the inner-probe coarse run from the archived, unchanged C++ source. Seed 20260918; h=10^7; N=2000; means (1, 1e-4); sigma=1e-9; theta=4e-5; dtmax=0.02. No resampling. Fine and upper trajectories are unchanged.",'',f"- Coarse data rows: **{len(c)}** (excluding the CSV header).",f"- Common coarse/fine snapshots: **{p['common_snapshots']}** (the analyzer matches times rounded to 10 decimal places).",'- All **22/22** original instrumented assertions pass. These check the fine/upper trajectories; the coarse run supplies the refinement comparison.','- All actual step increments satisfy the source caps; the full timestamp grid equals the archived upper run.','- Initialization manifest is byte-for-byte unchanged.','', '| Quantity | Maximum absolute coarse/fine discrepancy |','| --- | ---: |']
for key,val in p['max_discrepancies'].items():lines.append(f'| `{key}` | {val:.17g} |')
lines+=['','The unchanged analysis script regenerated the summary and validation figure. Its plot uses fine and upper trajectories, so changing coarse alone does not change the plotted curves.','', 'The JSON provenance record gives the compiler, exact commands, source hash, and binary hash. All deliverable hashes are in `SWINGBY_COARSE_RERUN_SHA256.json`. Historical coarse outputs are preserved separately and excluded from the refreshed code package. No theorem or parameter changes were made.']
report.write_text('\n'.join(lines)+'\n')
# Restore the previously delivered source package, then replace only generated artifacts and documentation.
dest=out/'SWINGBY_10M_SIMULATION_CODE'
with zipfile.ZipFile(backup/'SWINGBY_10M_SIMULATION_CODE.zip') as z:z.extractall(out)
for filename in ['SWINGBY_SIM_coarse.csv','SWINGBY_SIM_coarse_initial.txt','SWINGBY_STRENGTHENING_SIMULATION.json','SWINGBY_STRENGTHENING_VALIDATION.png','SWINGBY_SIM_coarse_rerun.log','SWINGBY_SIM_coarse_rerun_provenance.json','SWINGBY_COARSE_RERUN_REPORT.md']:
 shutil.copy2(out/filename,dest/'outputs'/filename)
r=dest/'README.md'
s=r.read_text()
s=s.replace('This package was assembled from existing local artifacts; no training was rerun.','The inner-probe coarse trajectory was regenerated from this unchanged source on 2026-09-20. The fine and upper runs remain unchanged. See outputs/SWINGBY_COARSE_RERUN_REPORT.md and the provenance JSON for exact commands and checks.')
s=s.replace('The exact original compiler\ncommand and toolchain were not recorded in the supplied summary; the commands\nabove are build instructions, not a recovered execution log.','The coarse rerun compiler and exact commands are recorded in\noutputs/SWINGBY_SIM_coarse_rerun_provenance.json. The original fine/upper\ncompiler commands were not recorded in the supplied summary.')
r.write_text(s)
manifest={str(q.relative_to(dest)):sha(q) for q in sorted(dest.rglob('*')) if q.is_file() and q.name!='SHA256_MANIFEST.json'}
(dest/'SHA256_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
zpath=out/'SWINGBY_10M_SIMULATION_CODE.zip'
with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
 for q in sorted(dest.rglob('*')):
  if q.is_file():z.write(q,Path(dest.name)/q.relative_to(dest))
paths=[root/'work/sim_sbstrength.cpp',root/'work/analyze_sbstrength_sim.py']+[out/n for n in ['SWINGBY_SIM_coarse.csv','SWINGBY_SIM_coarse_initial.txt','SWINGBY_SIM_fine.csv','SWINGBY_SIM_upper.csv','SWINGBY_STRENGTHENING_SIMULATION.json','SWINGBY_STRENGTHENING_VALIDATION.png','SWINGBY_SIM_coarse_rerun.log','SWINGBY_SIM_coarse_rerun_provenance.json','SWINGBY_10M_SIMULATION_CODE.zip']]
(out/'SWINGBY_COARSE_RERUN_SHA256.json').write_text(json.dumps({str(q.relative_to(root)):sha(q) for q in paths},indent=2)+'\n')
print(json.dumps(p,indent=2))
