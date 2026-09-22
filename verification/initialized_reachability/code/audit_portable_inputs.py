from pathlib import Path
import json,hashlib,sys,numpy as np
p=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(p/'provenance'))
import theory_guided_swing_by_phase_diagram as ex
z=np.load(p/'inputs/canonical_witness_states.npz');a=np.load(p/'inputs/training_state.npz');early=np.load(p/'inputs/canonical_initial_reference.npz');late=np.load(p/'inputs/canonical_late_reference.npz');snap=np.load(p/'inputs/canonical_snapshot_states.npz')
X,_=ex.make_data(ex.Config());phi=float(np.interp(4.,a['branch_time'],a['branch_phi_deg']));blocks=ex.define_final_blocks_from_branch({'time':a['time'],'U':a['U'],'W':a['W']},phi)
checks=dict(dataset_matches_generator=np.array_equal(X,z['X']),initialization_matches_archived_rows=np.array_equal(z['initial_theta'],np.vstack([a['U'][0],a['W'][0]])),initial_reference_starts_at_exact_archive=np.array_equal(early['theta'][0],z['initial_theta']),dataset_identical_in_early_late_and_witness=all(np.array_equal(q['X'],X) for q in [early,late,z]),cohort_matches_ID_only_rule=np.array_equal(blocks['strong'],z['strong_mask']),late_reference_matches_endpoints=np.array_equal(late['theta'][[0,-1]],z['theta']),late_reference_matches_snapshots=np.array_equal(late['theta'][[0,1500,2500]],snap['theta']))
audit=json.loads((p/'INPUT_AUDIT.json').read_text())
for name in ['theory_guided_swing_by_phase_diagram.py','full_flow_diagnostic.py','canonical_G_check.py','swing_grid_preregister.py']:
 expected=next(v for k,v in audit['input_sha256'].items() if k.endswith('/'+name))
 checks['pinned_source_'+name]=hashlib.sha256((p/'provenance'/name).read_bytes()).hexdigest()==expected
assert all(checks.values()),checks
report=dict(status='PASS with recorded dataset provenance qualification',checks=checks,repository_commit=audit['repository_commit'],qualifications=audit['qualifications'])
(p/'PORTABLE_INPUT_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
