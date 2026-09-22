from pathlib import Path
import csv,json,math,hashlib
root=Path.cwd();out=root/'outputs'
def read(tag):
 return [{k:float(v) for k,v in row.items()} for row in csv.DictReader((out/f'SWINGBY_SIM_{tag}.csv').open())]
fine,coarse,upper=map(read,['fine','coarse','upper'])
q0=fine[0]['A2'];times={k:math.log(q/(1-q)*(1-q0)/q0) for k,q in [('pre',.69),('spec',.71),('a',.715),('b',.95)]}
for rows in [fine,coarse,upper]:assert abs(rows[-1]['t']-times['b'])<1e-9,'Simulation not complete'
def at(rows,t):
 r=min(rows,key=lambda x:abs(x['t']-t));assert abs(r['t']-t)<1e-9,(r['t'],t);return r
def summary(rows):
 minimum=min(rows,key=lambda x:x['E']);cross=[]
 for a,b in zip(rows,rows[1:]):
  if a['dE']<0<b['dE']:cross.append([a['t'],b['t']])
 return dict(sampled_min_time=minimum['t'],sampled_min_error=minimum['E'],sampled_maximum_improvement=rows[0]['E']-minimum['E'],derivative_crossing_brackets=cross,endpoint_secant=at(rows,times['b'])['E']-at(rows,times['a'])['E'],initial_drop_at_00017=rows[0]['E']-at(rows,.00017)['E'],initial_value=rows[0]['E'])
bytime={round(r['t'],10):r for r in coarse};common=[(r,bytime[round(r['t'],10)]) for r in fine if round(r['t'],10) in bytime]
comparison={key:max(abs(a[key]-b[key]) for a,b in common) for key in ['E','core_mass','S','weak_fit','angle_u','angle_w','dE']}
ledger=json.loads((out/'SWINGBY_STRENGTHENING_CHECKS.json').read_text())['witness'];lv={k:float(v) for k,v in ledger.items()}
late=[r for r in fine if r['t']>=times['spec']-1e-10]
checks={'S':max(r['S'] for r in fine)<2,'core_mass':min(r['core_mass'] for r in late)>.55,'core_angle':max(max(r['angle_u'],r['angle_w']) for r in late)<.13,'weak_fit':max(abs(r['weak_fit']) for r in fine)<.011,'z':max(r['z'] for r in fine)<lv['Z'],'l':max(r['l'] for r in fine)<lv['el'],'q_log':max(r['log_u_A'] for r in fine)<lv['q'],'q_direction':max(r['direction_error'] for r in fine)<lv['q'],'Cc':max(r['c_error'] for r in fine)<lv['Cc'],'B_movement':max(r['B_movement'] for r in fine)<lv['P']*3.8,'D_movement':max(r['D_movement'] for r in fine)<lv['DD'],'J_D':max(r['J_D_observed'] for r in fine)<lv['JD'],'J_B':max(r['J_B_observed'] for r in fine)<lv['JB'],'J_p':max(r['J_p_observed'] for r in fine)<lv['JP'],'Gamma_core':max(r['Gamma_core_observed'] for r in fine)<lv['GC'],'norm_factor':max(r['norm_factor_observed'] for r in fine)<math.exp(lv['V']),'core_positive':min(r['min_core_a'] for r in fine)>0,'Mstar':max(abs(r['m']-r['A2']) for r in fine)<lv['M'],'inner_late_rate':min(r['dE'] for r in fine if r['t']>=times['pre']-1e-10)>1e-5,'upper_late_rate':min(r['dE'] for r in upper if r['t']>=times['pre']-1e-10)>3.56e-6,'inner_initial_rate':max(r['dE'] for r in fine if r['t']<=.00017+1e-12)<-1.5e-9,'upper_initial_rate':max(r['dE'] for r in upper if r['t']<=.0055)<-2.67e-8}
assert all(checks.values()),checks
metrics={key:max(r[key] for r in fine) for key in ['S','weak_fit','z','l','log_u_A','direction_error','c_error','B_movement','D_movement','J_D_observed','J_B_observed','J_p_observed','Gamma_core_observed','norm_factor_observed','balance_error']}
result={'model':'Exact certified h=10^7, N=2000, d=2, s=.04, mu=(1,1e-4), sigma=1e-9; no data or neuron subsampling.','method':'Classical RK4 with actual empirical gates; bulk aggregation only for masks certified uniform across a cluster, all boundary neurons evaluated sample by sample. Double precision, fixed seed 20260918, no resampling.','probability_role':'Illustration only; all theorem claims are analytic.','probe_inner':4e-5,'probe_upper':.00035,'steps':{'coarse':.02,'fine':.01,'early':'1e-5 through .00017; .001 through .1 in refined runs'},'times':times,'inner':summary(fine),'upper':summary(upper),'coarse_fine_common_snapshots':len(common),'coarse_fine_max_discrepancy':comparison,'instrumented_checks':checks,'observed_maxima':metrics,'snapshots':{k:{j:at(fine,t)[j] for j in ['t','E','core_mass','angle_u','angle_w','S','weak_fit','dE']} for k,t in times.items()},'caveats':'Minimum brackets are sampled numerical derivative crossings, not rigorous locations or uniqueness results. The upper-probe run uses dt=.02; refinement validation is for the same training trajectory at the inner probe, not an independent rigorous integrator error bound. Tiny differences below rounding scale are not interpreted.'}
result['files_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'work/sim_sbstrength.cpp',out/'SWINGBY_SIM_fine.csv',out/'SWINGBY_SIM_coarse.csv',out/'SWINGBY_SIM_upper.csv']}
(out/'SWINGBY_STRENGTHENING_SIMULATION.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
# Standard scientific plot, separate from the theorem proof.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axs=plt.subplots(2,3,figsize=(13,7.6),layout='constrained')
for rows,label,col in [(fine,r'$\theta=4\times10^{-5}$','#136f93'),(upper,r'$\theta=3.5\times10^{-4}$','#bc5c21')]:
 t=np.array([r['t'] for r in rows]);err=np.array([r['E'] for r in rows]);delta=err-err[0]
 axs[0,0].plot(t,delta,label=label,color=col,lw=1.6)
 mask=t<=.8
 axs[0,1].plot(t[mask],delta[mask],label=label,color=col,lw=1.6)
 axs[0,2].plot(t,np.array([r['dE'] for r in rows]),label=label,color=col,lw=1.4)
for ax in axs[0,:]:ax.axhline(0,color='.65',lw=.7);ax.ticklabel_format(axis='y',style='sci',scilimits=(0,0))
axs[0,0].set(title='Error change over training',ylabel=r'$E(t)-E(0)$');axs[0,0].legend(fontsize=9)
axs[0,1].set(title='Initial transient (magnified)',ylabel=r'$E(t)-E(0)$')
axs[0,2].set(title='Error derivative',ylabel=r'$\dot E(t)$')
t=np.array([r['t'] for r in fine]);mass=np.array([r['core_mass'] for r in fine]);S=np.array([r['S'] for r in fine])
axs[1,0].plot(t,mass,label='Core learned mass',color='#136f93');axs[1,0].plot(t,S,label='Total mass',color='#929b35');axs[1,0].axhline(.55,color='.4',ls=':',label='Certified core floor');axs[1,0].legend(fontsize=8);axs[1,0].set(title='Learned mass',ylabel='Mass')
axs[1,1].plot(t,[r['angle_u'] for r in fine],label='Input',color='#136f93');axs[1,1].plot(t,[r['angle_w'] for r in fine],label='Output',ls='--',color='#bc5c21');axs[1,1].axhline(.13,color='.4',ls=':',label='Certified cone');axs[1,1].legend(fontsize=8);axs[1,1].set(title='Largest core angle',ylabel='Radians')
axs[1,2].plot(t,[r['weak_fit'] for r in fine],color='#136f93');axs[1,2].axhline(.011,color='.4',ls=':',label='Certified upper bound');axs[1,2].set(ylim=(.0098,.0111),title='Weak concept remains unlearned',ylabel=r'$f_2(e_2)$');axs[1,2].legend(fontsize=8)
for ax in axs.flat:
 ax.set_xlabel('Physical training time');ax.grid(alpha=.15)
 if ax is not axs[0,1]:ax.axvline(times['spec'],color='#733b73',ls='--',lw=.8);ax.axvspan(times['a'],times['b'],color='#e8d7bd',alpha=.3)
fig.suptitle('Certified parameter point: 10 million neurons, 2,000 noisy samples\nNumerical validation only; dashed line marks declared specialization, shading marks rebound snapshots',fontsize=12)
fig.savefig(out/'SWINGBY_STRENGTHENING_VALIDATION.png',dpi=190)
