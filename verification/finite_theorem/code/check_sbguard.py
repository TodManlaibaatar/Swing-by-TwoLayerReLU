from fractions import Fraction as F
from pathlib import Path
import math,json,hashlib,re
b=F('4.3975');z=F('1.043');beta=b-z;p=b/beta
rho=F('0.0001');e=F('.025');U=F('.337');W=F('.347')
err=F('1.011')/2*((1+e*e*U*W)*F('.00237')+5*rho*(U+W))+F('1.011')*e*e*(2*W+2*U+U*W)+F('1.011')*rho*max(F('.01')+F('.61')+F('2.26')*U,2*F('2.26')*W)+F('.0001')
assert err<F('.0026')
osc=F('2.26')*F('.01')**2/F('.99')+F('.002501')+F('.001001')+F('.000866')+F('.0002')
assert osc<F('.0051')
assert 2*e*e*(U/F('.99')+W*F('1.01'))*(1+F('.00237')/2)<F('.000866')
rs=F('.01')*F('2.01')+F('.0026')+(U-F('.135'))*F('.0051')/4
rt=F('.01')/F('.99')*F('1.41')+F('.0026')+(W-F('.122'))*F('.0051')/4
assert rs<F('.024') and rt<F('.018')
qerr=e*(U+W*F('1.01')**2)*F('.053')+F('1.01')*F('.0002')
assert qerr<F('.0012')
assert F('4.4795')*(1-F('.99')**2)>F('.0891')
assert F('4.4795')*(F('1.01')**2-1)>F('.0900')
v_growth=F('.99')*(F('4.4795')-e*W*F('.053'))-F('.0002')
assert v_growth>F('4.43')
M_growth=4*F('2.24')-2*5*rho-2*e*W*F('.053')-F('.0002')
assert M_growth==F('8.95788045') and M_growth>F('8.957')
coarse=F('1.000101')/2*((1+e*e*F('.30')*F('.32'))*5*F('.5')/e+5*F('.5')*F('.62'))
assert coarse>F('50.78') and coarse-F('.99')*F('.0343462')>F('50.74')
# Beta-moment and initialization inequalities, rational checks over representative dimensions/orders.
for d in [2,3,4,10,100,1000]:
 for k in range(1,20):
  moment=F(1)
  for j in range(k):moment*=F(d*(j+1),d+2*j)*2
  assert moment<=2**k*math.factorial(k)
 tau2=1. if d==2 else -math.expm1(-2*math.log(2)/(d-2))
 assert d*tau2>=2*math.log(2)
# Static projected-energy identity with arbitrary planar vectors and shared frozen input perpendicular norm.
states=0
for vs in [[(.4,.1),(-.2,.3)],[(.2,-.4),(.3,.1)]]:
 for zs in [[(.3,.2),(-.1,.4)],[(.5,-.1),(.1,.2)]]:
  data=[(3.,.05),(-.1,2.)];ws=[.07,-.04]
  dot=lambda a,b:sum(x*y for x,y in zip(a,b))
  act=[[max(dot(v,x),0) for x in data] for v in vs]
  out=[[sum(zs[i][k]*act[i][j] for i in range(2)) for k in range(2)] for j in range(2)]
  per=[sum(ws[i]*act[i][j] for i in range(2)) for j in range(2)]
  A=[[[sum((data[j][k]-out[j][k])*data[j][l]*(dot(vs[i],data[j])>0)/2 for j in range(2)) for l in range(2)] for k in range(2)] for i in range(2)]
  pp=[[sum(data[j][k]*(dot(vs[i],data[j])>0)*ws[i]*per[j]/2 for j in range(2)) for k in range(2)] for i in range(2)]
  vd=[[sum(A[i][l][k]*zs[i][l] for l in range(2))-pp[i][k] for k in range(2)] for i in range(2)]
  zd=[[sum(A[i][k][l]*vs[i][l] for l in range(2)) for k in range(2)] for i in range(2)]
  Pdot=sum(dot(vs[i],vd[i])+dot(zs[i],zd[i]) for i in range(2))
  exact=sum(2*dot(data[j],out[j])-2*dot(out[j],out[j])-per[j]**2 for j in range(2))/2
  assert abs(Pdot-exact)<1e-12
  states+=1
result={'old_log_d_coefficient':float(1/beta),'old_rho_dimension_power':float(p),'old_concentration_power':float(2*p),'holding_individual_error_upper':float(err),'holding_mean_input_upper':float(rs),'holding_mean_output_upper':float(rt),'q_adverse_charge_upper':float(qerr),'longitudinal_growth_lower':float(M_growth),'coarse_late_allowance':float(coarse),'projected_energy_identity_checks':states,'status':'passed','scope':'Exact rational constants and static identities, not trajectory experiments or a common primitive-region proof.'}
Path('outputs/SWINGBY_BRANCH_DIMENSION_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
