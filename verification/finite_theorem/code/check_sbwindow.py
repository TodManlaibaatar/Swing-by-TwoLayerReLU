from fractions import Fraction as F
from pathlib import Path
import json, math, runpy
# Rational enclosures: no training or trajectory experiments.
def exp_pos(x,n=140):
    assert x>=0
    term=F(1); total=term
    for k in range(1,n+1):term=term*x/k;total+=term
    nxt=term*x/(n+1)
    assert x<F(n+2)
    return total,total+nxt/(1-x/F(n+2))
def eneg(x):
    lo,hi=exp_pos(x);return 1/hi,1/lo
def log_pos(x,n=180):
    assert x>=1
    z=(x-1)/(x+1); lo=2*sum((z**(2*k+1)/F(2*k+1) for k in range(n)),F(0))
    hi=lo+2*z**(2*n+1)/(F(2*n+1)*(1-z*z))
    return lo,hi
c=F('2.24')*F('.95')/F('1.01');d=F('.22884')/F('1.01')-F('.02')
loglo,loghi=log_pos(1+c*F('.32')/d);tylo,tyhi=loglo/c,loghi/c
assert tyhi<F('.689')
A=F('.3392215');int_y_hi=F('.32')/c-d*tylo/c
cross=A*F('1.02')-F('.5049')*int_y_hi
assert cross>F('.3033')
gu=F('.0516265');gw=(F('.39')-F('2.26')*(F('.05')*F('.32')+F('.122')))/F('1.01')-F('.02')
assert F('1.02')+F('.135')/gu<F('3.635')
assert F('1.02')+F('.122')/gw<F('3.148')
lm=F('1.53425');pre=F('.119');post=F('.033375')
ru=pre/lm+(F('.555')-pre/lm)*eneg(lm*F('1.02'))[1]
rm=post/lm+(ru-post/lm)*eneg(lm*F('3.08'))[1]
assert ru<F('.178') and rm<F('.0232')
rho=F('0.0001'); f=exp_pos(20*rho/F('8.8'))[1]*(rho/2+(F('2.1')+20*rho)*rho/2+48*rho*rho/F('10.8'))
assert f<F('.000156') and rho+F('2.2')*f<F('.000444')
cross_output=F('7.5')*f+12*rho
assert cross_output<F('.00237')
network=F('1.000101')/2*((1+F('.025')**2*F('.3')*F('.32'))*F('.00237')+5*rho*F('.62'))
target=F('1.000101')*F('1.336')*F('.025')**2
restored=F('1.000101')*rho*max(F('.01')+F('.58')+F('2.26')*F('.30'),2*F('2.26')*F('.32'))
assert network<F('.001341') and target<F('.000836') and restored<F('.000145')
assert network+target+restored+F('.0001')<F('.0025')
osc=F('2.26')*F('0.00000001')/(1-F('.0001'))+F('.002501')+F('.001001')+F('.000778')+F('.0002')
assert osc<F('.0045')
assert F('.0025')+F('.000201')+F('.00072')<F('.004')
# Full finite-state verification of the early comparison using deterministic fixtures.
x=runpy.run_path('work/check_sbcouple.py')
dot=x['dot']; norm=x['norm'];plus=x['plus'];matvec=x['matvec'];trans=x['trans'];add=x['add'];scale=x['scale']
count=0
for data in x['datasets']:
 for u in x['vectors']:
  for vv in x['vectors']:
   v=[(.7,.02),(-.04,.3),u,vv]
   for rotate in (False,True):
    z=[(.6*p-.8*q,.8*p+.6*q) if rotate else (p,q) for p,q in v]
    w=1/len(data);acts=[[plus(dot(vi,xx)) for _,xx in data] for vi in v]
    al=[[float(dot(vi,xx)>0) for _,xx in data] for vi in v]
    out=[[sum(z[j][k]*acts[j][m] for j in range(4)) for k in range(2)] for m in range(len(data))]
    AA=[[[sum(w*(xx[k]-out[m][k])*xx[l]*al[i][m] for m,(_,xx) in enumerate(data)) for l in range(2)] for k in range(2)] for i in range(4)]
    dv=[matvec(trans(AA[i]),z[i]) for i in range(4)];dz=[matvec(AA[i],v[i]) for i in range(4)]
    dd=[add(v[i],scale(-1,z[i])) for i in range(4)];ddd=[add(dv[i],scale(-1,dz[i])) for i in range(4)]
    S=sum(dot(vi,vi) for vi in v);D=math.sqrt(sum(dot(di,di) for di in dd))
    Dp=sum(dot(dd[i],ddd[i]) for i in range(4))/D if D else math.sqrt(sum(dot(di,di) for di in ddd))
    H=Hp=0.
    for i,(p,q) in enumerate(v):
     if p>0 and q>0:H+=p*q;Hp+=q*dv[i][0]+p*dv[i][1]
     if p*q<0:H-=p*q;Hp-=q*dv[i][0]+p*dv[i][1]
    Sigma=[[sum(w*xx[k]*xx[l] for _,xx in data) for l in range(2)] for k in range(2)]
    lam=(Sigma[0][0]+Sigma[1][1]+math.sqrt((Sigma[0][0]-Sigma[1][1])**2+4*Sigma[0][1]**2))/2
    Cx=Sigma[0][0]+Sigma[1][1];ell=max(sum(w*xx[k]**2 for kk,xx in data if kk==k) for k in range(2))
    chi=sum(w*abs(xx[1-k])*(norm(xx)+xx[k]) for k,xx in data)
    bc=sum(w*abs(xx[0]*xx[1]) for _,xx in data)
    Z=math.sqrt(S)*D;Zp=lam*Z+math.sqrt(S)*Dp
    rhsH=(Cx+ell*S)*H+lam*(1+2*S)*Z+bc*S+chi*S*S
    rhsZ=ell*S*H+(lam+(lam+math.sqrt(lam*Cx))*S)*Z+chi*S*S
    assert Hp<=rhsH+1e-8 and Zp<=rhsZ+1e-8
    Sprime=2*sum(w*(dot(xx,out[m])-dot(out[m],out[m])) for m,(_,xx) in enumerate(data))
    assert Sprime<=2*lam*S+1e-9
    # Exact Gaussian target formula versus finite empirical target; static checks only.
    sigmas=(.05,.08); mus=(3.,2.)
    GP=[]; disc=[]
    for i,vi in enumerate(v):
     variance=sum(sigmas[k]**2*vi[k]**2 for k in range(2)); sd=math.sqrt(variance)
     gp=[0.,0.]
     for kk in range(2):
      mean=mus[kk]*vi[kk]; aa=mean/sd
      probability=.5*(1+math.erf(aa/math.sqrt(2)))
      erelu=mean*probability+sd*math.exp(-aa*aa/2)/math.sqrt(2*math.pi)
      for k in range(2):gp[k]+=.5*(sigmas[k]**2*vi[k]*probability+(mus[kk]*erelu if k==kk else 0.))
     for k in range(2):
      if vi[k]<=0:assert gp[k]>=sigmas[k]**2*vi[k]-1e-10
     emp=[sum(w*xx[k]*acts[i][m] for m,(_,xx) in enumerate(data)) for k in range(2)]
     disc.append(norm(add(emp,scale(-1,gp)))/norm(vi));GP.append(gp)
    nu_local=max(disc);dcurrent=D/math.sqrt(S)
    for k in range(2):
     neg=[i for i in range(4) if v[i][k]<0]
     yy=math.sqrt(sum(v[i][k]**2 for i in neg))
     yp=sum(v[i][k]*dv[i][k] for i in neg)/yy if yy else 0.
     rhs=sigmas[k]**2*yy+(nu_local+lam*S)*math.sqrt(S)+lam*(1+S)*D
     assert yp<=rhs+1e-8
    dprime=Dp/math.sqrt(S)-lam*dcurrent
    assert dprime<=(-lam+(lam+math.sqrt(lam*Cx))*S)*dcurrent+ell*H+chi*S+1e-8
    count+=1
# Exact constant checks for the signed refinement and its positive time exponent.
rho_max=F('0.0001');d_coefficient=F('7.8')/(F('4.4')-11*rho_max)
assert d_coefficient<F('1.78')
assert 5+5*F('1.0001')*F('1.78')+5<19
b0=F('4.3975');zeta0=F('1.043')
assert 38/b0+F('1.78')<11 and 2/b0<F('.456')
g_bound=4*F('2.24')-2*5*rho_max-2*F('.025')*F('.32')*(F('2.1')*F('.025')+5*rho_max)-F('.0002')
assert g_bound>F('8.957') and 10-F('8.957')==zeta0 and b0-zeta0==F('3.3545')
assert F('.208')/4==F('.052')
result={'rational_clock_bounds':{'input_integral_lower':float(cross),'output_crossing_upper':float(tyhi),'mean_radius_upper_at_4_1':float(rm)},'from_zero_scalar_bound':{'F_over_e_upper':float(f),'complement_upper':float(rho+F('2.2')*f)},'early_finite_state_checks':count,'negative_coordinate_static_checks':count,'signed_feedback_constants':{'d_over_rho_upper':float(d_coefficient),'seed_growth_lower':float(g_bound),'relative_decay_gap':float(b0-zeta0),'c21_over_eM_lower':.052},'status':'passed','scope':'Symbolic/rational comparison checks only; no training experiments or common-region certification.'}
Path('outputs/SWINGBY_FINITE_WINDOW_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
