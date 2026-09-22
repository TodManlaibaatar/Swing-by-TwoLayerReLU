from check_sbstrength import *
import sympy as sp
m.mp.dps=80
v=ledger(sig='1e-9',mu='1e-4',aa='.715',ab='.95')
v['spec']=D('.71');v['mass']=v['spec']*v['share']*(1-v['GC'])**2-m.sqrt(v['s'])*v['JP']
v['angle']=m.atan(v['hi']/m.sqrt(v['spec'])*(1+m.sqrt(2)*v['JD'])/(1-v['GC']-m.sqrt(2)*v['hi']*v['JP']/m.sqrt(v['spec'])))
thlo=D('3.5e-5');thhi=D('4.5e-5');axis=v['raw']-v['charge'];arcgap=axis-D('.8')*thhi
marg=monotone(v);de=descent(v['h'],thlo,thhi,v['a'],v['mu']);T=D('.00017')
# Integrate the explicit polynomial in exp(lambda*t); no trajectory simulation.
x=sp.symbols('x');s=sp.Rational(1,25);ga=sp.Rational(99,100)
cc,dd,LL=sp.symbols('c d L');ff=s*(x*x-1);rr=2*ga*(dd+ff)+(dd+ff)**2
poly=sp.Poly(2*ga*ga*cc*(LL-ff)-(LL-ff+cc*s*x*x)*rr-cc*4*s*x*(x-1)*(ga+dd+ff)**2,x)
def integral(d,t):
 vals={cc:sp.Float(str(d['c']),90),dd:sp.Float(str(d['d0']),90),LL:sp.Float(str(d['L0']),90)}
 total=m.mpf('0')
 for (power,),coeff in poly.terms():
  coef=D(str(coeff.subs(vals).evalf(85)))
  total+=coef*(t if power==0 else m.expm1(power*d['lam']*t)/(power*d['lam']))
 return total*(1-d['a'])**2/2
adrop=integral(de,T);cdec=(1-v['a'])**2*de['margin'](T)/2
old=descent('1e10','1.5e-5','2.5e-5','1e-10','1e-5',tol=D('.00257747154594767'));Told=D('.00035');old_drop=integral(old,Told)
# Tight individual range-based Hoeffding events.
h=v['h'];momfail=6*m.exp(-2*h*D('.002')**2)+2*m.exp(-8*h*D('.002')**2)+2*m.exp(-2*h*D('.001')**2)+2*m.exp(-2*h*D('.01')**2)
xnorm=5*m.sqrt(D('.248'));b0fail=2*m.exp(-xnorm*xnorm/2)/(xnorm*m.sqrt(2*m.pi))+8*m.sqrt(2)/(m.pi*m.sqrt(h))
fail=momfail+b0fail+D('.005')+2000*m.exp(-50)
tb=m.log(v['ab']/(1-v['ab'])*(1-v['lo']**2)/v['lo']**2)/2
checks={
 'weighted_cap':v['W']<D('.005'),'q_cap':v['q']<D('.001'),'coefficient_error':3*v['P']/v['lo']<D('.001'),
 'norm_stop':v['Sstop']<2,'B_stop':v['B']<D('.3'),'q_nu':v['q']<v['nu']/8,
 'W_nu':v['W']<v['nu']/16,'g_reserve':3*v['P']/v['lo']<v['nu']/16,
 'core_purity':(1-v['GC'])/m.sqrt(2)>v['a']*m.exp(v['V']),
 'core_mass':v['mass']>D('.55'),'core_angle':v['angle']<D('.13'),'core_error':v['GC']<D('.25'),
 'time':tb<v['H'],'weak_fit':v['K']+v['ef2']<D('.011'),
 'spatial_res1':v['Ph']*(m.sqrt(v['ab'])-v['lo'])+v['C']+v['ef1']+3*thhi<D('.030'),
 'spatial_res2':1+v['K']+v['ef2']+v['B']*m.sqrt(2)*thhi<D('1.011'),
 'spatial_modulus':D('.030')*3+D('1.011')*(v['B']*m.sqrt(2)+thhi)<D('.4'),
 'arc_rebound':arcgap>D('1.1e-5'),'init_cov':de['lam']<D('.496'),
 'early_margin':cdec>D('1.5e-9'),'early_drop':adrop>D('2.04e-12'),
 'weak_early':de['margin'](T,weak=True)>0,'maximal_root_bracket':D('.00018270')<de['root']<D('.00018271'),
 'old_early_margin':(1-old['a'])**2*old['margin'](Told)/2>D('9e-11'),
 'old_drop':old_drop>D('7.45e-12'),'late_increase':marg['Eprime']>D('6e-6'),'failure':fail<D('.1'),
 'initial_order':T<1,'late_order':D('.69')<v['spec']<v['aa']<v['ab'],
}
outer=monotone(v,'.0004')
outer_th=D('.0004')
outer_V2=v['Z']*(outer['Tplus']+outer_th)+outer_th*(outer['Tplus']+v['DD'])+outer_th*v['B']*(v['el']+v['B']*v['Z']+v['P'])+v['P']*(m.sqrt(2)+v['B'])
outer_rate=(outer['f1min']*outer['f1prime']-D('1.011')*outer_V2)/2
dt_late=m.log(D('.95')*D('.285')/(D('.05')*D('.715')))
outer_rebound=outer_rate*dt_late
inner=monotone(v)
inner_V2=v['Z']*(inner['Tplus']+thhi)+thhi*(inner['Tplus']+v['DD'])+thhi*v['B']*(v['el']+v['B']*v['Z']+v['P'])+v['P']*(m.sqrt(2)+v['B'])
inner_rate=(inner['f1min']*inner['f1prime']-D('1.011')*inner_V2)/2
Fend=v['s']*m.expm1(2*de['lam']*T);Dm=4*v['s']*m.exp(de['lam']*T)*m.expm1(de['lam']*T)
dmax=de['d0']+D('.05')*(outer_th-thlo)+Fend
Lmin=v['s']*(1/(4*m.pi)-D('.002'))*m.cos(outer_th)-2*v['a']*v['s']-Fend
Smaxearly=v['s']*m.exp(2*de['lam']*T);rrmax=2*de['gamma']*dmax+dmax*dmax
Jc_lower=2*de['gamma']**2*Lmin-Smaxearly*rrmax-Dm*(de['gamma']+dmax)**2
Jd_abs=2*(de['gamma']+dmax)*(v['s']*(1/(4*m.pi)-D('.002'))+outer_th*(Smaxearly+Dm))
JL_Ltheta=2*de['gamma']**2*outer_th*v['s']*(1/(4*m.pi)-D('.002'))*m.sin(outer_th)
checks.update({'outer_rate':outer_rate>D('3.56e-6'),'outer_rebound':outer_rebound>D('7.2e-6'),'inner_signed_rebound':inner_rate*dt_late>D('2e-5'),'kernel_c_derivative':Jc_lower>D('.006'),'kernel_d_derivative':Jd_abs<D('.0063'),'kernel_L_theta':JL_Ltheta<D('2e-9'),'outer_residual_negative':v['K']+v['ef2']+v['B']*m.sqrt(2)*outer_th<m.cos(outer_th),'outer_residual_abs':1+v['K']+v['ef2']+v['B']*m.sqrt(2)*outer_th<D('1.011')})

upper=dict(de)
upper.update(c=m.sin(D('.0003'))-2*v['a'],d0=de['d0']+D('.05')*(D('.0004')-thlo),L0=v['s']*(1/(4*m.pi)-D('.002'))*m.cos(D('.0004'))-2*v['a']*v['s'])
upperT=D('.0055');upper_drop=integral(upper,upperT)
def upper_J(t):
 xx=m.exp(upper['lam']*t);ff=v['s']*(xx*xx-1);dd=upper['d0']+ff;LL=upper['L0']-ff;cc=upper['c'];gg=upper['gamma'];rr=2*gg*dd+dd*dd
 return 2*gg*gg*cc*LL-(LL+cc*v['s']*xx*xx)*rr-cc*4*v['s']*xx*(xx-1)*(gg+dd)**2
upper_rate=(1-v['a'])**2*upper_J(upperT)/2
upper_root=m.findroot(upper_J,(D('.005'),D('.006')))
def budget_checks(c0,L0,d00,Tval,lambda0):
 xx=m.exp(lambda0*Tval);ff=v['s']*(xx*xx-1);ss=v['s']*xx*xx;dd=d00+ff;ll=L0-ff;dm=4*v['s']*xx*(xx-1);rr=2*D('.99')*dd+dd*dd
 return min(ll,2*D('.99')**2*c0-rr,2*D('.99')**2*ll-ss*rr-dm*(D('.99')+dd)**2)>0
checks['inner_budget_monotonicity']=budget_checks(de['c'],de['L0'],de['d0'],T,de['lam'])
checks['upper_budget_monotonicity']=budget_checks(upper['c'],upper['L0'],upper['d0'],upperT,upper['lam'])
checks['late_factors_positive']=outer['f1min']>0 and outer['f1prime']>0
# Uniform weak budget, using the larger transported probe error through upperT.
xw=m.exp(de['lam']*upperT);fw=v['s']*(xw*xw-1);cw=m.cos(D('.0004'))-2*v['a']/v['mu'];lw=v['s']*D('.248')-v['s']*(D('.0004')+2*v['a']/v['mu'])-fw
dw=max(de['net']+D('.05')*2*v['a']/v['mu'],upper['d0'])+fw;rw=2*D('.99')*dw+dw*dw;dmw=4*v['s']*xw*(xw-1)
checks['upper_weak_nonnegative']=2*D('.99')**2*cw*lw-(lw+cw*v['s']*xw*xw)*rw-cw*dmw*(D('.99')+dw)**2>0
checks.update({'upper_sector_drop':upper_drop>D('2.15e-9'),'upper_sector_rate':upper_rate>D('2.67e-8'),'upper_root_bracket':D('.00571064')<upper_root<D('.00571066')})

# Verify integrated polynomial matches numerical quadrature of the same analytic bound.
checks['polynomial_integral']=abs(adrop-m.quad(de['margin'],[0,T])*(1-v['a'])**2/2)<D('1e-65')
assert all(checks.values()), {k:val for k,val in checks.items() if not val}
selected=['h','ep','sig','mu','a','P','B','b0','Z','el','W','q','Cc','DD','M','V','JD','JB','JP','GC','mass','angle','Sstop','raw','charge','ef1','ef2']
vals={k:m.nstr(v[k],35) for k in selected}
vals.update({k:m.nstr(val,35) for k,val in dict(axis_gap=axis,arc_gap=arcgap,upper_descent_time=upperT,upper_descent_drop=upper_drop,upper_descent_rate=upper_rate,upper_descent_max_root=upper_root,descent_time=T,descent_rate=cdec,descent_drop=adrop,descent_max_root=de['root'],constant_rate_optimum=de['opt'],old_drop=old_drop,late_derivative=outer_rate,outer_rebound=outer_rebound,inner_rebound=inner_rate*dt_late,sector_width=outer_th-thlo,failure=fail,moment_failure=momfail,b0_failure=b0fail,normalized_terminal=tb).items()})
result={'checks':checks,'witness':vals,'method':'80-digit evaluation of explicit comparison inequalities and exact polynomial antiderivative; no training simulations.'}
Path('outputs/SWINGBY_STRENGTHENING_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
