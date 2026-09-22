from pathlib import Path
import mpmath as m, sympy as sp, json
m.mp.dps=80
s=m.mpf('.04');h=m.mpf('1e10');eps=m.sqrt(s/h);alo=m.sqrt(m.mpf('.249')*s);ahi=m.sqrt(m.mpf('.251')*s)
H=m.mpf('3.5');nu=m.mpf('.1');a=m.mpf('1e-10');mu=m.mpf('1e-5');sig=m.mpf('1e-11');N=2000
P=30*(a+(mu+a)**2);Bb=m.sqrt(s)+P*H
W=(2*eps+10*P*H/alo)*m.exp((Bb+m.mpf('.02'))*H);q=H*(2*W+3*P/alo);Cc=3*q+(H+1)*W+P*H
M=4*q+2*W;V=H*(M+W+P);JD=m.exp(V)*H*(W+P)/alo;JB=m.exp(V)*H*P/alo;JP=H*(1+JD)*W+2*m.exp(V)*H*P/alo
Gc=m.expm1(M*H)+m.sqrt(2)*m.exp(2*M*H)*H*(JP+(1+JD)*W+P*m.exp(V)/alo)
K=m.mpf('.251')*s;ef1=Cc*m.sqrt(K)+m.sqrt(2)*P*H;ef2=H*(W+P)*m.sqrt(K)+Bb*P*H
Plo=m.sqrt(s)*m.mpf('.077')/m.sqrt(m.mpf('.251'));Phi=m.sqrt(s)*m.mpf('.082')/m.sqrt(m.mpf('.249'));C0=s*m.mpf('1e-4')
Aa=m.sqrt(m.mpf('.75'));Ab=m.sqrt(m.mpf('.9'));As=m.sqrt(m.mpf('.7'))
raw=Plo**2*(Ab-Aa)*(Ab+Aa-2*ahi)/2-C0*Phi*(Ab-Aa)
La=Phi*(Aa-alo)+C0;Lb=Phi*(Ab-alo)+C0
charge=(La+Lb)*ef1+ef1**2+2*ef2+ef2**2
axis_gap=raw-charge;theta=m.asin(m.mpf('2e-5'));center_gap=axis_gap-m.mpf('.8')*theta
Smax=1-nu/2+(m.sqrt(s)+H*((1+Bb)*W+P))**2+Bb**2
mass=m.mpf('.7')*m.mpf('.8')*(1-Gc)**2-m.sqrt(s)*JP
ang=m.atan(ahi/As*(1+m.sqrt(2)*JD)/(1-Gc-m.sqrt(2)*ahi*JP/As))
tb=m.log(9*(1-alo**2)/alo**2)/2
ell=m.log(10400);bh=m.sqrt(ell/(4*h))+4*ell/(3*h);radnet=m.sqrt(2)*s*bh
radprobe=radnet+(s+s/4)*m.mpf('.5e-6');radstrong=max(radprobe,radnet+(s+s/4)*2*a)
radweak=radnet+(s+s/4)*2*a/mu
# Initial derivative: worst lower c, L and conservative upper residual factors.
gam=1-s/4;c=m.mpf('1.49e-5');L=m.mpf('.003');tau=m.mpf('1e-8');Searly=m.mpf('.04000001');df=m.mpf('9.02e-7')
Jstrong=2*gam**2*c*L-(L+c*Searly)*(2*gam*df+df**2)-c*m.mpf('.42')*m.mpf('.41')*tau*(gam+df)**2
Jstrong_weighted=(1-a)**2*Jstrong/2
cw=m.mpf('.999');Lw=m.mpf('.0095');dfw=m.mpf('2e-6')
Jweak=2*gam**2*cw*Lw-(Lw+cw*Searly)*(2*gam*dfw+dfw**2)-cw*m.mpf('.42')*m.mpf('.41')*tau*(gam+dfw)**2
normalx=8*m.sqrt(m.mpf('.249'));tail=2*m.exp(-normalx**2/2)/(normalx*m.sqrt(2*m.pi))+8*m.sqrt(2)/(m.pi*m.sqrt(h))
fail=12*m.exp(-h*m.mpf('1e-8')/2)+tail+N*m.exp(-a*a/(2*sig*sig))+m.mpf('.005')
checks={
 'Bbar':Bb<m.mpf('.3'),'W':W<min(m.mpf('.01'),nu/16),'q':q<nu/8,'Eg':3*P/alo<nu/16,'Sstop':Smax<2,
 'core_purity':(1-Gc)/m.sqrt(2)>a*m.exp(V),'core_mass':mass>m.mpf('.55'),'core_angle':ang<m.mpf('.13'),
 'axis_gap':axis_gap>m.mpf('5.193e-5'),'center_gap':center_gap>m.mpf('3e-5'),'weak_fit':K+ef2<m.mpf('.011'),'time':tb<H,
 'radial_grid':radnet<m.mpf('8.61e-7'),'radial_strong':radstrong<m.mpf('9e-7'),'radial_weak':radweak<m.mpf('1.9e-6'),
 'initial_derivative':Jstrong_weighted>m.mpf('4e-8'),'weak_initial_nonnegative':Jweak>0,'failure':fail<m.mpf('.1'),
 'spatial_modulus':m.mpf('.029')*3+m.mpf('1.011')*(Bb*m.sqrt(2)+m.mpf('2.5e-5'))<m.mpf('.4'),
}
# Exact algebra, independent of numeric checks.
x,th,zz=sp.symbols('x th zz');kk=1-x*(1+th)
checks['radial_identity']=sp.expand(kk*(1+th)-zz**2-(1-x+(1-2*x)*th-x*th**2-zz**2))==0
z1,z2,p0,aa,ab,c0=sp.symbols('z1 z2 p0 aa ab c0')
checks['coherent_secant']=sp.expand(((p0*(ab-aa)+c0)**2-c0**2)/2-(c0*p0*(ab-aa)+p0**2*(ab-aa)**2/2))==0
assert all(checks.values()),checks
vals={k:m.nstr(v,30) for k,v in dict(s0=s,h=h,epsilon=eps,A0_lower=alo,A0_upper=ahi,H_normalized=H,physical_horizon=2*H,mu2=mu,sigma=sig,sigma_over_epsilon=sig/eps,P=P,Wbar=W,q_star=q,C_c=Cc,J_D=JD,J_B=JB,J_p=JP,Gamma_core=Gc,core_mass=mass,core_angle=ang,axis_raw=raw,axis_charge=charge,axis_gap=axis_gap,center_gap=center_gap,initial_radial_net=radnet,initial_descent_margin=Jstrong_weighted,total_failure_bound=fail).items()}
result={'checks':checks,'witness':vals,'method':'Exact symbolic identities and 80-digit evaluation of deterministic inequalities; no training simulations or mechanism-discovery experiments.'}
Path('outputs/SWINGBY_COHERENT_ALGEBRA_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
