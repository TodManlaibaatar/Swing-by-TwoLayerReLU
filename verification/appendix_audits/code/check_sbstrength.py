import mpmath as m,json
from pathlib import Path
m.mp.dps=60
D=m.mpf

def ledger(h='1e7',sig='1e-10',mu='1e-5',aa='.71',ab='.94',qwin='.002',cwin='.001'):
 h=D(h);sig=D(sig);mu=D(mu);aa=D(aa);ab=D(ab);qw=D(qwin);cw=D(cwin)
 s=D('.04');ep=m.sqrt(s/h);lo=m.sqrt((D('.25')-qw)*s);hi=m.sqrt((D('.25')+qw)*s);a=10*sig;H=D('3.8');nu=1-ab
 P=30*(a+(mu+a)**2);B=m.sqrt(s)+P*H;cap=D('.001');b0=D('1.25')*ep
 Z=(b0+P*(1+6*B/lo)*H)*m.exp((B*B+D('.01'))*H)
 I=m.atanh(m.sqrt(ab))-m.atanh(lo);JJ=m.log((1-lo**2)/(1-ab))/2
 el=B*Z*m.exp(2*cap)*I/m.sqrt(ab)+2*P*H*m.exp(cap)/lo
 q=H*max(el+el**2+Z**2+2*P/lo,el+B*Z+2*P/lo)
 W=Z+el;Cc=3*q+(H+1)*el+P*H;DD=Z*m.exp(2*q)*JJ+P*H
 M=4*q+2*W;V=H*(M+W+P);JD=m.exp(V)*H*(W+P)/lo;JB=m.exp(V)*H*P/lo;JP=H*(1+JD)*W+2*m.exp(V)*H*P/lo
 GC=m.expm1(M*H)+m.sqrt(2)*m.exp(2*M*H)*H*(JP+(1+JD)*W+P*m.exp(V)/lo)
 spec=D('.7');share=(D('.204577471545947667884441881686257181')-D('.002'))/(D('.25')+qw)
 mass=spec*share*(1-GC)**2-m.sqrt(s)*JP
 angle=m.atan(hi/m.sqrt(spec)*(1+m.sqrt(2)*JD)/(1-GC-m.sqrt(2)*hi*JP/m.sqrt(spec)))
 K=(D('.25')+qw)*s;ef1=Cc*m.sqrt(K)+m.sqrt(2)*P*H;ef2=DD*m.sqrt(K)+B*P*H
 Pl=m.sqrt(s)*(1/(4*m.pi)-D('.002'))/m.sqrt(D('.25')+qw);Ph=m.sqrt(s)*(1/(4*m.pi)+D('.002'))/m.sqrt(D('.25')-qw);C=s*cw
 A=m.sqrt(aa);Ab=m.sqrt(ab);raw=Pl**2*(Ab-A)*(Ab+A-2*hi)/2-C*Ph*(Ab-A)
 La=Ph*(A-lo)+C;Lb=Ph*(Ab-lo)+C;charge=(La+Lb)*ef1+ef1**2+2*ef2+ef2**2
 Sstop=1-nu/2+(m.sqrt(s)+H*((1+B)*W+P))**2+B**2
 return locals()
if __name__=='__main__':
 v=ledger()
 for k in ['ep','P','B','b0','Z','el','W','q','Cc','DD','M','V','JD','JB','JP','GC','share','mass','angle','Sstop','raw','charge']:
  print(k,m.nstr(v[k],14))
 print('gap',m.nstr(v['raw']-v['charge'],16))

def descent(h,theta_lo,theta_hi,a,mu,s=D('.04'),tol=D('.002'),n_grid=11):
 h=D(h);theta_lo=D(theta_lo);theta_hi=D(theta_hi);a=D(a);mu=D(mu)
 gamma=1-s/4;ell=m.log(800*(n_grid+2))
 net=m.sqrt(2)*s*(m.sqrt(ell/(4*h))+4*ell/(3*h))
 train_error=net+(s+s/4)*2*a/mu
 lam=(gamma+train_error)*((1+a)**2+(mu+a)**2)/2
 d0=net+(s+s/4)*max((theta_hi-theta_lo)/(2*(n_grid-1)),2*a)
 c=m.sin(theta_lo)-2*a
 L0=s*(1/(4*m.pi)-tol)*m.cos(theta_hi)-2*a*s
 def margin(t,weak=False):
  x=m.exp(lam*t);F=s*(x*x-1);S=s*x*x;dm=4*s*x*(x-1)
  dd=d0+F;cc=c;LL=L0-F
  if weak:
   dd=net+(s+s/4)*2*a/mu+F
   cc=m.cos(theta_hi)-2*a/mu
   LL=s*(D('.25')-tol)-s*(theta_hi+2*a/mu)-F
  rr=2*gamma*dd+dd*dd
  return 2*gamma*gamma*cc*LL-(LL+cc*S)*rr-cc*dm*(gamma+dd)**2
 root=m.findroot(lambda t:margin(t),(D('0'),D('.003')))
 opt=m.findroot(lambda t:margin(t)+t*m.diff(margin,t),(root/3,root*D('.7')))
 return locals()

def monotone(v,theta_max='4.5e-5',Apre2='.69'):
 th=D(theta_max);Apre=m.sqrt(D(Apre2));Ab=m.sqrt(v['ab']);P=v['P'];B=v['B'];l=v['el'];z=v['Z'];q=v['q'];M=v['M'];kmin=1-v['ab']-M
 Tminus=Apre*v['Pl']-3*q*m.sqrt(v['K'])-P*v['H']
 Tplus=Ab*v['Ph']+3*q*m.sqrt(v['K'])+P*v['H']
 f1min=v['Pl']*(Apre-v['hi'])-v['C']-v['ef1']-3*th
 f1prime=kmin*m.cos(th)*Tminus-th*(l+m.sqrt(2)*(l+B*z+P))-2*m.sqrt(2)*P
 f2prime=z*(Tplus+th)+th*B+th*B*(l+B*z+P)+P*(m.sqrt(2)+B)
 Eprime=(f1min*f1prime-D('1.011')*f2prime)/2
 return locals()
