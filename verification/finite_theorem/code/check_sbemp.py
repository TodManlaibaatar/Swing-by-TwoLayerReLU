from fractions import Fraction as Q
from pathlib import Path
import math, json, re
R=Path(__file__).resolve().parents[1]
e,mu=Q(1,10),Q(2)
B=[(Q(1,2),Q(3,5),Q(1,5),Q(1,4)),(Q(2,5),Q(1,3),Q(7,25),Q(3,10))]
# Entries are (v1,z1,input slope/e,output slope/e); exact finite samples.
D1=[(Q(3),Q(1,5)),(Q(14,5),Q(-1,4)),(Q(16,5),Q(1,10))]
D2=[(e*mu*z,mu*y) for z,y in [(Q(-1,5),Q(1)),(Q(2,5),Q(11,10)),(Q(-1,2),Q(9,10))]]
ave=lambda xs:sum(xs)/len(xs)
pos=lambda x:max(Q(0),x)
E=lambda x:(Q(1,17)*pos(x[1]-x[0]/3),-Q(1,19)*pos(x[1]-x[0]/3))
M=sum(v*z for v,z,s,t in B)
weights=[v*z/M for v,z,s,t in B]
sbar=sum(p*s for p,(_,_,s,t) in zip(weights,B))
tbar=sum(p*t for p,(_,_,s,t) in zip(weights,B))
stbar=sum(p*s*t for p,(_,_,s,t) in zip(weights,B))
a=ave([x*x for x,y in D1])/mu**2
b=ave([x*y for x,y in D1])/(e*mu**2)
c=ave([y*y for x,y in D1])/mu**2
fB=lambda x:(sum(v*z*pos(x[0]+e*s*x[1]) for v,z,s,t in B),e*sum(v*z*t*pos(x[0]+e*s*x[1]) for v,z,s,t in B))
checks=0
for slope in [Q('.14'),Q('.20'),Q('.28'),Q('.32')]:
 for sel in [Q(0),Q(1,2),Q(1)]:
  gate=lambda x: (Q(1) if x[0]+e*slope*x[1]>0 else Q(0) if x[0]+e*slope*x[1]<0 else sel)
  h=ave([(y/mu)**2*gate((x,y)) for x,y in D2])
  g=ave([(x/(e*mu))*(y/mu)*gate((x,y)) for x,y in D2])
  Z=ave([(x/(e*mu))**2*gate((x,y)) for x,y in D2])
  J=lambda r:ave([(y/mu)*pos(x/(e*mu)+r*y/mu)*gate((x,y)) for x,y in D2])
  Jz=lambda r:ave([(x/(e*mu))*pos(x/(e*mu)+r*y/mu)*gate((x,y)) for x,y in D2])
  C=[[2*ave([E(x)[j]*x[k]*gate(x) for x in D1+D2])/mu**2 for k in range(2)] for j in range(2)]
  mat=[[ave([(x[j]-fB(x)[j]-E(x)[j])*x[k]*gate(x) for x in D1+D2]) for k in range(2)] for j in range(2)]
  b12=(1-M)*b+g-sum(v*z*J(s) for v,z,s,t in B)
  b21=b+g-a*M*tbar
  err11=-e**2*M*sbar*b+e**2*Z-e**2*sum(v*z*Jz(s) for v,z,s,t in B)-C[0][0]
  err12=-c*M*sbar-C[0][1]/e
  err21=-e**2*M*stbar*b-e**2*sum(v*z*t*Jz(s) for v,z,s,t in B)-C[1][0]/e
  err22=c-e**2*M*tbar*b-e**2*M*stbar*c-e**2*sum(v*z*t*J(s) for v,z,s,t in B)-C[1][1]
  assert 2*mat[0][0]/mu**2==a*(1-M)+err11
  assert 2*mat[0][1]/(e*mu**2)==b12+err12
  assert 2*mat[1][0]/(e*mu**2)==b21+err21
  assert 2*mat[1][1]/mu**2==h+err22
  checks+=4
  # Exact quotient/weight dynamics, including nonzero ambient force.
  for v,z,s,t in B:
   if s!=slope:continue
   q=z/v; P1,P2=Q(1,1000),Q(-1,1700)
   v2=e*s*v;z2=e*t*z
   vd1=mat[0][0]*z+mat[1][0]*z2-P1
   vd2=mat[0][1]*z+mat[1][1]*z2-P2
   zd1=mat[0][0]*v+mat[0][1]*v2
   zd2=mat[1][0]*v+mat[1][1]*v2
   F=(1-M)*b+g-sum(vj*zj*J(sj) for vj,zj,sj,tj in B)+t*h-a*(1-M)*s
   G=b+g+s*h-a*(M*tbar+(1-M)*t)
   ru=q*(err12+t*err22-s*err11-e**2*s*t*(b21+err21))-2*(P2-e*s*P1)/(e*mu**2*v)
   rw=(err21+s*err22-t*err11-e**2*s*t*(b12+err12))/q
   kval=(a*(1-M)+err11)*(q+1/q)+e**2*s*(b12+err12)/q+e**2*t*q*(b21+err21)-2*P1/(mu**2*v)
   assert 2*(vd2*v-v2*vd1)/(mu**2*e*v*v)==q*F+ru
   assert 2*(zd2*z-z2*zd1)/(mu**2*e*z*z)==G/q+rw
   assert 2*(zd1*v+z*vd1)/(mu**2*z*v)==kval
   checks+=3
# Gaussian values certified by rational alternating-series enclosures.
cl=Q('0.398942280401432677939946059934381868');ch=cl+Q(1,10**36)
pl=Q('3.14159265358979323846264338327950288');ph=pl+Q(1,10**35)
assert 2*ph*cl**2<1<2*pl*ch**2
class I:
 def __init__(self,lo,hi=None):self.lo=Q(lo);self.hi=Q(lo if hi is None else hi)
 def __add__(self,o):
  o=o if isinstance(o,I) else I(o);return I(self.lo+o.lo,self.hi+o.hi)
 __radd__=__add__
 def __neg__(self):return I(-self.hi,-self.lo)
 def __sub__(self,o):return self+-o if isinstance(o,I) else self+(-Q(o))
 def __rsub__(self,o):return -self+o
 def __mul__(self,o):
  o=o if isinstance(o,I) else I(o);v=[self.lo*o.lo,self.lo*o.hi,self.hi*o.lo,self.hi*o.hi];return I(min(v),max(v))
 __rmul__=__mul__
 def __truediv__(self,o):return self*Q(1,o)
def normal(x):
 ep=lambda n:sum((-x*x/2)**k/math.factorial(k) for k in range(n+1))
 ip=lambda n:sum((-1)**k*x**(2*k+1)/(2**k*math.factorial(k)*(2*k+1)) for k in range(n+1))
 return I(cl*ep(13),ch*ep(12)),I(Q(1,2)+cl*ip(13),Q(1,2)+ch*ip(12))
f=lambda s:normal(s)[0]
h=lambda s:normal(s)[1]
K=lambda s:f(s)+s*h(s)
L,U,l,W,sm,sp,tm,tp=map(Q,['.14','.32','.115','.33','.20','.29','.21','.28'])
a=Q('2.25');ms=[Q(0),Q(1,2)]
sec=lambda x:((U-x)*K(L)+(x-L)*K(U))/(U-L)
h0=(h(L)+h(U))/2;dh=h(U)-h(L)
Ch=max(W-L,U-l)*dh/2;Dh=(U-L)*dh/4
lower=lambda xs:I(min(x.lo for x in xs),min(x.hi for x in xs))
upper=lambda xs:I(max(x.lo for x in xs),max(x.hi for x in xs))
v=[lower([(1-m)*(f(L)-a*L)+(l-m*sp)*h(L) for m in ms]),-upper([f(U)-m*K(sm)+W*h(U)-a*(1-m)*U for m in ms]),K(L)-a*(tp+l)/2,a*(tm+W)/2-K(U),lower([(1-m)*(K(sm)-a*sm) for m in ms])+h0*(tm-sm)-Ch,-upper([(1-m)*(sec(sp)-a*sp)+h0*(tp-sp)+Ch+m*Dh for m in ms]),K(sm)-a*tm,a*tp-sec(sp)]
assert all(x.lo>Q('.023') for x in v)
assert Q('.99')*min(x.lo for x in v)>Q('.020')
e=Q('.075');ql2=(1+e*e*L*L)/(1+e*e*W*W);qhi2=(1+e*e*U*U)/(1+e*e*l*l)
assert ql2>Q('.999')**2 and 1/qhi2>Q('.999')**2
body=(R/'work/sbemp-continuation-body.tex').read_text()
labels=re.findall(r'\\label\{([^}]+)\}',body);refs=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',body)
assert len(labels)==len(set(labels)) and set(refs)<=set(labels)
stack=[]
for kind,name in re.findall(r'\\(begin|end)\{([^}]+)\}',body):
 if kind=='begin':stack.append(name)
 else:assert stack and stack.pop()==name
assert not stack
result={'exact_rational_identity_cases':checks,'eight_certified_population_margin_lower_bounds':[float(x.lo) for x in v],'per_equation_error_allowance':.020,'balance_factor_lower_bound_at_e_0075':.999,'status':'Algebra and deterministic design checks only; no training experiments or empirical trajectory certification.'}
(R/'outputs/SWINGBY_EMPIRICAL_TWO_MEAN_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
