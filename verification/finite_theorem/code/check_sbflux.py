from fractions import Fraction as F
from pathlib import Path
import sympy as s
import json,re
R=Path(__file__).resolve().parents[1]
x,y,a,b,c,d,p1,p2,delta1,delta2=s.symbols('x y a b c d p1 p2 delta1 delta2',real=True)
vd1=a*(x-delta1)+c*(y-delta2)-p1
vd2=b*(x-delta1)+d*(y-delta2)-p2
expr=(a+d)*x*y+b*x*x+c*y*y-y*(a*delta1+c*delta2+p1)-x*(b*delta1+d*delta2+p2)
assert s.expand(vd1*y+vd2*x-expr)==0
r,eta=s.symbols('r eta',positive=True)
A=s.Matrix([[-s.Rational(9,2)*r*r*eta*eta,1-2*r*r*eta],[1-s.Rational(9,2)*r*r*eta,-2*r*r]])
v=s.Matrix([r*eta,r]);dot=A.T*v
production=s.simplify((dot[0]*v[1]+v[0]*dot[1])/r**2)
assert s.expand(production-(1+eta**2-s.Rational(13,2)*r**2*eta*(1+eta**2)))==0
assert s.limit(production,eta,0)==1
# Exact static data checks, with signed inputs and outputs and gate endpoints.
data=[(F(-1,5),F(2)),(F(0),F(11,5)),(F(1,4),F(9,5))]
mu,e=F(2),F(1,10)
neurons=[(F(1,2),F(2,3),F(1,7)),(F(-1,3),F(1,2),F(-2,5)),(F(2,5),F(-1,2),F(1,8)),(F(-1,2),F(-1,4),F(1,7))]
# (v1,v2,delta1); z1=v1-delta1. Bound mismatch separately neuronwise,
# avoiding square roots, then verify the Cauchy bound by squaring.
ave=lambda vals:sum(vals)/len(vals)
pos=lambda val:max(F(0),val)
V=sum(pos(v1)**2 for v1,v2,de in neurons)
Q=sum(pos(v1)*pos(v2) for v1,v2,de in neurons)
num=0
for slope in [F(-2),F(-1),F(0),F(1),F(2)]:
 for sel in [F(0),F(1,2),F(1)]:
  gate=lambda xx,yy: F(1) if xx+e*slope*yy>0 else F(0) if xx+e*slope*yy<0 else sel
  f=lambda xx,yy:sum((v1-de)*pos(v1*xx+v2*yy) for v1,v2,de in neurons)
  lhs=pos(ave([f(xx,yy)*yy*gate(xx,yy) for xx,yy in data])/(e*mu**2))
  gp=ave([pos(xx/(e*mu))*(yy/mu)*gate(xx,yy) for xx,yy in data])
  h=ave([(yy/mu)**2*gate(xx,yy) for xx,yy in data])
  mismatch=ave([sum(abs(de)*pos(v1*xx+v2*yy) for v1,v2,de in neurons)*yy*gate(xx,yy) for xx,yy in data])/(e*mu**2)
  assert lhs<=gp*V+h*Q/e+mismatch
  num+=1
body=(R/'work/sbflux-continuation-body.tex').read_text()
labels=re.findall(r'\\label\{([^}]+)\}',body)
refs=re.findall(r'\\(?:ref|eqref)\{([^}]+)\}',body)
assert len(labels)==len(set(labels)) and set(refs)<=set(labels)
stack=[]
for kind,name in re.findall(r'\\(begin|end)\{([^}]+)\}',body):
 if kind=='begin':stack.append(name)
 else:assert stack and stack.pop()==name
assert not stack
result={'symbolic_transport_identity':True,'signed_output_rational_cases':num,'SIM_static_production_identity':str(production),'boundary_limit':1,'labels':len(labels),'scope':'Deterministic algebra checks; no training runs and no initialized-trajectory certification.'}
(R/'outputs/SWINGBY_WEIGHTED_TRANSPORT_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
