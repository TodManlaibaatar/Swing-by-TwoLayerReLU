"""Symbolic identity checks only; no training experiments."""
import sympy as s

a,t,S0=s.symbols('a t S0', positive=True)
v=s.Matrix([a,a])
A=(s.eye(2)-v*v.T)/2
adot=s.simplify((A*v)[0])
assert s.simplify(adot-a*(1-2*a*a)/2)==0
assert s.simplify(4*a*adot-(2*a*a)*(1-2*a*a))==0
St=S0*s.exp(t)/(1-S0+S0*s.exp(t))
assert s.simplify(s.diff(St,t)-St*(1-St))==0
assert s.simplify(St.subs(t,s.log((1-S0)/S0))-s.Rational(1,2))==0

c,q,r11,r12,r21,r22=s.symbols('c q r11 r12 r21 r22', real=True)
u=s.Matrix([c,q]); ua=s.Matrix([-q,c]); R=s.Matrix([[r11,r12],[r21,r22]])
proxy=(u.T*R*ua)[0]
claimed=c*q*(r22-r11)+r12*(c*c-q*q)-(r21-r12)*q*q
assert s.expand(proxy-claimed)==0

v1,v2,z1,z2,p1,p2=s.symbols('v1 v2 z1 z2 p1 p2',real=True)
v=s.Matrix([v1,v2]);z=s.Matrix([z1,z2]);P=s.Matrix([p1,p2])
vdot=R.T*z-P;zdot=R*v
Mdot=zdot*v.T+z*vdot.T
assert s.expand(Mdot[1,1]-(r22*(v2*v2+z2*z2)+r21*v1*v2+r12*z1*z2-z2*p2))==0
print('Verified logistic balanced flow, horizon evaluation, angular proxy, and M22 derivative.')
