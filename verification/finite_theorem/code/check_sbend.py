"""Symbolic identities and rational proof budgets only. No trajectories or experiments."""
from pathlib import Path
import sympy as s
import json
C={}
La,Lb,k,Xa,Xb,Pa,Pb=s.symbols('La Lb k Xa Xb Pa Pb', real=True)
a=((k-Lb)**2+Xb**2+Pb**2-(k-La)**2-Xa**2-Pa**2)/2
b=(La-Lb)*(k-(La+Lb)/2)+(Xb**2-Xa**2+Pb**2-Pa**2)/2
assert s.expand(a-b)==0;C['all_neuron_leakage_secant']=True
assert s.Rational(47,48)-s.Rational(25,32)==s.Rational(19,96)
assert s.Rational(19,96)-s.Rational(7,96)==s.Rational(1,8)
C['gate_removal_signed_background_budget']=True
assert s.Rational(5,8)-s.Rational(5,16)-s.Rational(1,16)==s.Rational(1,4)
assert 36+6*s.sqrt(2)<45
C['explicit_initial_clock_budget']=True
q,a,b,c,d,p,z=s.symbols('q a b c d p z', real=True)
ph=q*(b*s.cos(z)+d*s.sin(z))-p
ps=(c*s.cos(z)-a*s.sin(z))/q
assert s.expand(ps-ph+(a/q+q*d)*s.sin(z)-(c/q-q*b)*s.cos(z)-p)==0
C['exact_polar_lag']=True
# Matrix invariant, arbitrary two active neurons and two transverse coordinates.
a=s.Matrix(s.symbols('a0:2'));c=s.Matrix(s.symbols('c0:2'))
D=s.Matrix(2,2,s.symbols('D0:4'))
m=a.dot(c);ap=(1-m)*c-D.T*D*a;cp=(1-m)*a;Dp=-D*a*a.T
invdot=cp*c.T+c*cp.T+Dp.T*D+D.T*Dp-ap*a.T-a*ap.T
assert s.simplify(invdot)==s.zeros(2)
C['rank_one_reference_full_gram_invariant']=True
A,A0=s.symbols('A A0',positive=True)
dec=s.sqrt((1-A*A)/(1-A0*A0))
assert s.simplify(s.diff(dec,A)*A*(1-A*A)+A*A*dec)==0
C['reference_transverse_decay']=True
# Exact probe factorization using v^T v=1.
r,k,eta,sgn,dec=s.symbols('r k eta s dec',real=True)
bv=s.Matrix(s.symbols('b0:2'));BB2=s.Matrix(s.symbols('C0:2'))
raw=A*r*dec*bv+sgn*k*(BB2-(1-dec)*bv*sgn*eta)
claimed=sgn*k*(BB2-bv*sgn*eta)+bv*dec*(A*r+eta*k)
assert s.simplify(s.expand(raw-claimed).subs(sgn**2,1))==s.zeros(2,1)
C['all_neuron_probe_reference_factorization']=True
assert s.Rational(9,16)-s.Rational(1,16)-s.Rational(1,16)==s.Rational(7,16)
assert s.Rational(7,16)-s.Rational(8,128)==s.Rational(3,8)
assert s.Rational(3,8)-s.Rational(32,1024)>s.Rational(1,8)
assert s.Rational(3,4)*s.Rational(63,64)**2>s.Rational(2,3)
assert s.Rational(2,3)-s.Rational(4,32)>s.Rational(1,2)
C['learned_mass_and_rebound_reserves']=True
assert s.Rational(40,160)+s.Rational(40*25,16*256)<1
assert 4*s.sqrt(s.Rational(2,8192))==s.Rational(1,16)
assert sum([s.Rational(1,32),s.Rational(1,16),s.Rational(1,32),s.Rational(1,16)])<s.Rational(1,4)
C['data_and_geometry_probability_constants']=True
assert s.Rational(1,5)-s.Rational(1,20)>s.Rational(1,8)
assert s.Rational(1,40)+s.Rational(6,240)==s.Rational(1,20)
C['learned_window_lag_reserve']=True
# Bound rank-one endpoint times without numerical simulation.
assert -s.Rational(19,100)*8+s.Rational(9,10)<0
assert s.Rational(9,16)*s.sqrt(8)-s.Rational(1,8)>1
assert s.sqrt(s.Rational(2,512))*s.Rational(9,8)<=s.Rational(1,8)
C['strict_snapshot_time_order_and_transverse_drop']=True
C['no_experiments_run']=True
Path('outputs/SWINGBY_ENDPOINT_ALGEBRA_CHECKS.json').write_text(json.dumps(C,indent=2)+'\n')
print(json.dumps(C,indent=2))
