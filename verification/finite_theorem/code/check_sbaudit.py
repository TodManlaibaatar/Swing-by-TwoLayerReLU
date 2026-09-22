"""Exact algebra and conservative constants only; no trajectories or experiments."""
from pathlib import Path
import sympy as s
import json

checks = {}
eta,z,k,sgn = s.symbols('eta z k s', real=True)
A,T,Ad,Td = s.symbols('A T Ad Td', real=True)
exact = (eta**2*A**2 + (eta**2*T-sgn*k)**2)/2
stated = s.Rational(1,2)+eta**2*(A**2/2-sgn*k*T-z**2/2)+eta**4*T**2/2
delta = s.expand(exact-stated).subs(sgn**2,1).subs(k**2,1-eta**2*z**2)
assert s.simplify(delta) == 0
checks['exact_probe_error'] = True
rate = s.diff(exact,A)*Ad+s.diff(exact,T)*Td
assert s.simplify(rate/eta**2-(A*Ad+(eta**2*T-sgn*k)*Td)) == 0
checks['exact_probe_derivative'] = True

a11,a12,a21,a22 = s.symbols('a11 a12 a21 a22')
u1,u2,w1,w2,p1,p2 = s.symbols('u1 u2 w1 w2 p1 p2')
v=s.Matrix([u1,u2]); w=s.Matrix([w1,w2]); P=s.Matrix([p1,p2])
R=s.Matrix([[a11,a12],[a21,a22]])
assert s.simplify((R*v)*v.T+w*(R.T*w-P).T-(R*v*v.T+w*w.T*R-w*P.T)) == s.zeros(2)
checks['active_matrix_rate'] = True

# Initial target pairing, at one neuron with both selectors active.
x=s.Matrix(s.symbols('x1 x2 x3')); xi=s.Matrix(s.symbols('xi1 xi2 xi3'))
u=s.Matrix(s.symbols('u1 u2 u3'))
K=(u.dot(xi)*u.dot(x))*s.eye(3)+xi.dot(x)*(u*u.T)
assert s.expand((xi.T*K*x)[0]-2*xi.dot(x)*u.dot(xi)*u.dot(x)) == 0
checks['balanced_initial_kernel_pairing'] = True

# Active-side gate-exit limit, full background retained.
q,qd=s.symbols('q qd'); F=s.Matrix(s.symbols('F1 F2 F3'))
Fd=s.Matrix(s.symbols('Fd1 Fd2 Fd3')); w=s.Matrix(s.symbols('w1 w2 w3'))
wd=s.Matrix(s.symbols('wd1 wd2 wd3'))
expr=(F+w*q-xi).dot(Fd+wd*q+w*qd)
assert s.expand(expr.subs(q,0)-(F-xi).dot(Fd)-(F-xi).dot(w)*qd) == 0
checks['gate_exit_limit'] = True
phi,psi,phid,rad,amp=s.symbols('phi psi phid rad amp',real=True)
normal=s.Matrix([s.sin(phi),-s.cos(phi)])
tangent=s.Matrix([-s.sin(phi),s.cos(phi)])
assert s.trigsimp(rad*phid*tangent.dot(normal)+rad*phid) == 0
assert s.trigsimp(s.Matrix([amp*s.cos(psi),amp*s.sin(psi)]).dot(normal)-amp*s.sin(phi-psi)) == 0
checks['angular_exit_sign'] = True

assert s.Rational(3,4)-s.Rational(1,16) == s.Rational(11,16)
assert 2*s.sqrt(s.Rational(6,1536)) == s.Rational(1,8)
assert s.Rational(8,64) == s.Rational(1,8)
assert s.Rational(6,24) == s.Rational(1,4)
assert 1-s.Rational(1,4)-s.Rational(1,8)-s.Rational(1,4) == s.Rational(3,8)
checks['initial_descent_budget_constants'] = True
c=s.symbols('c',real=True)
angular=s.sqrt(1-c**2)+(s.pi-s.acos(c))*c
assert s.simplify(s.diff(angular,c)-(s.pi-s.acos(c))) == 0
assert angular.subs(c,0) == 1
checks['directional_noise_monotonicity_identity'] = True
checks['no_numerical_experiments'] = True
Path('outputs/SWINGBY_STAGE0_ALGEBRA_CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n')
print(json.dumps(checks,indent=2))
