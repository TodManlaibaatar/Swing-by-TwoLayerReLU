"""Symbolic/rational verification and evaluation of proved data formulas.
No optimization runs or empirical training experiments are performed.
"""
from pathlib import Path
import sympy as s
import math, json, re
from fractions import Fraction as Q

v1,v2,z1,z2,A11,A12,A21,A22,P1,P2 = s.symbols(
    "v1 v2 z1 z2 A11 A12 A21 A22 P1 P2", nonzero=True)
vd1=A11*z1+A21*z2-P1
vd2=A12*z1+A22*z2-P2
zd1=A11*v1+A12*v2
zd2=A21*v1+A22*v2
xi,om,q=v2/v1,z2/z1,z1/v1
assert s.simplify((vd2*v1-v2*vd1)/v1**2 -
    (q*(A12+A22*om-xi*(A11+A21*om))-(P2-xi*P1)/v1))==0
assert s.simplify((zd2*z1-z2*zd1)/z1**2 -
    (1/q*(A21+A22*xi-om*(A11+A12*xi))))==0
assert s.simplify((zd1*v1-z1*vd1)/v1**2 -
    (A11*(1-q*q)+A12*xi-q*q*A21*om+q*P1/v1))==0

mu,sg,U,ph,Ph = s.symbols("mu sg U ph Ph", positive=True)
ts=U*mu/(sg*s.sqrt(1+U*U))
G0=mu*sg/s.sqrt(1+U*U)*ph-ts*sg**2*U/(1+U*U)*ph
G=mu*sg/(1+U*U)**s.Rational(3,2)*ph
assert s.simplify(G0-G)==0
H22=(mu**2+sg**2)*Ph+2*mu*sg*U/s.sqrt(1+U*U)*ph-ts*sg**2*U**2/(1+U*U)*ph
Y=U*(mu**2+sg**2)*Ph+mu*sg*s.sqrt(1+U*U)*ph
assert s.simplify(G+U*H22-Y)==0

# Illustrative evaluation of the closed-form finite-data inequalities.
mu,sg,U,M,n=2.,.15,.0225,.5,100000
delta_s=delta_env=.025
L=math.sqrt(2*math.log(4*n/delta_env))
x=math.log(8/delta_s)
t=U*mu/(sg*math.sqrt(1+U*U))
phi=math.exp(-t*t/2)/math.sqrt(2*math.pi)
Phi=(1+math.erf(t/math.sqrt(2)))/2
G=mu*sg/(1+U*U)**1.5*phi
Y=U*(mu*mu+sg*sg)*Phi+mu*sg*math.sqrt(1+U*U)*phi
VX=sg*sg*(mu*mu+sg*sg)
VY=VX+U*U*(mu**4+6*mu*mu*sg*sg+3*sg**4)
VZ=(math.sqrt(VX)+M*math.sqrt(VY))**2
RX=sg*L*(mu+sg*L)
RY=(mu+sg*L)*(sg*L+U*(mu+sg*L))
RZ=RX+M*RY
pL=min(1,4*math.exp(-L*L/2))
err=lambda V,R: math.sqrt(2*V*x/n)+4*R*x/(3*n)+math.sqrt(V*pL)
assert G-M*Y-err(VZ,RZ)>0
assert mu>sg*L

# Rigorous rational enclosures for the four population face margins.
# The decimal pi enclosure and its induced normal-density enclosure are
# checked algebraically; the alternating series have decreasing terms.
pi_lo=Q('3.14159265358979323846264338327950288')
pi_hi=Q('3.14159265358979323846264338327950289')
c_lo=Q('0.398942280401432677939946059934381868')
c_hi=c_lo+Q(1,10**36)
assert 2*pi_hi*c_lo*c_lo<1<2*pi_lo*c_hi*c_hi
def add(*ivals):
    return (sum(x[0] for x in ivals),sum(x[1] for x in ivals))
def scale(c,ival):
    return (c*ival[0],c*ival[1]) if c>=0 else (c*ival[1],c*ival[0])
def const(c):return (Q(c),Q(c))
def normals(x):
    assert 0<=x<=1
    exp_part=lambda n:sum((-x*x/2)**k/math.factorial(k) for k in range(n+1))
    int_part=lambda n:sum((-1)**k*x**(2*k+1)/(2**k*math.factorial(k)*(2*k+1)) for k in range(n+1))
    density=(c_lo*exp_part(13),c_hi*exp_part(12))
    cdf=(Q(1,2)+c_lo*int_part(13),Q(1,2)+c_hi*int_part(12))
    return density,cdf
Lr,Ur,lr,Wr=map(Q,['0.14','0.32','0.115','0.33'])
m_lo,m_hi,rho2=Q('.49'),Q('.50'),Q('2.25')
fL,PL=normals(Lr)
fU,PU=normals(Ur)
KL=add(fL,scale(Lr,PL))
KU=add(fU,scale(Ur,PU))
face_intervals=[
    add(scale(1-m_hi,fL),scale(lr-m_hi*Ur,PL),const(-rho2*(1-m_lo)*Lr)),
    add(const(rho2*(1-m_hi)*Ur),scale(-1,fU),scale(m_lo,KL),scale(-Wr,PU)),
    add(KL,const(-Q('.33')-rho2*(1-m_lo)*lr)),
    add(const(Q('.24')+rho2*(1-m_hi)*Wr),scale(-1,KU)),
]
claimed=list(map(Q,['.0118','.0062','.0108','.0320']))
assert all(lo>floor for (lo,hi),floor in zip(face_intervals,claimed))
assert min(lo for lo,hi in face_intervals)/2>Q('.003')
assert Q('.24')<rho2*Q('.495')*Q('.24')<Q('.33')
aggregate_intervals=[
    add(scale(Q('.5'),fL),scale(lr-Ur/2,PL),const(-rho2*Lr/2)),
    add(const(rho2*Ur/2),scale(-1,fU),scale(Q('.5'),KL),scale(-Wr,PU)),
    add(KL,const(-rho2*(Q('.29')+lr)/2)),
    add(const(rho2*(Q('.20')+Wr)/2),scale(-1,KU)),
    add(KL,const(-rho2*Q('.20'))),
    add(const(rho2*Q('.29')),scale(-1,KU)),
]
aggregate_claims=list(map(Q,['.0150','.0109','.0172','.0170','.0228','.0733']))
assert all(lo>floor for (lo,hi),floor in zip(aggregate_intervals,aggregate_claims))
assert min(lo for lo,hi in aggregate_intervals)/2>Q('.003')
assert rho2*Lr-fL[0]-Ur*PL[0]<0
assert rho2*Ur-KL[1]>0

root=Path(__file__).resolve().parents[1]
body=(root/"work/c21-coupled-continuation-body.tex").read_text()
labels=re.findall(r"\\label\{([^}]+)\}",body)
refs=re.findall(r"\\(?:ref|eqref)\{([^}]+)\}",body)
assert len(labels)==len(set(labels))
assert set(refs)<=set(labels)
stack=[]
for kind,name in re.findall(r"\\(begin|end)\{([^}]+)\}",body):
    if kind=="begin":stack.append(name)
    else:assert stack and stack.pop()==name
assert not stack
result={
 "verification":"five symbolic identities; rational four-face bounds; fixed-formula evaluation, not experiments",
 "population_face_lower_bounds":[float(lo) for lo,hi in face_intervals],
 "population_face_check":"exact rational alternating-series enclosures; not a finite-sample trajectory certificate",
 "aggregate_holding_six_face_lower_bounds":[float(lo) for lo,hi in aggregate_intervals],
 "aggregate_holding_progress":"0 < raw strong progress <= 0.50; weighted mean output tilt in [0.20, 0.29]",
 "illustrative_data_parameters":{"mu2":mu,"sigma":sg,"U":U,"M_cap":M,"n":n},
 "G_U":G,"Y_U_population":Y,
 "lower_H_min":G-err(VX,RX),
 "lower_H_min_minus_M_Y":G-M*Y-err(VZ,RZ),
 "data_event_probability_lower_bound":1-delta_s-4*n*math.exp(-L*L/2),
 "note":"This data-side example does not certify trajectory charges or the full theorem.",
 "labels":len(labels)
}
(root/"outputs/SWINGBY_C21_COUPLED_CHECKS.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
