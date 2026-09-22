"""Exact algebra checks for the signed-gap continuation; no training runs."""
from fractions import Fraction as F
from pathlib import Path
import hashlib
import json
import sympy as sp

root = Path(__file__).resolve().parents[1]
L,Q,K,U,feedback,S,T = sp.symbols("L Q K U feedback S T", nonzero=True)
g = (L+Q+K-feedback-U)/S
kappa = Q/T-feedback/(2*T)
rhs = L/S-(1-T/S)*Q/T+(1/(2*T)-1/S)*feedback+K/S-U/S
assert sp.simplify(g-kappa-rhs) == 0

lp,qp,cp,up,pp,sp_,tp = sp.symbols("lp qp cp up pp sp tp", nonzero=True)
anis = (lp+qp+cp-up)/sp_-(2*qp+cp-pp)/(2*tp)
tau = tp/sp_
expanded = lp/sp_-(1-tau)*qp/tp-(1-2*tau)*cp/(2*tp)-up/sp_+pp/(2*tp)
assert sp.simplify(anis-expanded) == 0

l,u,m = sp.symbols("l u m", positive=True)
assert sp.simplify(
    m*m - 4*l*u/(l+u)**2*((l+u)*m-l*u)
    - (m-2*l*u/(l+u))**2
) == 0

l1,l2 = F(9,2),F(2)
a,eta,s,low = F(1,2),F(1,100),F(1,10),F(1,10000)
S1 = a*(1+eta**2)
tau = eta**2/(1+eta**2)
S2 = F(1,1000)
tl,th = low**2/(1+low**2),s**2/(1+s**2)
Sl = S2*(th-tau)/(th-tl)
Sh = S2-Sl
assert Sl > 0 and Sh > 0
b=c=a*eta
d=a*eta**2
f=-Sl*low/(1+low**2)-Sh*s/(1+s**2)
h=Sl/(1+low**2)+Sh/(1+s**2)
T1,T2 = tau*S1,tau*S2
assert Sl*low**2/(1+low**2)+Sh*s**2/(1+s**2) == T2
R=l2*(1-d-h)
L1,Q1,C1=l1*(1-a)*a,R*d,-l1*c*c-l2*(b+f)*b
L2,Q2,C2=R*h,F(0),-l2*(b+f)*f
A1=(L1+Q1+C1)/S1-(2*Q1+C1)/(2*T1)
A2=(L2+Q2+C2)/S2-(2*Q2+C2)/(2*T2)
face=A1-A2
assert face == -F(438243077997,6626125026248)
assert face < -F(3,50)
S,T = S1+S2,T1+T2
fo=l1*c*c+l2*(b+f)**2
g=(L1+L2+Q1+Q2+C1+C2)/S
k=(Q1+Q2)/T-fo/(2*T)
assert F(1876,1000) < g-k < F(1878,1000)

# Direct lower bound using only the first output-projection feedback.
zeta_squared=c*c/(S1*T1)
B=(1-2*T/S)*l1*zeta_squared*S1/2-(1-T/S)*R
certificate=(L1+L2)/S+min(F(0),B)
assert certificate > F(137,100)

# All possible theta in [0,1] in the documented sufficient box.
feedback_floor=F(49,50)*F(22,5)*F(4,5)*F(2,5)/2
box=F(147,100)-F(1,20)+min(F(0),feedback_floor-F(39,20))-F(1,20)
assert box == F(687,6250)

body=(root/"work/signed-gap-continuation-body.tex").read_text()
import re
labels=re.findall(r"\\label\{([^}]+)\}",body)
refs=re.findall(r"\\(?:eqref|ref)\{([^}]+)\}",body)
assert len(labels)==len(set(labels))
assert set(refs)<=set(labels), set(refs)-set(labels)
stack=[]
for kind,name in re.findall(r"\\(begin|end)\{([^}]+)\}",body):
    if kind=="begin": stack.append(name)
    else:
        assert stack and stack.pop()==name
assert not stack

out={
    "verification":"exact rational and symbolic algebra; no empirical experiments",
    "source_sha256":hashlib.sha256(Path("/Users/todmanlaibaatar/Downloads/main-12.tex").read_bytes()).hexdigest(),
    "face_gap_exact":str(face),
    "face_gap_decimal":float(face),
    "log_ratio_derivative":float(-2*face),
    "actual_signed_gap":float(g-k),
    "direct_certificate_lower_bound":float(certificate),
    "numerical_box_margin_exact":str(box),
    "symbolic_identities_verified":3,
    "continuation_labels":len(labels)
}
(root/"outputs/SWINGBY_SIGNED_GAP_CHECKS.json").write_text(json.dumps(out,indent=2)+"\n")
print(json.dumps(out,indent=2))
