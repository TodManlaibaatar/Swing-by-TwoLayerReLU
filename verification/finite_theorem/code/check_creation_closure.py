import sympy as s
import math

# Exact projected-gradient identity for one neuron; sums preserve it.
x11,x12,x21,x22,p1,p2,a1,a2,w1,w2,e11,e12,e21,e22=s.symbols(
    'x11 x12 x21 x22 p1 p2 a1 a2 w1 w2 e11 e12 e21 e22')
x=[s.Matrix([x11,x12]),s.Matrix([x21,x22])]
ee=[s.Matrix([e11,e12]),s.Matrix([e21,e22])]
p=[p1,p2]; aa=[a1,a2]; w=s.Matrix([w1,w2])
D=(p1*p1+p2*p2)/2
ci=(p1*a1+p2*a2)/(2*D)
for gg in ([1,1],[1,0],[0,1]):
    ell=(p1*gg[0]*x[0]+p2*gg[1]*x[1])/(2*D)
    kk=[gg[j]*ell.dot(x[j]) for j in range(2)]
    d=(p1*ee[0]+p2*ee[1])/(2*D)
    R=[ee[j]-p[j]*d for j in range(2)]
    lhs=ci*(ee[0]*a1+ee[1]*a2)/2+w*sum(w.dot(ee[j])*kk[j] for j in range(2))/2
    K=D*(ci*ci*s.eye(2)+ell.dot(ell)*(w*w.T))
    rem=ci*sum((R[j]*(aa[j]-ci*p[j]) for j in range(2)),s.zeros(2,1))/2
    rem+=w*sum(w.dot(R[j])*(kk[j]-ell.dot(ell)*p[j]) for j in range(2))/2
    assert all(s.factor(v)==0 for v in lhs-K*d-rem)

# Middle numerator factorization and shared-ratio influence remainder.
xx,yy,pp,tau,m,g=s.symbols('x y p tau m g')
old=((1-tau*tau)*xx*yy+tau*(xx*xx-yy*yy)-m*tau*pp*(xx-tau*yy))*g
new=((tau*xx+yy)*(xx-tau*yy)-m*tau*pp*(xx-tau*yy))*g
assert s.expand(old-new)==0
U,V,dd,hh=s.symbols('U V dd hh')
assert s.factor(U/(1+dd)-V/(1+hh)-(U-V-dd*U/(1+dd)+hh*V/(1+hh)))==0

# Exact leading Gaussian integral and elementary tail inequality ingredients.
z=s.symbols('z',positive=True)
phi=s.exp(-z*z/2)/s.sqrt(2*s.pi)
assert s.simplify(s.diff(1/s.sqrt(2*s.pi)-phi,z)-z*phi)==0

def j_constants(r,theta,z):
    d0=r/4
    cn=1/math.sqrt(2*math.pi); vn=math.sqrt(.5-cn*cn)
    F=math.exp(-z*z/2)*cn-cn*.5*math.erfc(z/math.sqrt(2))
    Cl=math.sqrt(3)/4*(1+1/r+1/(r*math.sin(theta)))
    Cr=Cl+3/r
    H=math.sqrt(.5+d0*d0+z*z*d0*d0*(r*r/2+d0*d0))
    Ch=.5*(vn*(1+z*d0)+cn*z*((z+cn)*math.sqrt(2/math.pi)+z*d0)
           +vn*math.sqrt(1+z*z*(r*r+d0*d0)))
    C=Cr*H+Ch
    cut=min(d0,math.sin(theta)*F/(4*C))
    assert cut>0 and F>=.5*cn*math.exp(-z*z/2)
    return C,F,cut
print('Passed exact projected-gradient identity (three gate configurations), middle factorization, ratio remainder, and Gaussian leading-integral derivative.')
print('Analytic sufficient region only; no quadrature or training runs:')
for r in (.5,2/3,.8):
    C,F,cut=j_constants(r,math.pi/18,2.16)
    print(f'r={r:.6g}, theta=10 deg, z=2.16: C={C:.6g}, F={F:.6g}, sigma/mu1 <= {cut:.8g}')
