"""Symbolic checks of the new identities; no training simulations."""
import sympy as s
x=s.Matrix(s.symbols('x1 x2',real=True))
v=[s.Matrix(s.symbols('v11 v12',real=True)),s.Matrix(s.symbols('v21 v22',real=True))]
z=[s.Matrix(s.symbols('z11 z12',real=True)),s.Matrix(s.symbols('z21 z22',real=True))]
alpha=s.symbols('alpha1 alpha2',real=True)
f=sum((alpha[i]*z[i]*(v[i].dot(x)) for i in range(2)),s.zeros(2,1))
fd=sum((alpha[i]*s.Matrix([z[i][0]*v[i][0]*x[0],z[i][1]*v[i][1]*x[1]]) for i in range(2)),s.zeros(2,1))
fo=f-fd
ftt=alpha[0]*s.Matrix([0,z[0][1]*v[0][1]*x[1]])+alpha[1]*s.Matrix([z[1][0]*v[1][0]*x[0],0])
loss=(x-f).dot(x-f)/2
energy_rate=0
for i,q in [(0,1),(1,0)]:
    energy_rate-=v[i][q]*s.diff(loss,v[i][q])+z[i][q]*s.diff(loss,z[i][q])
claimed=-fo.dot(fo)+(x-fd).dot(fo)+2*(x-f).dot(ftt)
assert s.expand(energy_rate-claimed)==0

a,c,D,eta,lam,k=s.symbols('a c D eta lam k',positive=True)
adot=lam*(c-a*(a*a+eta*eta)); cdot=lam*a*(1-a*c); Ddot=-lam*a*a*D
balance=c*c+eta*eta*D*D-a*a-eta*eta
bdot=s.diff(balance,a)*adot+s.diff(balance,c)*cdot+s.diff(balance,D)*Ddot
assert s.expand(bdot+2*lam*a*a*balance)==0
Hdot=cdot-adot*D-a*Ddot
assert s.expand(Hdot.subs(c,a*D)-lam*a*(1-D*D+(a*a+eta*eta)*D))==0
E=(eta*eta*(c*(a+k)-1)**2+(k-eta*eta*D*(a+k))**2)/2
Edot=s.diff(E,a)*adot+s.diff(E,c)*cdot+s.diff(E,D)*Ddot
s0=a*a+eta*eta
q0=eta*(a+k)
claimed0=lam*eta*a*q0*(a*(a+k)*(1-s0)-(1-a*a)+a*k-(1-s0)**2)
assert s.expand(Edot.subs({c:a,D:1})-claimed0)==0

# Normal-form exact remainder, s=-1, including its directional derivative.
aa,bb,cc,dd,zz=s.symbols('aa bb cc dd zz',real=True)
AA=(aa-1)*zz-bb
H=AA*AA/2+cc*zz-dd-zz*zz/2
KE=dd*zz*zz+zz*zz*(bb*AA-cc*zz)/(1+k)+(cc*zz-dd*k)**2/2+eta*eta*bb*bb*zz**4/(2*(1+k)**2)
Eplan=(eta*eta*((aa-1)*zz-bb*k)**2+(-k+eta*eta*(dd*k-cc*zz))**2)/2
num=s.together(Eplan-s.Rational(1,2)-eta*eta*H-eta**4*KE).as_numer_denom()[0]
assert s.rem(s.Poly(s.expand(num),k),s.Poly(k*k-(1-eta*eta*zz*zz),k)).is_zero
print('Verified signed transverse identity, balance, sign barrier, initial derivative, and switching normal form.')
