import sympy as s

a,c,D,eta,z,k,lam = s.symbols('a c D eta z k lam', positive=True)
ad=lam*(c-a*(a*a+eta*eta))
cd=lam*a*(1-a*c)
Dd=-lam*a*a*D
def dt(f):
    return s.diff(f,a)*ad+s.diff(f,c)*cd+s.diff(f,D)*Dd
balance = c*c+eta*eta*D*D-a*a-eta*eta
assert s.expand(dt(balance)+2*lam*a*a*balance) == 0
A=c*(a*z+k)-z
T=-D*(a*z+k)
assert s.expand(dt(T)-lam*D*(k*a*a+z*(2*a**3+a*eta*eta-c)))==0
E=eta*eta*A*A/2+(k+eta*eta*T)**2/2
assert s.expand(dt(E)/eta**2-A*dt(A)-(k+eta*eta*T)*dt(T))==0

# Independent exact residual of the active matrix, and quartic expansion.
M=s.Matrix([[a*c,-eta*c],[-eta*a*D,eta*eta*D]])
v=s.Matrix([eta*z,-k])
res=M*v-v
assert s.expand(res.dot(res)/2-E)==0
alpha,b,cc,d = a*c,-c,-a*D,D
A0=(alpha-1)*z-b
T0=cc*z-d*k
H=A0*A0/2+cc*z-d-z*z/2
K=d*z*z+z*z*(b*A0-cc*z)/(1+k)+T0*T0/2+eta*eta*b*b*z**4/(2*(1+k)**2)
num=s.together(E-s.Rational(1,2)-eta*eta*H-eta**4*K).as_numer_denom()[0]
assert s.rem(s.Poly(num,k),s.Poly(k*k+eta*eta*z*z-1,k)).as_expr()==0
num=s.together(dt(E)-eta*eta*dt(H)-eta**4*dt(K)).as_numer_denom()[0]
assert s.rem(s.Poly(num,k),s.Poly(k*k+eta*eta*z*z-1,k)).as_expr()==0

# Check the primitive finite-time logistic comparison and all numerical margins.
beta,q0,t=s.symbols('beta q0 t', positive=True)
q=beta/(1+(beta/q0-1)*s.exp(-2*lam*beta*t))
assert s.simplify(s.diff(q,t)-2*lam*q*(beta-q))==0
assert s.Rational(3,4)*(1-3/(4*s.sqrt(2)))>s.Rational(1,3)
assert s.sqrt(s.Rational(15,32))-s.Rational(1,2)>0
assert 15*s.sqrt(7)/128>s.Rational(1,4)
assert s.Rational(97,32)<4
CE=2+12/s.Rational(7,4)+8+s.Rational(1,4)/(2*s.Rational(7,4)**2)
CD=5+50/s.Rational(7,4)+40+s.Rational(10,16)/s.Rational(7,4)**2
assert CE<20 and CD<80
print('Passed exact-flow balance, leakage derivative, probe error, quartic profile and derivative, logistic equation, and all numerical proof margins.')
