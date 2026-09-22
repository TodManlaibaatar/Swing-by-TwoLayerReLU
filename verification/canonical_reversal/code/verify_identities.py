import sympy as s,json
from pathlib import Path
r=s.symbols('r',real=True)
a=s.symbols('a0:2');b=s.symbols('b0:2');c=s.symbols('c0:2');d=s.symbols('d0:2')
u=[s.Matrix([a[i],b[i]]) for i in range(2)];w=[s.Matrix([c[i],d[i]]) for i in range(2)];v=s.Matrix([r,-1]);M=sum((w[i]*u[i].T for i in range(2)),s.zeros(2));Y=(M*v)[0]-r;ell=s.Matrix([Y,1]);Phi=Y**2/2+(M*v)[1]
checks={}
for i in range(2):
 checks[f'gradient_u_{i}']=all(s.expand(s.diff(Phi,u[i][j])-v[j]*(w[i].dot(ell)))==0 for j in range(2))
 checks[f'gradient_w_{i}']=all(s.expand(s.diff(Phi,w[i][j])-ell[j]*(u[i].dot(v)))==0 for j in range(2))
x1,x2=s.symbols('x1 x2');alpha=s.symbols('alpha0:2');F=sum(alpha[i]*c[i]*(a[i]*x1+b[i]*x2) for i in range(2));L=sum(alpha[i]*c[i]*(d[i]-b[i])*x2 for i in range(2));Z=-sum(alpha[i]*c[i]*a[i]*x1 for i in range(2));H=sum(alpha[i]*c[i]*d[i]*x2 for i in range(2));FO=s.symbols('FO');checks['weak_feature_identity']=s.expand(H-F-L-Z)==0
checks['signed_source_expansion']=s.expand((F+FO-x1)*H-(F**2+FO*F-x1*F+(F+FO-x1)*(L+Z)))==0
assert all(checks.values())
Path('outputs/crossing-closure/EXACT_IDENTITY_CHECKS.json').write_text(json.dumps(checks,indent=2)+'\n')
print(checks)
