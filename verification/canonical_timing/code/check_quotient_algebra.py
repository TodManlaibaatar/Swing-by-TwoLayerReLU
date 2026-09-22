from pathlib import Path
import sympy as s,json
p=Path(__file__).resolve().parent.parent
u=[s.Matrix(s.symbols(f'u{i}_0 u{i}_1')) for i in range(3)];w=[s.Matrix(s.symbols(f'w{i}_0 w{i}_1')) for i in range(3)];A=s.Matrix(2,2,s.symbols('a:4'))
U=sum((v*v.T for v in u),s.zeros(2));W=sum((v*v.T for v in w),s.zeros(2));M=sum((b*a.T for a,b in zip(u,w)),s.zeros(2))
du=[-A.T*v for v in w];dw=[-A*v for v in u]
dM=sum((b*a.T+d*c.T for a,b,c,d in zip(u,dw,du,w)),s.zeros(2))
dU=sum((b*a.T+a*b.T for a,b in zip(u,du)),s.zeros(2));dW=sum((b*a.T+a*b.T for a,b in zip(w,dw)),s.zeros(2))
for q in [dM+A*U+W*A,dU+A.T*M+M.T*A,dW+A*M.T+M*A.T]:assert q.applyfunc(s.expand)==s.zeros(2)
a,b=s.symbols('a b',positive=True);Y=s.Matrix([[a,a],[b,-b]]);Q=s.Matrix([[s.Rational(4,5),-s.Rational(3,5)],[s.Rational(3,5),s.Rational(4,5)]]);Yp=Y*Q
assert Q.T*Q==s.eye(2);assert (Yp*Yp.T-Y*Y.T).applyfunc(s.expand)==s.zeros(2)
assert Yp==s.Matrix([[7*a/5,a/5],[b/5,-7*b/5]])
report=dict(status='PASS',checks=['All three signed moment equations','Orthogonal same-moment balanced counterexample','Different active-probe rank-one contribution'],residual_convention='rho=f-x; A=mean(g*rho*x^T)')
(p/'QUOTIENT_ALGEBRA_CHECKS.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
