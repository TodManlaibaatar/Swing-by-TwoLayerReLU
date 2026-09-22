"""Independent exact-symbolic checks of the smooth field derivatives."""
import sympy as s,json
from pathlib import Path
q=s.symbols('q');N=4;h=3
X=s.Matrix([[2,1],[-1,3],[1,-2],[3,2]])
U=s.Matrix([[s.Rational(1,3),s.Rational(2,5)],[s.Rational(-1,7),s.Rational(3,4)],[s.Rational(2,3),s.Rational(-1,2)]])
W=s.Matrix([[s.Rational(1,4),s.Rational(-1,5)],[s.Rational(2,7),s.Rational(1,3)],[s.Rational(-1,4),s.Rational(2,3)]])
pu=s.Matrix([[1,2],[-2,1],[3,-1]]);pw=s.Matrix([[2,-1],[1,3],[-1,2]])
G=s.Matrix([[1,0,1],[0,1,0],[1,0,0],[1,1,1]])
def had(a,b):return a.multiply_elementwise(b)
def field(u,w):
 act=had(X*u.T,G);e=X-act*w
 return had(e*w.T,G).T*X/N,act.T*e/N
act=had(X*U.T,G);e=X-act*W
Jp=act*pw+had(X*pu.T,G)*W
adj=lambda z:(had(z*W.T,G).T*X/N,act.T*z/N)
A=[]
for i in range(h):
 ai=s.zeros(2)
 for n in range(N):ai+=G[n,i]*e.row(n).T*X.row(n)/N
 A.append(ai)
Hpu=s.Matrix.vstack(*[(A[i].T*pw.row(i).T).T for i in range(h)])
Hpw=s.Matrix.vstack(*[(A[i]*pu.row(i).T).T for i in range(h)])
a,b=adj(Jp);DF=(Hpu-a,Hpw-b)
gvx=had(X*pu.T,G);DJJ=(had(Jp*pw.T,G).T*X/N,gvx.T*Jp/N);a,b=adj(2*gvx*pw);D2F=(-2*DJJ[0]-a,-2*DJJ[1]-b)
poly=field(U+q*pu,W+q*pw)
for k in range(2):
 assert (poly[k].diff(q).subs(q,0)-DF[k])==s.zeros(h,2)
 assert (poly[k].diff(q,2).subs(q,0)-D2F[k])==s.zeros(h,2)
# The Hermite midpoint identities are independent of the reference integrator.
t=s.symbols('t');a0,a1,f0,f1,H=s.symbols('a0 a1 f0 f1 H',nonzero=True)
z=t/H;herm=(2*z**3-3*z**2+1)*a0+(-2*z**3+3*z**2)*a1+(z**3-2*z**2+z)*H*f0+(z**3-z**2)*H*f1
expected=[(a0+a1)/2+H*(f0-f1)/8,3*(a1-a0)/(2*H)-(f0+f1)/4,(f1-f0)/H,12*(a0-a1)/H**3+6*(f0+f1)/H**2]
for j in range(4):assert s.simplify(s.diff(herm,t,j).subs(t,H/2)-expected[j])==0
result=dict(status='PASS',exact_checks=['DF=-J*J+residual Hessian, both parameter blocks','D2F[p,p]=-2 DJ[p]*Jp-J*D2f[p,p], both blocks','All four cubic-Hermite midpoint identities'])
(Path(__file__).resolve().parent.parent/'LOCAL_FORMULA_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
