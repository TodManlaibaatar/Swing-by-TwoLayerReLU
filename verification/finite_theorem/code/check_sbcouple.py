from pathlib import Path
import sympy as s
import math, json, re
r,eta,a,b,th=s.symbols('r eta a b th', positive=True)
l1=s.Rational(9,2); l2=s.Integer(2)
# Exact static mixed-sign self counterexample.
V=s.Matrix([[1,r*eta],[-s.Rational(2,9),r]])
F1=V[:,0]+V[:,1]*(r*eta)
F2=V[:,1]*r
A=s.Matrix.hstack(l1*(s.Matrix([1,0])-F1),l2*(s.Matrix([0,1])-F2))
dv=A.T*V[:,1]
qdot=s.expand(r*dv[0]+r*eta*dv[1])
assert s.limit(qdot/r**2,eta,0)==1
# Positive strong output still creates skew; matched Q1 self-skew and production.
selfX=(l2-l1)*a*b
selfA=-s.Matrix([[l1*a*a,l2*a*b],[l1*a*b,l2*b*b]])
selfdv=selfA.T*s.Matrix([a,b])
assert s.simplify(b*selfdv[0]+a*selfdv[1]+(l1+l2)*(a*a+b*b)*a*b)==0
assert selfX!=0

def dot(x,y):return sum(a*b for a,b in zip(x,y))
def norm(x):return math.sqrt(dot(x,x))
def plus(x):return max(x,0.)
def add(x,y):return [x[i]+y[i] for i in range(2)]
def scale(a,x):return [a*v for v in x]
def matvec(A,x):return [dot(row,x) for row in A]
def trans(A):return list(zip(*A))

def audit(v,z,data,B,W,E):
    w=1/len(data); n=len(v)
    alpha=[[float(dot(vi,x)>0) for k,x in data] for vi in v]
    acts=[[plus(dot(vi,x)) for k,x in data] for vi in v]
    f=[[sum(z[j][k]*acts[j][m] for j in range(n)) for k in range(2)] for m in range(len(data))]
    A=[[[sum(w*(x[k]-f[m][k])*x[l]*alpha[i][m] for m,(_,x) in enumerate(data)) for l in range(2)] for k in range(2)] for i in range(n)]
    dv=[matvec(trans(A[i]),z[i]) for i in range(n)]
    dz=[matvec(A[i],v[i]) for i in range(n)]
    delta=[[v[i][k]-z[i][k] for k in range(2)] for i in range(n)]
    dd=[add(dv[i],scale(-1,dz[i])) for i in range(n)]
    R=sum(dot(v[i],v[i]) for i in E); D=math.sqrt(sum(dot(delta[i],delta[i]) for i in E))
    I=[i for i in E if v[i][0]>0 and v[i][1]>0]
    J=[i for i in E if v[i][0]*v[i][1]<0]
    Q=sum(v[i][0]*v[i][1] for i in I)
    N4=sum(plus(v[i][0])*plus(-v[i][1]) for i in E)
    N2=sum(plus(-v[i][0])*plus(v[i][1]) for i in E); N=N4+N2
    Qp=sum(v[i][1]*dv[i][0]+v[i][0]*dv[i][1] for i in I)
    Np=-sum(v[i][1]*dv[i][0]+v[i][0]*dv[i][1] for i in J)
    Dp=sum(dot(delta[i],dd[i]) for i in E)/D if D else math.sqrt(sum(dot(dd[i],dd[i]) for i in E))
    Sig=[[sum(w*x[k]*x[l] for _,x in data) for l in range(2)] for k in range(2)]
    lam=(Sig[0][0]+Sig[1][1]+math.sqrt((Sig[0][0]-Sig[1][1])**2+4*Sig[0][1]**2))/2
    Cx=Sig[0][0]+Sig[1][1]
    chi=sum(w*abs(x[1-k])*(norm(x)+x[k]) for k,x in data)
    U=sum(norm(v[i])*norm(z[i]) for i in B+W)+R
    mu=math.sqrt(lam*sum(w*sum((x[k]-f[m][k])**2 for k in range(2)) for m,(_,x) in enumerate(data)))
    K=mu # Pointwise residual bound suffices for this algebraic check.
    ell=[[sum(w*x[k]**2*alpha[i][m] for m,(kk,x) in enumerate(data) if kk==k) for k in range(2)] for i in range(n)]
    c1=sum(z[i][1]*plus(v[i][0]) for i in B+W)
    c2=sum(z[i][0]*plus(v[i][1]) for i in B+W)
    p=[sum(z[i][k]*plus(v[i][k]) for i in B+W)+sum(plus(v[i][k])**2 for i in E) for k in range(2)]
    def row(G):
        RG=sum(dot(v[i],v[i]) for i in G)
        b2=sum(ell[i][1]*v[i][0]**2 for i in G)
        b4=sum(ell[i][0]*v[i][1]**2 for i in G)
        theta=max((sum(ell[i][k]*(1-p[k]) for k in range(2)) for i in G),default=0)
        beta=K*math.sqrt(RG)+lam*RG*math.sqrt(R)
        tail=chi*(1+U)*RG
        return RG,b2,b4,theta,beta,tail
    RI,bI2,bI4,tI,beI,tailI=row(I)
    RJ,bJ2,bJ4,tJ,beJ,tailJ=row(J)
    zQ=math.sqrt(sum(dot(v[i],v[i])*(ell[i][1]-ell[i][0])**2 for i in E))
    zN=max(math.sqrt(sum(dot(v[i],v[i])*ell[i][k]**2 for i in E)) for k in range(2))
    zH=math.sqrt(sum(dot(v[i],v[i])*(ell[i][1]*c2-ell[i][0]*c1)**2 for i in E))
    rhsQ=(tI-bI2-bI4)*Q+max(bI2,bI4)*N+beI*D+plus(-bI2*c2-bI4*c1)+tailI
    rhsN=(bJ2+bJ4)*Q+tJ*N+beJ*D+plus(bJ2*c2+bJ4*c1)+tailJ
    rhsD=zQ*Q+zN*N+(mu+math.sqrt(lam*Cx)*R)*D+zH+chi*U*math.sqrt(R)
    assert all(x<=y+1e-9 for x,y in zip((Qp,Np,Dp),(rhsQ,rhsN,rhsD))), (v,z,(Qp,Np,Dp),(rhsQ,rhsN,rhsD))
    # The spectral refinement uses the exact least eigenvalue of sym(A0).
    ms=[]
    for i in E:
        d1=ell[i][0]*(1-p[0]);d2=ell[i][1]*(1-p[1])
        hh=ell[i][1]*(c2+Q-N2)+ell[i][0]*(c1+Q-N4)
        ms.append((d1+d2-math.sqrt((d1-d2)**2+hh**2))/2)
    improved=min(mu,-min(ms)+chi*(1+U)+lam*math.sqrt(R)*D)
    assert Dp<=rhsD+(improved-mu)*D+1e-9
    # Check exact skew split and error bound, gate-by-gate.
    for i in E:
        X=sum(w*(f[m][0]*x[1]-f[m][1]*x[0])*alpha[i][m] for m,(_,x) in enumerate(data))
        X0=ell[i][1]*c2-ell[i][0]*c1+(ell[i][1]-ell[i][0])*Q+ell[i][0]*N4-ell[i][1]*N2
        assert abs(X-X0)<=chi*U+math.sqrt(lam*Cx*R)*D+1e-9
    return 1

vectors=[(0.3,0.4),(0.3,-0.4),(-0.3,0.4),(-0.3,-0.4),(0.01,0.6),(0.6,0.01)]
datasets=[[(0,[3.,0.]),(1,[0.,2.])],[(0,[3.,0.1]),(0,[2.9,-0.15]),(1,[0.1,2.1]),(1,[-0.15,1.9])]]
checks=0
for data in datasets:
 for u in vectors:
  for v in vectors:
   vv=[(0.7,0.02),(-0.04,0.3),u,v]
   for rotate in (False,True):
    zz=[(0.6*x-0.8*y,0.8*x+0.6*y) if rotate else (x,y) for x,y in vv]
    checks+=audit(vv,zz,data,[0],[1],[2,3])
M=s.Rational(1,2);mass=s.Rational(1,10);kk=s.sqrt(l1*(l1+l2));gg=2*l1*(1-M)
CC=s.Matrix([[l1*(1-M)+l2*(1-mass)-l1*mass-gg,(kk+l1*mass)*s.sqrt(mass/M)],[(l1-l2)*s.sqrt(mass*M),-l2*(1-mass)+kk*mass-gg/2]])
assert s.simplify(s.trace(CC)-(-s.Rational(99,20)+3*s.sqrt(13)/20))==0
assert s.simplify(CC.det()-(s.Rational(1413,400)-51*s.sqrt(13)/100))==0
assert float(CC.det())>0 and float(s.trace(CC))<0
# Angular integrations underlying the uniform-subset initialization bounds.
ang=s.symbols('ang',real=True)
assert s.integrate(s.cos(ang)*s.sin(ang),(ang,0,s.pi/2))/(2*s.pi)==1/(4*s.pi)
assert s.integrate(s.cos(ang)**2*s.sin(ang)**2,(ang,0,s.pi/2))/(2*s.pi)==s.Rational(1,32)
result={'exact_symbolic_obstructions':'passed','finite_state_algebra_checks':checks,'matrix_comparison':'passed','spectral_refinement':'passed','scope':'Algebra verification only; no gradient-flow experiments or parameter-region certificate.'}
Path('outputs/SWINGBY_COUPLED_SIGNED_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
