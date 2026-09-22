"""Exact-rational corner tests for arithmetic used by the static certificate.
These test implementation against exact fractions, not agreement with floats.
"""
from interval_bounds import *
from fractions import Fraction as F
from itertools import product
import json

def frac(x):return F.from_float(float(x))
def contains(out,exact):
 exact=np.asarray(exact,dtype=object)
 assert out.c.shape==exact.shape
 for l,q,h in zip(out.lo().flat,exact.flat,out.hi().flat):assert frac(l)<=q<=frac(h),(l,q,h)
checks=0
for c1,r1,c2,r2 in [(1.,2**-40,-1.,2**-42),(2**60,1.,2**-60,2**-70),(-.1,2**-55,.3,2**-54),(1e-200,1e-210,1e-100,1e-110),(2**-100,0.,-2**-100,0.)]:
 a=I(c1,r1);b=I(c2,r2)
 for x,y in product([frac(c1)-frac(r1),frac(c1)+frac(r1)],[frac(c2)-frac(r2),frac(c2)+frac(r2)]):
  contains(a+b,x+y);contains(a*b,x*y);checks+=2
A=I(np.array([[.1,-.2],[3.,1e-200]]),np.array([[2**-55,2**-54],[2**-51,1e-210]]))
B=I(np.array([[.7,2.],[-.1,1e100]]),np.array([[2**-54,2**-50],[2**-56,1e80]]))
AB=A@B
for signs in product([-1,1],repeat=8):
 ac=np.array([frac(c)+s*frac(r) for c,r,s in zip(A.c.flat,A.r.flat,signs[:4])],dtype=object).reshape(2,2)
 bc=np.array([frac(c)+s*frac(r) for c,r,s in zip(B.c.flat,B.r.flat,signs[4:])],dtype=object).reshape(2,2)
 contains(AB,ac@bc);contains(A.sum(),ac.sum());checks+=2
# Dot product accumulation with severe cancellation.
x=np.array([1e100,1.,-1e100,2**-50]);y=np.ones(4)
contains(I(x)@I(y),sum((frac(a)*frac(b) for a,b in zip(x,y)),F(0)));checks+=1
result={'status':'PASS','exact_rational_enclosure_checks':checks,'scope':'Arithmetic regression only; does not certify trajectory reachability or replace the real-arithmetic proof.'}
print(json.dumps(result,indent=2))
(Path(__file__).resolve().parent.parent/'INTERVAL_ARITHMETIC_CHECKS.json').write_text(json.dumps(result,indent=2)+'\n')
