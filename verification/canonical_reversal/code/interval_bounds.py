"""Outward enclosures for two STATIC witness balls, not trajectory reachability.
Uses IEEE-754 round-to-nearest dot-product gamma bounds, outward nextafter
for scalar bounds, and mpmath interval sine/cosine for the exact -85deg probe.
"""
from pathlib import Path
import numpy as np,json,mpmath as mp,hashlib
UROUND=2.**-53;TINY=np.nextafter(0.,np.inf)
def up(x):return np.nextafter(x,np.inf)
def addp(a,b):return up(np.asarray(a)+np.asarray(b))
def mulp(a,b):return up(np.asarray(a)*np.asarray(b))
def gamma(k):return up((k*UROUND)/(1-k*UROUND))
def dotp(a,b):
 k=a.shape[-1];z=a@b
 return addp(up(z/np.nextafter(1-gamma(k),-np.inf)),k*TINY)
def sump(a,axis=None):
 a=np.asarray(a);k=a.size if axis is None else a.shape[axis]
 return addp(up(np.sum(a,axis=axis)/np.nextafter(1-gamma(k),-np.inf)),k*TINY)
class I:
 def __init__(self,c,r=0):self.c=np.asarray(c,dtype=float);self.r=np.broadcast_to(np.asarray(r,dtype=float),self.c.shape)
 def __getitem__(self,k):return I(self.c[k],self.r[k])
 @property
 def T(self):return I(self.c.T,self.r.T)
 def __neg__(self):return I(-self.c,self.r)
 def __add__(self,b):
  b=asI(b);c=self.c+b.c;err=mulp(gamma(1),addp(abs(self.c),abs(b.c)));return I(c,addp(addp(self.r,b.r),err))
 __radd__=__add__
 def __sub__(self,b):return self+-asI(b)
 def __rsub__(self,b):return asI(b)+-self
 def __mul__(self,b):
  b=asI(b);c=self.c*b.c;rad=addp(addp(mulp(abs(self.c),b.r),mulp(abs(b.c),self.r)),mulp(self.r,b.r));err=addp(mulp(gamma(1),abs(c)),TINY);return I(c,addp(rad,err))
 __rmul__=__mul__
 def __matmul__(self,b):
  b=asI(b);k=self.c.shape[-1];c=self.c@b.c;err=addp(mulp(gamma(k),dotp(abs(self.c),abs(b.c))),k*TINY)
  rad=addp(addp(dotp(abs(self.c),b.r),dotp(self.r,abs(b.c))),dotp(self.r,b.r));return I(c,addp(rad,err))
 def sum(self,axis=None):
  k=self.c.size if axis is None else self.c.shape[axis];c=self.c.sum(axis=axis);err=addp(mulp(gamma(k),sump(abs(self.c),axis)),k*TINY);return I(c,addp(sump(self.r,axis),err))
 def lo(self):return np.nextafter(self.c-self.r,-np.inf)
 def hi(self):return np.nextafter(self.c+self.r,np.inf)
 def absup(self):return addp(abs(self.c),self.r)
 def abslow(self):return np.maximum(0,np.nextafter(abs(self.c)-self.r,-np.inf))
def asI(x):return x if isinstance(x,I) else I(x)
def normup(x,axis=None):
 a=asI(x).absup();return up(np.sqrt(sump(mulp(a,a),axis)))
def stack(vals,axis=0):return I(np.stack([asI(v).c for v in vals],axis),np.stack([asI(v).r for v in vals],axis))
def scal(x):return float(np.asarray(x))
def positive_sum(*args):
 z=0.
 for a in args:z=addp(z,a)
 return z
