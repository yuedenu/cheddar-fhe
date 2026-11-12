#!/usr/bin/env python3
"""Check the emitted binary64 Chebyshev coefficients against 80-digit Horner.

The reference uses recovered normal coefficients and no Chebyshev conversion.
Both paths evaluate the complete activation on a fixed grid including endpoints.
"""
from pathlib import Path
from decimal import Decimal, localcontext
import re
root=Path(__file__).resolve().parents[1]
normal_header=(root/'include/extension/ReLUCoefficients.h').read_text()
original=[]
for stage,length in [(1,16),(2,16),(3,28)]:
    body=re.search(r'relu_p'+str(stage)+r' = \{([^}]+)\}',normal_header).group(1)
    original.append([float.fromhex(x.strip()) for x in body.split(',')])
    assert len(original[-1])==length
converted=[]
for body in re.findall(r'= \{([^}]+)\}',(root/'include/extension/ReLUChebyshev.h').read_text()):
    converted.append([float.fromhex(x.strip()) for x in body.split(',')])
assert [len(c) for c in converted]==[16,16,28]
def horner(c,x):
    a=Decimal(0)
    for n in c[::-1]:a=a*x+n
    return a
def cheby(c,x):
    # Direct Chebyshev recurrence, independent of the FHE evaluation tree.
    t0,t1=Decimal(1),x
    out=c[0]+c[1]*x
    for n in c[2:]:
        t0,t1=t1,2*x*t1-t0
        out+=n*t1
    return out
with localcontext() as ctx:
    ctx.prec=80
    original=[[Decimal.from_float(x) for x in p] for p in original]
    converted=[[Decimal.from_float(x) for x in p] for p in converted]
    worst=Decimal(0)
    for i in range(4097):
        x=Decimal(i-2048)/2048
        a=x;b=x
        for p,q in zip(original,converted):a=horner(p,a);b=cheby(q,b)
        error=abs(x*(a-b))
        assert error<=Decimal('1e-9'),(x,error)
        worst=max(worst,error)
print(f'PASS: 4097 activation inputs; max difference {worst:.6E}')
