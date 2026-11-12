#!/usr/bin/env python3
"""Derive an equivalent, normalized Chebyshev representation; no binary input.

q1(x)=p1(x)/2, q2(x)=p2(2*x)/(8/5), q3(x)=p3((8/5)*x).
Composition and final x*(1/2+q3(q2(q1(x)))) are unchanged.
Decimal arithmetic avoids cancellation while converting the recovered doubles.
"""
from decimal import Decimal, localcontext
from pathlib import Path
from math import comb
import re
root=Path(__file__).resolve().parents[1]
normal_header=(root/'include/extension/ReLUCoefficients.h').read_text()
raw=[]
for stage,length in [(1,16),(2,16),(3,28)]:
    body=re.search(r'relu_p'+str(stage)+r' = \{([^}]+)\}',normal_header).group(1)
    raw.append([float.fromhex(x.strip()) for x in body.split(',')])
    assert len(raw[-1])==length
with localcontext() as ctx:
    ctx.prec=100
    lines=['#pragma once','#include <vector>','namespace cheddar::cnn {']
    for stage,coefs in enumerate(raw):
        result=[Decimal(0)]*len(coefs)
        for n,c in enumerate(coefs):
            if not c: continue
            # x**n = sum_k 2**(1-n) binom(n,k) T_(n-2k)(x)
            input_factor=[Decimal(1),Decimal(2),Decimal(8)/5][stage]
            output_factor=[Decimal(2),Decimal(8)/5,Decimal(1)][stage]
            a=Decimal.from_float(c)*(input_factor**n)/output_factor
            for k in range(n//2+1):
                d=n-2*k
                result[d]+=a*Decimal(comb(n,k))*(Decimal(2)**(1-n))/(2 if d==0 else 1)
        lines.append('inline const std::vector<double> relu_cheb'+str(stage+1)+' = {'+', '.join(float(c).hex() for c in result)+'};')
    lines.append('}')
(root/'include/extension/ReLUChebyshev.h').write_text('\n'.join(lines)+'\n')
