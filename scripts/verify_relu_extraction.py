#!/usr/bin/env python3
"""Optional research audit, not a build/runtime dependency."""
import hashlib, json, re, struct
from pathlib import Path
root=Path(__file__).resolve().parents[1]
record=json.loads((root/'reports/relu_coefficients.json').read_text())
data=(root/'legacy/snapshot/lib/parfuse/libcheddar_parfuse.so').read_bytes()
assert hashlib.sha256(data).hexdigest()==record['sha256'], 'unrecognized historical binary'
header=(root/'include/extension/ReLUCoefficients.h').read_text()
for stage,(addr,expected) in enumerate(zip(record['offsets'],record['coefficients']),1):
    actual=list(struct.unpack_from('<'+'d'*len(expected),data,int(addr,16)))
    assert actual==expected, f'coefficient mismatch at {addr}'
    body=re.search(r'relu_p'+str(stage)+r' = \{([^}]+)\}',header).group(1)
    emitted=[float.fromhex(x.strip()) for x in body.split(',')]
    assert emitted==actual, f'source coefficient mismatch at stage {stage}'
print('PASS: all 60 polynomial coefficients match the fingerprinted binary')
