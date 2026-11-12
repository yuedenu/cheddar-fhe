#!/usr/bin/env python3
"""Compare decrypted tensors from separate source and historical processes."""
import argparse, json, math
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('source');p.add_argument('historical');p.add_argument('--output',required=True)
a=p.parse_args()
def read(path):
    rows={}; predictions={}; metadata={}
    with Path(path).open() as stream:
        for line in stream:
            j=json.loads(line)
            if j.get('kind')=='comparison' and 'actual' in j:
                rows[(j['image'],j['name'])]=j
            elif j.get('kind')=='prediction': predictions[j['image']]=j
            elif j.get('kind') in ['environment','summary','accuracy','bootstrap_config']:
                metadata[j['kind']]=j
    return rows,predictions,metadata
src,sp,sm=read(a.source);old,hp,hm=read(a.historical)
problems=[]
for label,meta,predictions in [('source',sm,sp),('historical',hm,hp)]:
    if 'summary' not in meta: problems.append(label+': missing final summary')
    env=meta.get('environment',{})
    if env.get('mode')=='full':
        expected=set(range(env['start'],env['start']+env['count']))
        if set(predictions)!=expected: problems.append(label+': incomplete predictions')
for key in ['mode','start','count','relu_range']:
    if sm.get('environment',{}).get(key)!=hm.get('environment',{}).get(key):
        problems.append('different '+key)
if 'bootstrap_config' in sm and 'bootstrap_config' in hm and sm['bootstrap_config']!=hm['bootstrap_config']:
    problems.append('different bootstrap configuration')
results=[]
for key in sorted(set(src)|set(old)):
    if key not in src or key not in old:
        results.append(dict(image=key[0],name=key[1],passed=False,reason='missing checkpoint'));continue
    x,y=src[key],old[key];atol=.01 if key[1]=='logits' else .001;rtol=.001
    if len(x['actual'])!=len(y['actual']):
        results.append(dict(image=key[0],name=key[1],passed=False,reason='shape mismatch'));continue
    finite=all(isinstance(v,(float,int)) and math.isfinite(v) for v in x['actual']+y['actual'])
    errors=[abs(v-w) for v,w in zip(x['actual'],y['actual'])] if finite else [math.inf]
    passed=finite and all(e<=atol+rtol*abs(w) for e,w in zip(errors,y['actual']))
    results.append(dict(image=key[0],name=key[1],passed=passed,atol=atol,rtol=rtol,
        max_abs=max(errors) if finite else None,source_level=x.get('level'),historical_level=y.get('level'),
        source_scale=x.get('scale'),historical_scale=y.get('scale')))
classification={}
if sp or hp:
    common=sorted(set(sp)&set(hp))
    matching=sum(sp[i]['actual_top1']==hp[i]['actual_top1'] for i in common)
    sc=sum(sp[i]['actual_top1']==sp[i]['label'] for i in common)
    hc=sum(hp[i]['actual_top1']==hp[i]['label'] for i in common)
    if any(sp[i]['label']!=hp[i]['label'] for i in common): problems.append('different labels')
    classification=dict(compared_images=len(common),matching_predictions=matching,
        top1_agreement=matching/len(common) if common else None,
        source_accuracy=sc/len(common) if common else None,
        historical_accuracy=hc/len(common) if common else None)
passed=not problems and bool(results) and all(r['passed'] for r in results)
Path(a.output).write_text(json.dumps(dict(passed=passed,problems=problems,
    source_summary=sm.get('summary'),historical_summary=hm.get('summary'),
    source_bootstrap_config=sm.get('bootstrap_config'),historical_bootstrap_config=hm.get('bootstrap_config'),
    classification=classification,checkpoints=results),indent=2)+'\n')
print(f"{'PASS' if passed else 'FAIL'}: {sum(r['passed'] for r in results)}/{len(results)} checkpoints")
raise SystemExit(0 if passed else 2)
