#!/usr/bin/env python3
"""Run LeNet validation and timing against source and historical backends."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--build',type=Path,default=ROOT/'build-release')
p.add_argument('--legacy-build',type=Path,default=ROOT/'build-source')
p.add_argument('--backend',choices=['source','legacy','both'],default='both')
p.add_argument('--mode',choices=['full','benchmark','all'],default='all')
p.add_argument('--count',type=int,default=1000)
p.add_argument('--relu-range',type=float,default=10)
p.add_argument('--warmup',type=int,default=2)
p.add_argument('--repeats',type=int,default=10)
p.add_argument('--output',type=Path,default=ROOT/'reports/lenet/acceptance')
a=p.parse_args()
if a.mode=='benchmark':a.count=1
a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=True)
lock=(ROOT/'reports/gpu-validation.lock').open('a')
print('Waiting for the project GPU validation lock',flush=True);fcntl.flock(lock,fcntl.LOCK_EX)
inputs=[ROOT/'data/cifar10/test_batch.bin',ROOT/'unittest/parameters/resnetparam_40.json',
        *sorted((ROOT/'data/lenet_weights').glob('*.npy'))]
(a.output/'inputs-sha256.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs},indent=2)+'\n')
results=[]
for backend in (['source','legacy'] if a.backend=='both' else [a.backend]):
    build=a.build if backend=='source' else a.legacy_build
    exe=(build/'apps'/('lenet' if backend=='source' else 'lenet_legacy')).resolve()
    report=a.output/f'{backend}.jsonl'
    command=[str(exe),'--mode',a.mode,'--count',str(a.count),'--layers','10',
             '--relu-range',str(a.relu_range),'--warmup',str(a.warmup),'--repeats',str(a.repeats),'--report',str(report)]
    print(f'Running LeNet {backend}, mode={a.mode}, count={a.count}',flush=True)
    executable_sha=hashlib.sha256(exe.read_bytes()).hexdigest()
    with (a.output/f'{backend}.log').open('w') as log:
        code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT).returncode
    summary={};images=set();checks=failed=ranges=bad_ranges=0
    if report.exists():
        for line in report.open():
            row=json.loads(line);kind=row['kind']
            if kind in ['environment','accuracy','benchmark','setup','summary']:summary[kind]=row
            if kind=='prediction':images.add(row['image'])
            if kind=='comparison':checks+=1;failed+=not row['pass']
            if kind=='activation_range':ranges+=1;bad_ranges+=not row['pass']
    library_paths={Path(line.split()[-1]) for line in summary.get('environment',{}).get('libraries',[])}
    result=dict(backend=backend,command=command,returncode=code,report=str(report),
                images=len(images),checks=checks,failed_checks=failed,activation_ranges=ranges,invalid_activation_ranges=bad_ranges,
                executable_sha256=executable_sha,
                libraries_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in library_paths},**summary)
    result['complete']=code in [0,2] and 'summary' in summary and (a.mode=='benchmark' or images==set(range(a.count)))
    if a.mode in ['benchmark','all']:
        b=summary.get('benchmark',{})
        # Classification may be poor even in exact plaintext with these weights.
        # A benchmark must agree with that model, not fabricate a correct label.
        result['timing_valid']=b.get('finite',False) and b.get('repeats')==a.repeats and b.get('polynomial_agreement')==a.repeats
    results.append(result)
    (a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(f'Finished {backend}: exit {code}, complete={result["complete"]}',flush=True)
if len(results)==2 and all(r['complete'] for r in results):
    subprocess.run([sys.executable,str(ROOT/'scripts/compare_reports.py'),str(a.output/'source.jsonl'),
                    str(a.output/'legacy.jsonl'),'--output',str(a.output/'comparison.json')])
    if a.mode in ['benchmark','all'] and all(r.get('timing_valid') for r in results):
        ratio=results[0]['benchmark']['inference']['median_seconds']/results[1]['benchmark']['inference']['median_seconds']
        (a.output/'performance.json').write_text(json.dumps({'source_over_legacy':ratio,'change_percent':100*(ratio-1)},indent=2)+'\n')
sys.exit(0 if all(r['complete']
                  and (a.mode=='full' or r.get('timing_valid',False))
                  and (r['backend']=='legacy' or r['summary']['pass'])
                  for r in results) else 2)
