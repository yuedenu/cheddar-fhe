#!/usr/bin/env python3
"""Measure the original and migrated source binaries consecutively on one GPU."""
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import time

from migration_validate import ROOT, digest, save, summarize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-build', required=True, type=Path)
    p.add_argument('--build', type=Path, default=ROOT / 'build-release')
    p.add_argument('--output', type=Path, default=ROOT / 'reports/migration/paired-performance')
    args = p.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'results.json').exists():
        p.error('Choose a new output directory to preserve the existing measurement.')
    with (ROOT / 'reports/gpu-validation.lock').open('a') as lock:
        print('Waiting for GPU validation to finish', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        results = []
        for model, value in [('resnet', 16), ('lenet', 10)]:
            for label, build in [('original', args.baseline_build), ('migrated', args.build)]:
                name = model + '-' + label
                exe = build.resolve() / 'apps' / model
                report = output / (name + '.jsonl')
                command = [str(exe), '--mode', 'benchmark', '--count', '1', '--relu-range', str(value),
                           '--warmup', '2', '--repeats', '10', '--report', str(report)]
                print('Running ' + name, flush=True)
                with (output / (name + '-gpu.csv')).open('w') as gpu, (output / (name + '.log')).open('w') as log:
                    monitor = subprocess.Popen(['nvidia-smi', '--query-gpu=timestamp,memory.used,utilization.gpu,temperature.gpu,clocks.sm,power.draw',
                                                '--format=csv', '-l', '5'], stdout=gpu, stderr=subprocess.STDOUT)
                    started = time.time()
                    try:
                        code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
                    finally:
                        monitor.terminate()
                        monitor.wait()
                summary = summarize(report)
                b = summary.get('benchmark', {})
                valid = (code == 0 and summary.get('summary', {}).get('pass', False)
                         and b.get('repeats') == 10 and b.get('polynomial_agreement') == 10 and b.get('finite', False))
                row = dict(model=model, label=label, returncode=code, valid=valid, command=command,
                           seconds=time.time()-started, executable_sha256=digest(exe),
                           library_sha256=digest(build.resolve() / 'libcheddar_source.so'), **summary)
                results.append(row)
                save(output / 'results.json', results)
                print(f'Finished {name}: exit {code}, valid={valid}', flush=True)
        comparison = {}
        for model in ['resnet', 'lenet']:
            old, new = [r for r in results if r['model'] == model]
            entry = dict(valid=old['valid'] and new['valid'])
            if entry['valid']:
                before = old['benchmark']['inference']['median_seconds']
                after = new['benchmark']['inference']['median_seconds']
                entry.update(original_median_seconds=before, migrated_median_seconds=after,
                             change_percent=100*(after/before-1),
                             original_stddev_seconds=old['benchmark']['inference']['stddev_seconds'],
                             migrated_stddev_seconds=new['benchmark']['inference']['stddev_seconds'])
            comparison[model] = entry
        save(output / 'comparison.json', comparison)
        print(json.dumps(comparison, indent=2))
        return 0 if all(r['valid'] for r in results) else 2


if __name__ == '__main__':
    raise SystemExit(main())
