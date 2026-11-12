#!/usr/bin/env python3
"""Run reproducible, separate-process GPU validation with an exclusive lock."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--suite', choices=['smoke', '1000', 'layers', 'operators', 'relu'], default='smoke')
parser.add_argument('--build', type=Path, default=ROOT / 'build')
parser.add_argument('--legacy-build', type=Path, default=ROOT / 'build-reference')
parser.add_argument('--backend', choices=['source', 'legacy', 'both'], default='source')
parser.add_argument('--historical', action='store_true', help='Alias for --backend both')
parser.add_argument('--configs', nargs='+', type=int, choices=[30, 35, 40, 48])
parser.add_argument('--params-dir', type=Path, default=ROOT / 'unittest/parameters')
parser.add_argument('--relu-range', type=float, default=10)
parser.add_argument('--output', type=Path, default=ROOT / 'reports/validation')
args = parser.parse_args()
args.build = args.build.resolve()
args.legacy_build = args.legacy_build.resolve()
args.output = args.output.resolve()
args.params_dir = args.params_dir.resolve()
args.output.mkdir(parents=True, exist_ok=True)

# Full models and evaluation keys must not compete for GPU memory.
gpu_lock = (ROOT / 'reports/gpu-validation.lock').open('a')
print('Waiting for the project GPU validation lock', flush=True)
fcntl.flock(gpu_lock, fcntl.LOCK_EX)

inputs = [ROOT / 'data/cifar10/test_batch.bin',
          *sorted((ROOT / 'data/weights').glob('*')),
          *sorted(args.params_dir.glob('resnetparam_*.json'))]
manifest = {str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in inputs if path.is_file()}
(args.output / 'inputs-sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
configs = args.configs or ([40] if args.suite in ['1000', 'layers', 'relu'] else [30, 35, 40, 48])
count = 1000 if args.suite == '1000' else 10 if args.suite == 'layers' else 1
mode = 'ops' if args.suite == 'operators' else 'relu' if args.suite == 'relu' else 'full'
backend = 'both' if args.historical else args.backend
backends = ['source', 'legacy'] if backend == 'both' else [backend]
results = []
for config in configs:
    reports = {}
    for backend in backends:
        build = args.legacy_build if backend == 'legacy' else args.build
        exe = build / 'apps' / ('resnet_legacy' if backend == 'legacy' else 'resnet')
        name = f'{args.suite}-{backend}-{config}'
        report = args.output / (name + '.jsonl')
        log = args.output / (name + '.log')
        command = [str(exe), '--params', str(args.params_dir / f'resnetparam_{config}.json'),
                   '--report', str(report), '--mode', mode, '--count', str(count),
                   '--layers', '10', '--relu-range', str(args.relu_range)]
        print('Running ' + name, flush=True)
        executable_sha = hashlib.sha256(exe.read_bytes()).hexdigest()
        with log.open('w') as stream:
            completed = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        results.append(dict(name=name, returncode=completed.returncode, report=str(report),
                            log=str(log), executable_sha256=executable_sha, command=command))
        if report.exists():
            with report.open() as stream:
                environment = json.loads(next(stream, '{}'))
            libraries = {Path(line.split()[-1]) for line in environment.get('libraries', [])}
            results[-1]['libraries_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                             for path in libraries}
        reports[backend] = report
        (args.output / 'status.json').write_text(json.dumps(results, indent=2) + '\n')
        print(f'Finished {name}: exit {completed.returncode}', flush=True)
    if len(backends) == 2:
        destination = args.output / f'{args.suite}-comparison-{config}.json'
        code = subprocess.run([sys.executable, str(ROOT / 'scripts/compare_reports.py'),
                               str(reports['source']), str(reports['legacy']),
                               '--output', str(destination)]).returncode
        results.append(dict(name=f'comparison-{config}', returncode=code, report=str(destination)))
(args.output / 'status.json').write_text(json.dumps(results, indent=2) + '\n')
sys.exit(0 if all(result['returncode'] == 0 for result in results) else 2)
