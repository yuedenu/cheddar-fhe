#!/usr/bin/env python3
"""Compare single-image source and historical inference in separate processes."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--build', type=Path, default=ROOT / 'build')
parser.add_argument('--legacy-build', type=Path, default=ROOT / 'build-reference')
parser.add_argument('--params', type=Path, default=ROOT / 'unittest/parameters/resnetparam_40.json')
parser.add_argument('--ranges', nargs='+', type=float, default=[10, 16])
parser.add_argument('--warmup', type=int, default=2)
parser.add_argument('--repeats', type=int, default=10)
parser.add_argument('--image', type=int, default=0)
parser.add_argument('--output', type=Path, default=ROOT / 'reports/benchmark')
args = parser.parse_args()
args.output = args.output.resolve()
args.output.mkdir(parents=True, exist_ok=True)
lock = (ROOT / 'reports/gpu-validation.lock').open('a')
print('Waiting for the project GPU validation lock', flush=True)
fcntl.flock(lock, fcntl.LOCK_EX)

inputs = [args.params.resolve(), ROOT / 'data/cifar10/test_batch.bin',
          *sorted((ROOT / 'data/weights').glob('*'))]
manifest = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs if path.is_file()}
(args.output / 'inputs-sha256.json').write_text(json.dumps(manifest, indent=2) + '\n')
results = []
comparisons = []


def gpu_snapshot():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,uuid,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,memory.used',
                                 '--format=csv'], capture_output=True, text=True)
        return dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except OSError as error:
        return dict(error=str(error))


for index, value in enumerate(args.ranges):
    summaries = {}
    # Reverse the process order for the second range to expose ordering effects.
    order = ['source', 'legacy'] if index % 2 == 0 else ['legacy', 'source']
    for backend in order:
        build = args.build if backend == 'source' else args.legacy_build
        exe = (build / 'apps' / ('resnet' if backend == 'source' else 'resnet_legacy')).resolve()
        name = f'benchmark-{backend}-B{value:g}'
        destination = args.output / (name + '.jsonl')
        command = [str(exe), '--mode', 'benchmark', '--count', '1', '--params', str(args.params.resolve()),
                   '--relu-range', str(value), '--start', str(args.image), '--warmup', str(args.warmup),
                   '--repeats', str(args.repeats), '--report', str(destination)]
        print('Running ' + name, flush=True)
        device_before = gpu_snapshot()
        with (args.output / (name + '.log')).open('w') as log:
            code = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT).returncode
        rows = [json.loads(line) for line in destination.read_text().splitlines()] if destination.exists() else []
        summary = {row['kind']: row for row in rows if row['kind'] in ['benchmark', 'benchmark_setup', 'summary', 'environment']}
        # Exit 2 records numerical comparison failures, expected from the old
        # package. Timing is usable only if every repetition finished and the
        # decrypted predictions remained finite and correct for this image.
        timing = summary.get('benchmark', {})
        valid = (code in (0, 2) and timing.get('repeats') == args.repeats
                 and timing.get('finite', False) and timing.get('correct') == args.repeats
                 and timing.get('polynomial_agreement') == args.repeats)
        row = dict(name=name, backend=backend, relu_range=value, command=command,
                   returncode=code, timing_valid=valid, report=str(destination),
                   device_before=device_before, device_after=gpu_snapshot(),
                   executable_sha256=hashlib.sha256(exe.read_bytes()).hexdigest(), **summary)
        libraries = {Path(line.split()[-1]) for line in summary.get('environment', {}).get('libraries', [])}
        row['libraries_sha256'] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in libraries}
        results.append(row)
        summaries[backend] = row
        (args.output / 'results.json').write_text(json.dumps(results, indent=2) + '\n')
        print(f'Finished {name}: exit {code}, timing_valid={valid}', flush=True)
    source, legacy = summaries['source'], summaries['legacy']
    comparison = dict(relu_range=value, timing_valid=source['timing_valid'] and legacy['timing_valid'])
    if comparison['timing_valid']:
        for scope in ['inference', 'total']:
            s = source['benchmark'][scope]['median_seconds']
            h = legacy['benchmark'][scope]['median_seconds']
            comparison[scope] = dict(source_median_seconds=s, legacy_median_seconds=h,
                                     source_over_legacy=s / h, change_percent=(s / h - 1) * 100)
    comparisons.append(comparison)
    (args.output / 'comparison.json').write_text(json.dumps(comparisons, indent=2) + '\n')
print(json.dumps(comparisons, indent=2))
sys.exit(0 if all(row['timing_valid'] for row in results) else 2)
