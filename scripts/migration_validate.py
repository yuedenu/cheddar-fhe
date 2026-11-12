#!/usr/bin/env python3
"""Revalidate the migrated source build against the preserved source reports."""
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def summarize(path):
    result = dict(images=0, checks=0, failed_checks=0, activation_ranges=0,
                  invalid_activation_ranges=0, max_logit_error=0., max_layer_error=0.)
    images = set()
    if not path.exists():
        return result
    for line in path.open():
        row = json.loads(line)
        kind = row['kind']
        if kind in ['environment', 'accuracy', 'summary', 'benchmark', 'setup', 'benchmark_setup']:
            result[kind] = row
        if kind == 'prediction':
            images.add(row['image'])
        if kind == 'comparison':
            result['checks'] += 1
            result['failed_checks'] += not row['pass']
            key = 'max_logit_error' if row['name'] == 'logits' else 'max_layer_error'
            result[key] = max(result[key], row.get('max_abs', 0.) or 0.)
        if kind == 'activation_range':
            result['activation_ranges'] += 1
            result['invalid_activation_ranges'] += not row['pass']
    result['images'] = len(images)
    result['all_1000_images'] = images == set(range(1000))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build', type=Path, default=ROOT / 'build-release')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/migration')
    parser.add_argument('--resume', action='store_true', help='Reuse successful commands only when inputs, library and executable hashes match.')
    args = parser.parse_args()
    build, output = args.build.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    status_path = output / 'gpu-status.json'
    if status_path.exists() and not args.resume:
        parser.error('Output already has gpu-status.json; choose a fresh --output directory.')
    previous = json.loads(status_path.read_text()) if status_path.exists() else []
    previous_by_name = {item['name']: item for item in previous}
    lock_path = ROOT / 'reports/gpu-validation.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open('a') as lock:
        print('Waiting for the project GPU validation lock', flush=True)
        fcntl.flock(lock, fcntl.LOCK_EX)
        inputs = [*sorted((ROOT / 'data').rglob('*')), *sorted((ROOT / 'unittest/parameters').glob('*.json'))]
        fingerprints = {str(p.relative_to(ROOT)): digest(p) for p in inputs if p.is_file()}
        if previous:
            if fingerprints != json.loads((output / 'inputs-sha256.json').read_text()):
                parser.error('Inputs changed; use a fresh output directory for a complete run.')
            old_result = json.loads((output / 'RESULT.json').read_text())
            if old_result['library_sha256'] != digest(build / 'libcheddar_source.so'):
                parser.error('Library changed; use a fresh output directory for a complete run.')
            archive = output / ('previous-attempt-' + time.strftime('%Y%m%d-%H%M%S'))
            archive.mkdir()
            for filename in ['gpu-status.json', 'RESULT.json']:
                shutil.copy2(output / filename, archive / filename)
            for item in previous:
                if item.get('returncode') != 0:
                    for suffix in ['.log', '.jsonl', '.json']:
                        path = output / (item['name'] + suffix)
                        if path.exists():
                            shutil.copy2(path, archive / path.name)
        save(output / 'inputs-sha256.json', fingerprints)
        executions = []

        def run(name, command):
            item = dict(name=name, command=[str(x) for x in command], state='running', started=time.time())
            if Path(command[0]).is_file():
                item['executable_sha256'] = digest(Path(command[0]))
            old = previous_by_name.get(name, {})
            # Recompute comparisons after any retry; successful GPU work can
            # be reused without repeating a multi-hour acceptance run.
            report = output / (name + '.jsonl')
            evidence_exists = (output / (name + '.xml')).exists() if name in ['basic-40', 'boot-40'] else report.exists()
            if (not name.endswith('-comparison') and old.get('returncode') == 0
                    and old.get('command') == item['command'] and evidence_exists
                    and old.get('executable_sha256') == item.get('executable_sha256')):
                executions.append(old)
                save(status_path, executions)
                print('Reusing verified successful run: ' + name, flush=True)
                return 0
            executions.append(item)
            save(status_path, executions)
            print('Running ' + name, flush=True)
            with (output / (name + '.log')).open('w') as log:
                proc = subprocess.run(item['command'], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            item.update(state='completed', returncode=proc.returncode, seconds=time.time()-item['started'])
            save(status_path, executions)
            print(f'Finished {name}: exit {proc.returncode}, {item["seconds"]:.1f}s', flush=True)
            return proc.returncode

        def app(name, executable, *options):
            return run(name, [build / 'apps' / executable, *options, '--report', output / (name + '.jsonl')])

        run('basic-40', [build / 'basic_test', '--gtest_filter=*bootparam_40_json',
                         '--gtest_output=xml:' + str(output / 'basic-40.xml')])
        run('boot-40', [build / 'boot_test', '--gtest_filter=*Testbed32*bootparam_40_json',
                        '--gtest_output=xml:' + str(output / 'boot-40.xml')])
        app('primitives-40', 'primitive_test')
        app('operators-40', 'resnet', '--mode', 'ops')
        app('relu-40', 'resnet', '--mode', 'relu')
        for config in [30, 35, 40, 48]:
            app(f'smoke-{config}', 'resnet', '--params', ROOT / f'unittest/parameters/resnetparam_{config}.json',
                '--count', '1', '--relu-range', '10')
        app('resnet-1000', 'resnet', '--mode', 'full', '--count', '1000', '--layers', '10', '--relu-range', '16')
        app('resnet-benchmark', 'resnet', '--mode', 'benchmark', '--count', '1', '--relu-range', '16',
            '--warmup', '2', '--repeats', '10')
        app('lenet-1000', 'lenet', '--mode', 'all', '--count', '1000', '--layers', '10', '--relu-range', '10',
            '--warmup', '2', '--repeats', '10')

        comparisons = {}
        for model, baseline in [('resnet', ROOT / 'reports/release-1000/1000-source-40.jsonl'),
                                ('lenet', ROOT / 'reports/lenet/acceptance/source.jsonl')]:
            destination = output / (model + '-comparison.json')
            run(model + '-comparison', [sys.executable, ROOT / 'scripts/compare_reports.py',
                                        output / (model + '-1000.jsonl'), baseline, '--output', destination])
            comparisons[model] = (json.loads(destination.read_text()) if destination.exists()
                                  else dict(passed=False, classification={}, problems=['Comparison did not produce a report.']))

        models = {name: summarize(output / (name + '-1000.jsonl')) for name in ['resnet', 'lenet']}
        timing = {'resnet': summarize(output / 'resnet-benchmark.jsonl').get('benchmark', {}),
                  'lenet': models['lenet'].get('benchmark', {})}
        old_resnet = json.loads((ROOT / 'reports/RESNET20_RESULT.json').read_text())
        old_lenet = json.loads((ROOT / 'reports/lenet/RESULT.json').read_text())
        baseline_times = {
            'resnet': next(p['inference']['source_median_seconds'] for p in old_resnet['performance'] if p['relu_range'] == 16),
            'lenet': old_lenet['backends']['source']['benchmark']['inference']['median_seconds']}
        performance = {}
        for name in models:
            b = timing[name]
            median = b.get('inference', {}).get('median_seconds')
            valid = b.get('finite', False) and b.get('repeats') == 10 and b.get('polynomial_agreement') == 10
            performance[name] = dict(valid=valid, current_median_seconds=median,
                                     previous_median_seconds=baseline_times[name],
                                     change_percent=100*(median/baseline_times[name]-1) if median else None,
                                     note='Previous measurement from the preserved source acceptance; separate session.')
        environments = [summarize(p).get('environment', {}) for p in output.glob('*.jsonl')]
        expected_library = (build / 'libcheddar_source.so').resolve()
        independence = bool(environments)
        for env in environments:
            libraries = {Path(s.split()[-1]).resolve() for s in env.get('libraries', [])}
            independence &= libraries == {expected_library}
            independence &= all(Path(env.get(key, '/')).resolve().is_relative_to(ROOT) for key in ['params', 'weights', 'data'])
        passed = (all(e['returncode'] == 0 for e in executions)
                  and all(m['all_1000_images'] and m.get('summary', {}).get('pass') for m in models.values())
                  and all(c['passed'] and c['classification'].get('matching_predictions') == 1000 for c in comparisons.values())
                  and all(p['valid'] for p in performance.values()) and independence)
        result = dict(passed=passed,
                      pass_scope='Build runtime independence, functional/numerical equivalence and valid timing samples; no latency-equivalence threshold.',
                      executions=executions, models=models, performance=performance,
                      comparisons={k: {f:v for f,v in c.items() if f != 'checkpoints'} for k,c in comparisons.items()},
                      project_local_runtime=bool(independence), library_sha256=digest(expected_library),
                      limits=['LeNet model quality remains unvalidated: the original weights predict class 4 on all 1000 images.',
                              'Comparison is against the restored source backend in cheddar, not the original ParFuse binaries.',
                              'Performance is reported separately; no latency threshold is inferred.'])
        save(output / 'RESULT.json', result)
        print(json.dumps(dict(passed=passed, performance=performance, project_local_runtime=bool(independence)), indent=2), flush=True)
        return 0 if passed else 2


if __name__ == '__main__':
    sys.exit(main())
