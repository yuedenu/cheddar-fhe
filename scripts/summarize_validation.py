#!/usr/bin/env python3
"""Summarize complete or running JSONL evidence without loading tensor arrays."""
import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('reports', nargs='+', type=Path)
parser.add_argument('--output', type=Path)
args = parser.parse_args()
results = []
for path in args.reports:
    result = dict(report=str(path.resolve()), complete=False, passed=False,
                  images=0, checks=0, failed_checks=0, activation_ranges=0,
                  invalid_activation_ranges=0, invalid_images=0,
                  max_logit_error=0., max_layer_error=0., failures=[])
    with path.open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                result['incomplete_line'] = True
                break
            kind = row['kind']
            if kind == 'environment':
                result['environment'] = row
            elif kind == 'bootstrap_config':
                result['bootstrap_config'] = row
            elif kind == 'prediction':
                result['images'] += 1
                result['last_image'] = row['image']
            elif kind == 'comparison':
                result['checks'] += 1
                metric = 'max_logit_error' if row['name'] == 'logits' else 'max_layer_error'
                if row['max_abs'] is not None:
                    result[metric] = max(result[metric], row['max_abs'])
                if not row['pass']:
                    result['failed_checks'] += 1
                    if len(result['failures']) < 20:
                        result['failures'].append({k: v for k, v in row.items()
                                                   if k not in ['actual', 'reference']})
            elif kind == 'activation_range':
                result['activation_ranges'] += 1
                result['invalid_activation_ranges'] += not row['pass']
            elif kind == 'invalid_activation':
                result['invalid_images'] += 1
            elif kind == 'accuracy':
                result['accuracy'] = row
            elif kind == 'summary':
                result['complete'] = True
                result['passed'] = row['pass']
                result['summary'] = row
    results.append(result)
encoded = json.dumps(results, indent=2) + '\n'
if args.output:
    args.output.write_text(encoded)
else:
    print(encoded, end='')
