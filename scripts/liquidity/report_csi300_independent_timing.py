#!/usr/bin/env python3
"""Update the existing timing note from measured, independently fitted DC cases."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil

for name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'

import numpy as np
from scipy.fft import next_fast_len
from scipy.stats import norm, rankdata

LABELS = {'full': 'Full-data ARTFIMA, MCMC',
          'dc_serial': 'Spectral DC ARTFIMA, sequential',
          'dc_parallel': 'Spectral DC ARTFIMA, parallel'}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--quality', type=Path, required=True)
    parser.add_argument('--tex', type=Path, required=True)
    args = parser.parse_args()
    root, quality, tex = (x.expanduser().resolve() for x in (args.input, args.quality, args.tex))
    config = json.loads((root/'config.json').read_text())
    if config.get('version') != 2 or len(config['dates']) != 1 or config['repetitions'] < 2:
        raise ValueError('This note requires one history with independent DC initialization')
    if any(digest(Path(config['source'])/name) != value for name, value in config['source_sha256'].items()):
        raise ValueError('Frozen numerical source changed')
    if digest(root/'benchmark_csi300_computation.py') != config['controller_sha256']:
        raise ValueError('Benchmark controller changed')
    quality_config = json.loads((quality/'config.json').read_text())
    if Path(quality_config['benchmark']) != root:
        raise ValueError('Stein results do not describe this benchmark')
    if any(digest(record['path']) != record['sha256'] for record in quality_config['files'].values()):
        raise ValueError('Stein inputs or diagnostic sources changed')
    discrepancy = json.loads((quality/'summary.json').read_text())['results']
    date, repetitions = config['dates'][0], config['repetitions']
    cases, summary, saved, interference = [], {}, {}, []
    # Use the same frozen rank-normalized bulk ESS implementation as the
    # existing sampler-cost report, without importing forecasting runtimes.
    diagnostic_source = Path(config['source'])/'scripts/liquidity/fit_csi300_observed_dc.py'
    source_text = diagnostic_source.read_text()
    node = next(n for n in ast.parse(source_text).body
                if isinstance(n, ast.FunctionDef) and n.name == 'diagnostics')
    namespace = dict(np=np, norm=norm, rankdata=rankdata, next_fast_len=next_fast_len)
    exec(ast.get_source_segment(source_text, node), namespace)
    for method, label in LABELS.items():
        records = []
        for rep in range(1, repetitions+1):
            folder = root/'cases'/date/f'{method}_r{rep}'
            value = json.loads((folder/'completed.json').read_text())
            monitor = json.loads((folder/'resource_monitor.json').read_text())
            if not monitor['finished'] or monitor['exit_code'] != 0:
                raise ValueError(f'Incomplete timing: {folder}')
            if any(event['uid'] != os.getuid() for event in monitor['interference']):
                raise ValueError(f'Another user was computing: {folder}')
            for check in monitor['checks']:
                if check['competing_compute_processes']:
                    interference.append(dict(method=method, repetition=rep, **check))
            if method != 'full' and (value['global_map_seconds'] != 0 or 'common_full_map' in value):
                raise ValueError('DC performed a global full-data MAP fit')
            if not value['quality']['passed']:
                raise ValueError(f'MCMC quality checks failed: {folder}')
            draws = np.load(folder/'posterior_draws.npy')
            _, bulk = namespace['diagnostics'](draws[None, ...])
            saved[method, rep] = draws
            row = dict(method=method, repetition=rep, n=value['n'], groups=value['groups'],
                       cores=value['slots'], fit=value['fit_seconds'], total=value['all_in_seconds'],
                       startup=value['startup_seconds'], fft=value['fft_seconds'],
                       full_map=value['global_map_seconds'], workers=value['action_seconds'],
                       aggregation=value['aggregation_seconds'], median_bulk_ess=float(np.median(bulk)))
            row['other'] = row['fit']-sum(row[x] for x in ('fft','full_map','workers','aggregation'))
            cases.append(row); records.append(row)
        # The quality diagnostic uses the second matched repetition, with all
        # 1,500 retained draws. Timings are medians over both repetitions.
        selected = records[1]
        times = {key: float(np.median([r[key] for r in records]))
                 for key in ('fit','total','startup','fft','full_map','workers','aggregation','other')}
        summary[method] = dict(label=label, groups=selected['groups'], cores=selected['cores'],
            repetitions=repetitions, median_bulk_ess=selected['median_bulk_ess'],
            ess_per_minute=60*selected['median_bulk_ess']/times['total'],
            stein_discrepancy=discrepancy[method]['discrepancy'], **times)
    if any(not np.array_equal(saved['dc_serial', rep], saved['dc_parallel', rep])
           for rep in range(1, repetitions+1)):
        raise ValueError('Sequential and parallel draws are not identical')
    with (root/'timing_cases.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(cases[0]))
        writer.writeheader(); writer.writerows(cases)
    with (root/'timing_summary.csv').open('w', newline='') as handle:
        rows = [dict(method=method, **row) for method, row in summary.items()]
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    full, serial, parallel = (summary[name] for name in LABELS)
    comparisons = dict(full_to_parallel_fit=full['fit']/parallel['fit'],
        full_to_parallel_total=full['total']/parallel['total'],
        serial_to_parallel_fit=serial['fit']/parallel['fit'],
        serial_to_parallel_total=serial['total']/parallel['total'],
        serial_parallel_draws_bitwise_equal=True)

    original = text = tex.read_text()

    def replace_row(label, replacement):
        nonlocal text
        text, count = re.subn('^'+re.escape(label)+r' & .*?\\\\$', lambda _: replacement,
                              text, flags=re.MULTILINE)
        if count != 1:
            raise ValueError(f'Expected exactly one row: {label}')

    for method, row in summary.items():
        replace_row(row['label'], f"{row['label']} & {row['groups']} & {row['cores']} & "
            f"{row['fit']:.1f} & {row['total']:.1f} & {row['stein_discrepancy']:.3f} & "
            f"{row['median_bulk_ess']:.0f} & {row['ess_per_minute']:.1f}"+r'\\')
    for label, key, digits in [('Spark startup and worker warm-up','startup',1),
            ('FFT, grouping and cache validation','fft',1),
            ('Full-data MAP initialization (serial)','full_map',1),
            ('Posterior worker stage','workers',1), ('Consensus and diagnostics','aggregation',2),
            ('Other driver work','other',2)]:
        replace_row(label, label+' & '+' & '.join(f'{summary[m][key]:.{digits}f}' for m in LABELS)+r'\\')
    text = re.sub(r'The (?:profile|benchmark) uses (?:one|two|\d+) matched repetitions?',
                  f'The benchmark uses {repetitions} matched repetitions', text)
    text = text.replace('All 1,500 draws, including rejection repeats, enter',
        'ESS and Stein discrepancy use all 1,500 retained draws from the second matched repetition. '
        'These draws, including rejection repeats, enter')
    text = re.sub(r'^(?:Lower DC discrepancy here|The discrepancy includes finite Monte Carlo error).*$',
        lambda _: 'The discrepancy includes finite Monte Carlo error; it does not establish '
            'negligible consensus bias or assess Whittle accuracy. Parallel DC reduces fitting time '
            f"from {full['fit']:.1f} to {parallel['fit']:.1f} seconds "
            f"({comparisons['full_to_parallel_fit']:.1f}-fold speedup), or "
            f"{comparisons['full_to_parallel_total']:.1f}-fold including Spark startup.",
        text, flags=re.MULTILINE)
    text = re.sub(r'^(?:In the original fitting benchmark,|DC shards fit).*$',
        lambda _: 'DC shards fit their MAPs and Hessians independently; no serial full-data MAP is '
            'computed for DC. The full-data chain fits its own MAP. The timed fits include FFT, '
            'local optimisation, MCMC, covariance regularisation and task I/O. Sequential and '
            'parallel DC produce bitwise-identical draws in both matched repetitions.',
        text, flags=re.MULTILINE)
    monitoring = ('No competing compute jobs were detected.' if not interference else
        f'The monitor recorded brief own-user Python activity in {len(interference)} '
        f'monitoring {"interval" if len(interference) == 1 else "intervals"}; '
        'no competing computation from other user accounts was detected.')
    text = re.sub(r'^(?:Explicit loop compilation takes|Both methods use).*$',
        lambda _: 'Both methods use the same compiled fixed-mixture MH kernel and proposal scaling '
            'rules. Shape-specific compilation is included; common tiny-array initialization is '
            'excluded. Input construction, clock calibration, forecast simulation and Stein '
            'evaluation are excluded for both methods. These are short, monitored shared-host '
            'measurements. '+monitoring, text, flags=re.MULTILINE)
    if text == original:
        print('Existing report already up to date')
    else:
        tex.write_text(text)
    manifest = dict(config_sha256=digest(root/'config.json'), quality_config_sha256=digest(quality/'config.json'),
        reporter_sha256=digest(__file__), tex=str(tex), tex_sha256=digest(tex),
        quality_repetition=2, cases=len(cases), source_sha256=config['source_sha256'],
        ess_diagnostic_source_sha256=digest(diagnostic_source),
        stages=summary, comparisons=comparisons, brief_own_user_activity=interference)
    (root/'report_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    shutil.copy2(__file__, root/Path(__file__).name)
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
