#!/usr/bin/env python3
"""Build all available daily CSI300 raw liquidity indices, retaining failures.

Persistent Ray actors process dates concurrently; each day runs serially in a subprocess. The underlying builder validates input
fingerprints and resumes completed constituent work. Re-running this command
rechecks every date; it does not silently trust a previous success marker.
"""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]


class DayActor:
    """One actor runs one serial date at a time, with an isolated daily log."""

    def process(self, command, day, root, log):
        root, log = Path(root), Path(log)
        missing = [kind for kind in ('orders', 'trades', 'cancels')
                   if not (root/kind/f'year={day[:4]}'/f'{day}.parquet').is_file()]
        row = dict(date=day, status='', returncode='', log=str(log))
        if missing:
            log.write_text('Missing archives: '+', '.join(missing)+'\n')
            row['status'] = 'missing_inputs'
            return row
        with log.open('w') as stream:
            stream.write(json.dumps(command)+'\n'); stream.flush()
            result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
        row.update(returncode=result.returncode,
                   status='built' if result.returncode == 0 else 'failed')
        return row


def day_command(args, day):
    command = [sys.executable, str(ROOT/'scripts/liquidity/build_csi300_hf_liquidity.py'),
               '--stage', 'build', '--start', day, '--end', day,
               '--root', str(args.root), '--out', str(args.out),
               '--weights',str(args.weights.resolve()),'--membership',str(args.membership.resolve()),
               '--max-weight-age-days', str(args.max_weight_age_days),
               '--weight-timing', 'month-end', '--diagnostic',
               '--allow-estimated-weights', '--shanghai-residual',
               '--grid-ms', str(args.grid_ms), '--notionals', '1000000', '5000000', '10000000',
               '--min-coverage', '.95', '--min-valid', '250',
               '--backend', 'serial', '--workers', '1']
    command.extend(['--weight-gap-policy',getattr(args,'weight_gap_policy','none')])
    if getattr(args,'reuse_from',None):
        command.extend(['--reuse-from',str(args.reuse_from.resolve())])
    if args.weight_schedule:
        command.extend(['--weight-schedule', str(args.weight_schedule.resolve())])
    if args.snapshots:
        command.extend(['--snapshots', str(args.snapshots.resolve())])
    return command


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/data/data/level2data'))
    p.add_argument('--out', type=Path)
    p.add_argument('--start', default='00000000')
    p.add_argument('--end', default='99999999')
    p.add_argument('--dates', nargs='+', help='Only these archive dates (targeted incremental repair)')
    p.add_argument('--ray-actors', type=int, default=80,
                   help='Number of Ray day actors; stocks run serially within each day (default: 80)')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--grid-ms',type=int,choices=[100,1000,3000],default=1000)
    p.add_argument('--max-weight-age-days', type=int, default=40)
    p.add_argument('--weight-gap-policy',choices=['none','next-matching'],default='next-matching')
    p.add_argument('--reuse-from',type=Path)
    p.add_argument('--weight-schedule',type=Path,help='CSV with month/effective_date/available_date')
    p.add_argument('--weights',type=Path,default=ROOT/'data/csi300/csi300_monthly_weights.csv')
    p.add_argument('--membership',type=Path,default=ROOT/'data/csi300/csi300_membership.csv')
    p.add_argument('--snapshots',type=Path,help='Verified snapshot root YYYYMMDD/CODE.parquet')
    p.add_argument('--fast', action='store_true', help=argparse.SUPPRESS)  # Legacy no-op.
    args = p.parse_args()
    if args.ray_actors < 1 or args.start > args.end or args.max_weight_age_days < 0:
        p.error('Require positive --ray-actors, nonnegative weight age, and start <= end')
    args.root = args.root.resolve()
    if args.out is None:
        args.out=ROOT/f'artifacts/liquidity/csi300_monthly_imputed_{args.grid_ms}ms'
    args.out = args.out.resolve()
    calendar = sorted({f.stem for f in (args.root/'trades').glob('year=*/*.parquet')
                       if len(f.stem) == 8 and f.stem.isdigit()})
    dates = [day for day in calendar if args.start <= day <= args.end]
    if args.dates:
        unknown=set(args.dates)-set(dates)
        if unknown:p.error(f'Requested dates outside archive/range: {sorted(unknown)}')
        dates=[day for day in dates if day in set(args.dates)]
    if not dates:
        p.error('No trade archive dates in requested interval')
    print(f'{len(dates)} dates: {dates[0]}–{dates[-1]}; output: {args.out}', flush=True)
    print('CSI300 cash full spreads: RMB 1/5/10 million per stock per side. Full verified index and thresholded available-stock diagnostics are separate.', flush=True)
    if args.dry_run:
        return
    if (args.out/'run.json').exists():
        existing=json.loads((args.out/'run.json').read_text())
        if existing.get('measure_version')!='zhao-cash-full-spread-v1':
            p.error('Output contains an earlier measure. Use a new Zhao output directory; old results cannot be converted or resumed.')
    logs = args.out/'batch_logs'
    logs.mkdir(parents=True, exist_ok=True)
    results = []
    status_path = args.out/'batch_status.csv'
    history = {}
    if status_path.exists():
        with status_path.open(newline='') as stream:
            history = {row['date']: row for row in csv.DictReader(stream)}
    import ray
    ray.init(num_cpus=args.ray_actors, include_dashboard=False, log_to_driver=False,
             runtime_env={'env_vars': {'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1',
                                       'MKL_NUM_THREADS': '1'}})
    actor_type = ray.remote(num_cpus=1, max_restarts=0)(DayActor)
    actors = [actor_type.remote() for _ in range(min(args.ray_actors, len(dates)))]
    iterator = iter(dates)
    pending = {}

    def submit(actor):
        day = next(iterator, None)
        if day is not None:
            ref = actor.process.remote(day_command(args, day), day, str(args.root),
                                       str(logs/f'{day}.log'))
            pending[ref] = (actor, day)

    try:
        for actor in actors:
            submit(actor)
        with tqdm(total=len(dates), desc='Archive days') as progress:
            while pending:
                ready, _ = ray.wait(list(pending), num_returns=1)
                ref = ready[0]
                actor, day = pending.pop(ref)
                try:
                    row = ray.get(ref)
                except ray.exceptions.RayError as exc:
                    log = logs/f'{day}.log'
                    with log.open('a') as stream:
                        stream.write(f'\nRay actor failure: {exc}\n')
                    row = dict(date=day, status='failed', returncode='', log=str(log))
                    ray.kill(actor)
                    actor = actor_type.remote()
                results.append(row)
                history[day] = row
                temp = status_path.with_suffix('.tmp')
                with temp.open('w', newline='') as stream:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                    writer.writerows(history[key] for key in sorted(history))
                temp.replace(status_path)
                progress.update(1)
                progress.set_postfix(last=day, status=row['status'])
                submit(actor)
    finally:
        ray.shutdown()
    # Aggregate all saved dates. A built day can still have missing index values;
    # diagnostics.csv reports validity and coverage separately from process status.
    summary_rc = 0
    if any((args.out/'index').glob('*.parquet')):
        summary_rc = subprocess.run(
            [sys.executable, str(ROOT/'scripts/liquidity/summarize_csi300_hf_liquidity.py'), str(args.out)],
            cwd=ROOT).returncode
    failed = sum(r['status'] != 'built' for r in results)
    print(f'Finished: {len(results)-failed} built, {failed} failed/missing; {status_path}', flush=True)
    raise SystemExit(1 if failed or summary_rc else 0)


if __name__ == '__main__':
    main()
