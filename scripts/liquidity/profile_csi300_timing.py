#!/usr/bin/env python3
"""Replay fixed proposals at short/production chain lengths, timing MCMC only."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import signal

from benchmark_csi300_computation import (
    monitor_process, pin_cpu, sha256, stop_case, wait_for_quiet, write_json)


def configure_spark_environment(source):
    """Make both frozen numerics and pickled helper imports visible to workers."""
    source = str(Path(source).resolve())
    helper = Path(__file__).resolve().parent
    paths = [source, str(helper), str(helper.parents[1])]
    os.environ['PYTHONPATH'] = os.pathsep.join(paths)
    os.environ['PYSPARK_PYTHON'] = sys.executable
    sys.path[:] = paths + [entry for entry in sys.path if entry not in paths]
    return os.environ['PYTHONPATH']


def profile_group(group, config, barrier=None):
    started = time.perf_counter()
    cpu = config['cpus'][group] if config['mode'] == 'dc_parallel' else config['cpus'][0]
    pin_cpu(cpu)
    sys.path.insert(0, config['source'])
    import numpy as np
    from scripts.liquidity import _spectral_coordinates as coordinates
    from scripts.liquidity._computation_timing import coordinate_step_timing
    if not Path(coordinates.__file__).resolve().is_relative_to(Path(config['source']).resolve()):
        raise ValueError('Replay loaded coordinates outside the frozen source')

    full = config['mode'] == 'full'
    reference = Path(config['benchmark'])/'cases'/config['date']/(
        ('full' if full else 'dc_parallel')+f'_r{config["reference_repetition"]}')
    metadata = json.loads((reference/'shards'/f'group{group:03d}.json').read_text())
    if full:
        inputs = Path(config['benchmark'])/'inputs'/config['date']
        pg = np.load(inputs/'periodogram.npy')
        omega = np.load(inputs/'omega.npy')
    else:
        with np.load(reference/'frequencies'/f'shard{group:03d}.npz') as data:
            pg, omega = data['periodogram'], data['omega']
    center = np.asarray(metadata['coordinate_map']['theta'])
    covariance = np.asarray(metadata['coordinate_map']['covariance'])*(2.38**2/4)
    sampling_call_started = time.perf_counter()
    with coordinate_step_timing(coordinates, barrier) as times:
        sampled, logp, accepted = coordinates.coordinate_sampler(
            center, covariance, pg, omega, 1 if full else config['groups'],
            config['length'], config['burn'],
            config['seed']+config['reference_repetition']*100_000+group*1000,
            backend='compiled', kernel='mixture', center=center)
    call_seconds = time.perf_counter()-sampling_call_started
    if not np.isfinite(sampled).all() or not np.isfinite(logp).all():
        raise ValueError('Nonfinite replay output')
    comparison = None
    if config['length'] == config['reference_iterations'] and config['burn'] == config['reference_burn']:
        saved = np.load(reference/'shards'/f'group{group:03d}_draws.npy')
        np.testing.assert_allclose(sampled, saved, rtol=1e-12, atol=1e-14)
        comparison = dict(bitwise_equal=bool(np.array_equal(sampled, saved)),
                          max_absolute_error=float(np.max(abs(sampled-saved))))
    thread_cpus=set()
    for task in Path('/proc/self/task').iterdir():
        try:thread_cpus.update(os.sched_getaffinity(int(task.name)))
        except ProcessLookupError:pass
    if thread_cpus!={cpu}:raise ValueError('Replay worker threads escaped their assigned CPU')
    record = dict(times,group=group,cpu=cpu,pid=os.getpid(),thread_cpus=sorted(thread_cpus),frequencies=len(pg),
                  call_seconds=call_seconds, retained=len(sampled),
                  acceptance=float(accepted.mean()), reference_comparison=comparison,
                  prepare_seconds=times['inputs_ready_at']-started)
    write_json(Path(config['case_folder'])/f'group{group:03d}.json', record)
    return record


def partition_profile(iterator, config):
    from pyspark import BarrierTaskContext
    context = BarrierTaskContext.get()
    groups = list(iterator)
    if len(groups) != 1:
        raise ValueError('A sampling-only worker must have one group')
    yield profile_group(groups[0], config, context.barrier)


def worker(case_json):
    config = json.loads(case_json.read_text())
    started = time.perf_counter()
    pythonpath = configure_spark_environment(config['source'])
    os.environ.setdefault('SPARK_LOCAL_HOSTNAME', 'localhost')
    os.environ.setdefault('SPARK_LOCAL_IP', '127.0.0.1')
    os.chdir(config['source'])
    spark = None
    if config['mode'] == 'dc_parallel':
        os.sched_setaffinity(0, set(config['cpus'][:config['groups']]))
        from pyspark.sql import SparkSession
        spark = (SparkSession.builder.master(f'local[{config["groups"]}]')
                 .appName('dts-mcmc-only-timing')
                 .config('spark.ui.enabled', 'false')
                 .config('spark.ui.showConsoleProgress', 'false')
                 .config('spark.driver.host', '127.0.0.1')
                 .config('spark.driver.bindAddress', '127.0.0.1')
                 .config('spark.python.worker.reuse', 'true')
                 .config('spark.pyspark.python', sys.executable)
                 .config('spark.executorEnv.PYTHONPATH', pythonpath)
                 .config('spark.executorEnv.OPENBLAS_NUM_THREADS', '1')
                 .config('spark.executorEnv.OMP_NUM_THREADS', '1')
                 .config('spark.executorEnv.MKL_NUM_THREADS', '1')
                 .config('spark.executorEnv.NUMBA_NUM_THREADS', '1').getOrCreate())
        spark.sparkContext.setLogLevel('ERROR')
    try:
        if spark is None:
            records = [profile_group(0, config)]
        else:
            records = (spark.sparkContext.parallelize(range(config['groups']), config['groups'])
                       .barrier().mapPartitions(lambda it: partition_profile(it, config)).collect())
        expected = config['n']//2-1
        if sum(x['frequencies'] for x in records) != expected:
            raise ValueError('Replay does not cover the complete positive spectrum')
        sampling_wall = (max(x['sampling_finished_at'] for x in records)
                         - min(x['sampling_started_at'] for x in records))
        result = dict(mode=config['mode'], iterations=config['length'], burn=config['burn'],
                      repetition=config['repetition'], date=config['date'], n=config['n'],
                      sampling_wall_seconds=sampling_wall,
                      summed_sampling_seconds=sum(x['compiled_sampling_seconds'] for x in records),
                      max_sampling_seconds=max(x['compiled_sampling_seconds'] for x in records),
                      max_compile_seconds=max(x['loop_compile_seconds'] for x in records),
                      elapsed_seconds=time.perf_counter()-started,
                      all_reference_replays_equal=all(
                          x['reference_comparison'] is not None and
                          x['reference_comparison']['max_absolute_error'] <= 1e-12 for x in records)
                          if config['length'] == config['reference_iterations'] else None,
                      workers=records)
        write_json(Path(config['case_folder'])/'completed.json', result)
        print(json.dumps({k:v for k,v in result.items() if k != 'workers'}), flush=True)
    finally:
        if spark is not None:
            spark.stop()


def controller(args):
    from tqdm import tqdm
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out/'controller.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    previous = json.loads((args.benchmark/'config.json').read_text())
    prepared = json.loads((args.benchmark/'inputs'/args.date/'prepared.json').read_text())
    config = {key:previous[key] for key in ['source','seed','groups','cpus','source_sha256']}
    if args.cpus is not None:
        if len(args.cpus)<config['groups'] or len(set(args.cpus))!=len(args.cpus):
            raise ValueError('A distinct physical CPU is required for every group')
        config['cpus']=args.cpus
    config.update(version=1, benchmark=str(args.benchmark), date=args.date,
                  n=prepared['n'], reference_repetition=args.reference_repetition,
                  reference_iterations=previous['iterations'], reference_burn=previous['burn'],
                  lengths=args.lengths or sorted({2000,previous['iterations']}),
                  repetitions=args.repetitions,modes=args.modes,
                  profiler_sha256=sha256(__file__), quiet_seconds=args.quiet_seconds,
                  interference_policy=args.interference_policy,
                  quiet_timeout=args.quiet_timeout,
                  scope='Fixed fitted proposals; full spectrum; explicit compilation excluded; barrier-synchronised MCMC execution',
                  reference_stream='Same seed and proposal as reference repetition for every timing repeat')
    for name, expected in config['source_sha256'].items():
        if sha256(Path(config['source'])/name) != expected:
            raise ValueError('Frozen numerical source changed')
    path = args.out/'config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Profile configuration changed; use a new directory')
    write_json(path, config)
    jobs = [(length, rep, mode) for length in config['lengths']
            for rep in range(1, args.repetitions+1) for mode in args.modes]
    # Alternate which algorithm runs first between paired repetitions.
    jobs.sort(key=lambda x: (x[0],x[1],(x[2]=='full') != bool(x[1]%2)))
    for length, rep, mode in tqdm(jobs, desc='Sampling-only timing'):
        folder = args.out/'cases'/f'{mode}_i{length}_r{rep}'
        if (folder/'completed.json').exists():
            continue
        folder.mkdir(parents=True, exist_ok=True)
        wait_for_quiet(folder,args.quiet_seconds,args.quiet_timeout)
        burn = previous['burn'] if length == previous['iterations'] else min(500,length//4)
        case = dict(config, mode=mode, length=length, burn=burn, repetition=rep,
                    case_folder=str(folder))
        write_json(folder/'case.json', case)
        command = [sys.executable,str(Path(__file__).resolve()),'--case-json',str(folder/'case.json')]
        with (folder/'run.log').open('w') as log:
            child = subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            write_json(folder/'process.json',dict(pid=child.pid,command=command,started_at=time.time()))
            try:
                code = monitor_process(child,folder/'resource_monitor.json',args.case_max_seconds,args.interference_policy)
            except BaseException:
                stop_case(child)
                raise
        monitoring=json.loads((folder/'resource_monitor.json').read_text())
        if code or (monitoring['interference'] and args.interference_policy=='exclude'):
            if (folder/'completed.json').exists():
                (folder/'completed.json').rename(folder/'excluded_completed.json')
            raise RuntimeError(f'Profile failed or encountered competing computation: {folder}')
    print(f'Completed sampling-only profiles: {args.out}',flush=True)


if __name__ == '__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Profile interrupted; completed cases remain resumable')
    signal.signal(signal.SIGTERM, interrupted)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--date',default='20211013')
    parser.add_argument('--lengths',type=int,nargs='+',
                        help='Defaults to 2,000 and the saved reference chain length')
    parser.add_argument('--repetitions',type=int,default=3)
    parser.add_argument('--modes',nargs='+',choices=['full','dc_parallel'],default=['full','dc_parallel'])
    parser.add_argument('--cpus',nargs='+',type=int)
    parser.add_argument('--reference-repetition',type=int,default=1)
    parser.add_argument('--quiet-seconds',type=int,default=60)
    parser.add_argument('--quiet-timeout',type=int,default=86400)
    parser.add_argument('--interference-policy',choices=['exclude','record'],default='exclude')
    parser.add_argument('--case-max-seconds',type=int,default=7200)
    parser.add_argument('--case-json',type=Path)
    arguments=parser.parse_args()
    worker(arguments.case_json) if arguments.case_json else controller(arguments)
