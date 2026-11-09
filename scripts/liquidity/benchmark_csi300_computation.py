#!/usr/bin/env python3
"""Bounded, resumable timing comparison using the pinned production DTS code.

All likelihood evaluations use every assigned positive frequency. DC shards
find their MAPs independently, without a serial full-data MAP prerequisite.
The posterior, fixed MH kernel and consensus rule remain unchanged. The
production numerical source snapshot and input hashes are recorded in the output.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import platform
import fcntl
from contextlib import nullcontext

for _name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[_name] = '1'
os.environ.setdefault('MPLCONFIGDIR', '/tmp/dts-computation-benchmark-mpl')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/dts-computation-benchmark-numba')
os.environ.setdefault('PYSPARK_PYTHON', sys.executable)

import numpy as np
from scipy.fft import rfft


def write_json(path, values):
    path = Path(path)
    temporary = path.with_name(path.name+f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(values, indent=2)+'\n')
    temporary.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def backend(source):
    source = str(Path(source).resolve())
    # Spark places the current working directory before PYTHONPATH. Merely
    # finding the source somewhere in sys.path does not make it authoritative.
    sys.path[:] = [source]+[entry for entry in sys.path if entry != source]
    from scripts.liquidity import probe_csi300_fractional_changes as probe
    from scripts.liquidity import _spectral_coordinates as coordinates
    from scripts.liquidity import _spectral_covariance as covariance
    from scripts.liquidity.compare_csi300_boundary_spark import sampling_quality
    import dts.mcmc as mcmc_module
    for module in (probe, coordinates, covariance, mcmc_module):
        if not Path(module.__file__).resolve().is_relative_to(Path(source)):
            raise RuntimeError(f'Numerical module loaded outside frozen source: {module.__file__}')
    return probe, coordinates, covariance, sampling_quality


def pin_cpu(cpu):
    """Rebind existing numerical pool threads when Spark reuses a worker."""
    for task in Path('/proc/self/task').iterdir():
        try:
            os.sched_setaffinity(int(task.name), {int(cpu)})
        except ProcessLookupError:
            pass


def warm(source, cpu):
    """Initialize numerical pools on one hardware thread, as in production."""
    pin_cpu(cpu)
    probe, coordinates, covariance, _ = backend(source)
    omega = 2*np.pi*np.arange(1, 33)/128
    q = 4*np.sin(omega/2)**2
    pg = np.ones_like(q)
    native = np.array([-.5, -4., 0., .5])
    z = coordinates.to_coordinates(native)
    probe.target(native, q, pg, 1)
    probe.spectrum_hessian(native, q, pg, 1)
    coordinates.coordinate_sampler(z, np.eye(4)*1e-4, pg, omega, 1,
                                   8, 2, 791, kernel='mixture', center=z)
    covariance.coordinate_densities(np.array([z]), q, pg, 1)
    return os.getpid()


def warm_partition(iterator, source, cpus):
    from pyspark import BarrierTaskContext
    context = BarrierTaskContext.get()
    index = context.partitionId()
    pid = warm(source, cpus[index % len(cpus)])
    context.barrier()
    probe,coordinates,covariance,_=backend(source)
    yield dict(partition=index,pid=pid,numerical_origins=[str(Path(m.__file__).resolve())
               for m in (probe,coordinates,covariance)])


def spectrum(values):
    n = len(values)
    transformed = rfft(values, workers=1)[1:n//2]
    pg = np.abs(transformed)**2/(2*np.pi*n)
    omega = 2*np.pi*np.arange(1, n//2)/n
    return 4*np.sin(omega/2)**2, pg, omega


OPTIMIZATION_POLICIES = {
    'robust': dict(starts='production pilot and complete-frequency multi-start', options=None),
    'two-start': dict(starts=[[-.99, -4., 0., .9], [-.999, -5., 0., .95]], options=None),
    'single-start': dict(starts=[[-.99, -4., 0., .9]], options=None),
    'single-loose': dict(starts=[[-.99, -4., 0., .9]],
                         options=dict(maxiter=100, ftol=1e-8, gtol=1e-6, maxls=30)),
}


def fit_with_budget(probe, q, pg, groups, policy='robust', starts=None):
    """Timing-only initialization experiment; retain the complete likelihood.

    Reduced budgets omit the exploratory pilot and the unused ARIMA null fit.
    Native bounds, derivative calculations, Newton polishing, induced-prior
    transformation and proposal construction are the production routines.
    """
    settings = OPTIMIZATION_POLICIES[policy]
    if policy == 'robust':
        return probe.fit_spectrum(q, pg, groups=groups, starts=starts)
    candidates = settings['starts'] if starts is None else starts
    original_minimize = None
    if settings['options'] is not None:
        original_minimize = probe.minimize

        def limited_minimize(*args, **kwargs):
            kwargs['options'] = {**kwargs.get('options', {}), **settings['options']}
            return original_minimize(*args, **kwargs)

        probe.minimize = limited_minimize
    try:
        fits = [probe.optimize_spectrum_start(np.asarray(point), q, pg, groups)
                for point in candidates]
    finally:
        if original_minimize is not None:
            probe.minimize = original_minimize
    fits.sort(key=lambda value: value['negative_log_posterior'])
    return dict(fits[0], map_candidates=fits, full_frequency_starts=len(fits),
                optimization_policy=policy, unused_null_fit_skipped=True)


def full_data_initialization(method, probe, q=None, pg=None, policy='robust'):
    """Only the full-data chain needs a full-data proposal fit."""
    if method != 'full':
        return None
    fit = fit_with_budget(probe, q, pg, groups=1, policy=policy)
    if fit['newton_distance'] > .01:
        raise ValueError('Unresolved full-data MAP')
    return fit


def fit_and_sample(q, pg, omega, groups, group, config, start=None, existing_fit=None):
    """Production local MAP, fixed mixture MH, and covariance rules."""
    probe, coordinates, covariance, quality_function = backend(config['source'])
    started = time.perf_counter()
    fit = existing_fit if existing_fit is not None else fit_with_budget(
        probe, q, pg, groups=groups, policy=config.get('optimization_policy', 'robust'),
        starts=None if start is None else [start])
    if fit['newton_distance'] > .01:
        raise ValueError(f'Unresolved MAP: group {group}')
    native_map_seconds = time.perf_counter()-started
    coordinate_started = time.perf_counter()
    transformed = coordinates.coordinate_map(fit['theta'], q, pg, groups)
    coordinate_map_seconds = time.perf_counter()-coordinate_started
    map_seconds = time.perf_counter()-started
    center = np.asarray(transformed['theta'])
    proposal = np.asarray(transformed['covariance'])*(2.38**2/4)
    started = time.perf_counter()
    if config.get('detailed_timing'):
        from scripts.liquidity._computation_timing import coordinate_step_timing
        timing_context = coordinate_step_timing(coordinates)
    else:
        timing_context = nullcontext({})
    sampler_started = time.perf_counter()
    with timing_context as step_times:
        draws, logp, accepted = coordinates.coordinate_sampler(
            center, proposal, pg, omega, groups, config['iterations'], config['burn'],
            config['seed']+config['repetition']*100_000+group*1000,
            backend='compiled', kernel='mixture', center=center)
    sampler_finished = time.perf_counter()
    sampling_seconds = time.perf_counter()-started
    if not np.isfinite(logp).all():
        raise ValueError('Nonfinite MCMC log density')
    result = dict(group=group, map_seconds=map_seconds,
                  sampling_seconds=sampling_seconds, acceptance=float(accepted.mean()),
                  native_map=fit['theta'], coordinate_map=transformed,
                  retained=len(draws), map_distance=fit['newton_distance'])
    if config.get('detailed_timing'):
        result.update(step_times, native_map_seconds=native_map_seconds,
                      coordinate_map_seconds=coordinate_map_seconds,
                      sampling_prepare_seconds=step_times['inputs_ready_at']-sampler_started,
                      sampling_extract_seconds=sampler_finished-step_times['sampling_finished_at'])
    covariance_seconds = 0.
    regularization = None
    if groups > 1:
        started = time.perf_counter()
        moments = covariance.quadrature_moments(q, pg, groups, center,
                                               transformed['covariance'])
        matrix, regularization = covariance.regularize_covariance(
            draws[None, ...], moments['covariance'])
        covariance_seconds = time.perf_counter()-started
        result.update(quadrature_order=moments['order'], **regularization)
    else:
        matrix = np.cov(draws, rowvar=False)
    destination = Path(config['case_folder'])/'shards'
    write_started = time.perf_counter()
    np.save(destination/f'group{group:03d}_draws.npy', draws)
    np.save(destination/f'group{group:03d}_covariance.npy', matrix)
    result['covariance_seconds'] = covariance_seconds
    if config.get('detailed_timing'):
        result['write_seconds'] = time.perf_counter()-write_started
        result['worker_finished_at'] = time.perf_counter()
        actual_cpus = set()
        for task in Path('/proc/self/task').iterdir():
            try:
                actual_cpus.update(os.sched_getaffinity(int(task.name)))
            except ProcessLookupError:
                pass
        expected_cpu = config['cpus'][group % config['slots']]
        if actual_cpus != {expected_cpu}:
            raise ValueError(f'Numerical worker threads escaped CPU {expected_cpu}: {actual_cpus}')
        result.update(pid=os.getpid(),thread_cpus=sorted(actual_cpus),
                      numerical_origins=[str(Path(m.__file__).resolve())
                      for m in (probe,coordinates,covariance)])
    write_json(destination/f'group{group:03d}.json', result)
    return result


def spark_group(group, config):
    pin_cpu(config['cpus'][group % config['slots']])
    path = Path(config['case_folder'])/'frequencies'/f'shard{group:03d}.npz'
    started = time.perf_counter()
    with np.load(path) as values:
        q, pg, omega = values['q'], values['periodogram'], values['omega']
    load_seconds = time.perf_counter()-started
    result = fit_and_sample(q, pg, omega, config['groups'], group, config)
    if config.get('detailed_timing'):
        result.update(load_seconds=load_seconds, worker_started_at=started)
        write_json(Path(config['case_folder'])/'shards'/f'group{group:03d}.json', result)
    return result


def uses_spark(method):
    return method.startswith('dc_') or method == 'spark_g1'


def prepare(args):
    if not 0 <= args.burn < args.iterations or args.iterations-args.burn < 512:
        raise ValueError('At least 512 retained draws are needed for covariance checks')
    if len(set(args.cpus)) < args.slots or args.slots > args.groups:
        raise ValueError('Provide a distinct physical CPU for every parallel slot')
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    production_config = json.loads((args.production/'run_config.json').read_text())
    baseline_config = json.loads((args.reference/'run_config.json').read_text())
    source = out/'source'
    if not source.exists():
        shutil.copytree(args.source, source)
        if args.fft_backend == 'jax':
            # Only the requested FFT implementation changes; the likelihood,
            # sampler and covariance routines stay pinned to production.
            shutil.copy2(Path(__file__).resolve().parents[2]/'dts/dfft_blocks.py',
                         source/'dts/dfft_blocks.py')
        if args.detailed_timing:
            shutil.copy2(Path(__file__).with_name('_computation_timing.py'),
                         source/'scripts/liquidity/_computation_timing.py')
    required = ['dts/mcmc.py', 'dts/aggregation.py', 'dts/dfft_blocks.py',
                'scripts/liquidity/probe_csi300_fractional_changes.py',
                'scripts/liquidity/_spectral_coordinates.py',
                'scripts/liquidity/_spectral_covariance.py']
    hashes = {name: sha256(source/name) for name in required}
    fft_override = None
    for name, value in hashes.items():
        expected = production_config['source_sha256'].get(name)
        if name == 'dts/dfft_blocks.py' and args.fft_backend == 'jax':
            current = Path(__file__).resolve().parents[2]/name
            if value != sha256(current):
                raise ValueError('JAX FFT implementation changed; use a new output directory')
            fft_override = dict(path=str(current), production_sha256=expected, experiment_sha256=value)
            continue
        if expected is not None and value != expected:
            raise ValueError(f'Production numerical code changed: {name}')
    timing_source = None
    if args.detailed_timing:
        timing_source = sha256(source/'scripts/liquidity/_computation_timing.py')
        if timing_source != sha256(Path(__file__).with_name('_computation_timing.py')):
            raise ValueError('Timing instrumentation changed; use a new output directory')
    config = dict(version=3 if args.detailed_timing else 2, production=str(args.production), reference=str(args.reference),
                  source=str(source), dates=args.dates, repetitions=args.repetitions,
                  iterations=args.iterations, burn=args.burn, groups=args.groups,
                  parallel_slots=args.slots, seed=args.seed, cpus=args.cpus,
                  source_sha256=hashes, controller_sha256=sha256(__file__),
                  optimization_policy=args.optimization_policy,
                  optimization_settings=OPTIMIZATION_POLICIES[args.optimization_policy],
                  fft_backend=args.fft_backend, fft_source_override=fft_override,
                  fft_compilation='JAX actual-shape compilation included in FFT time; kernels cached per worker' if args.fft_backend=='jax' else 'Native NumPy FFT kernels',
                  baseline_input=baseline_config['input'],
                  methods=list(args.methods),
                  dc_initialization='Independent shard-local multi-start MAP; no full-data MAP',
                  timing_scope='Fresh FFT, MAP, MCMC, diagnostics and consensus; Spark initialization separately included in all-in time; input liquidity construction and clock calibration excluded',
                  compilation='Tiny-array common initialization excluded; shape-specific JAX compilation included in the measured sampling stage')
    if args.detailed_timing:
        config.update(detailed_timing=True, timing_source_sha256=timing_source,
                      interference_policy=args.interference_policy,
                      compilation='Driver/worker initialization included in startup; explicit actual-shape loop compilation and synchronous MCMC execution measured separately',
                      timing_scope='Fresh FFT, independent MAP/proposals, compilation, MCMC, covariance and consensus; startup and driver imports included in all-in time; liquidity construction and clock calibration excluded',
                      quiet_seconds=args.quiet_seconds, quiet_timeout=args.quiet_timeout,
                      case_max_seconds=args.case_max_seconds)
    if args.serial_dc and 'dc_serial' not in config['methods']:
        config['methods'].append('dc_serial')
    path = out/'config.json'
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError('Use a new directory after a benchmark protocol change')
    write_json(path, config)
    if not (out/'host.json').exists():
        write_json(out/'host.json', dict(host=platform.node(), platform=platform.platform(),
                   cpu_topology=subprocess.check_output(['lscpu', '-p=CPU,CORE,SOCKET'], text=True),
                   selected_cpus=args.cpus, started_at=time.time(),
                   isolation='Pinned physical cores; single-threaded numerical workers; competing compute processes monitored; no exclusive scheduler reservation'))
    for date in args.dates:
        original = args.production/'days'/date
        folder = out/'inputs'/date
        folder.mkdir(parents=True, exist_ok=True)
        if (folder/'prepared.json').exists():
            continue
        response_path = original/'response.npy'
        y = np.load(response_path, mmap_mode='r')
        if len(y) < 10_000_000 or len(y) % 2:
            raise ValueError('The benchmark requires complete long training histories')
        print(f'Prepare {date}: {len(y):,} changes', flush=True)
        q, pg, omega = spectrum(y)
        for name, values in [('q', q), ('periodogram', pg), ('omega', omega)]:
            np.save(folder/f'{name}.npy', values)
        saved = json.loads((args.reference/'days'/date/'fit.json').read_text())
        if str(saved['train_last']) >= date:
            raise ValueError('Training data must precede the forecast date')
        raw = np.load(Path(baseline_config['input'])/'spread.npy', mmap_mode='r')
        profile = np.load(args.reference/'days'/date/'profile.npy')
        levels = np.log(raw[:saved['train_stop_day']])-profile
        levels = levels[np.isfinite(levels)]
        levels = levels[len(levels) % 2:]
        levels = (levels-levels.mean())/levels.std()
        np.save(folder/'levels.npy', levels)
        write_json(folder/'prepared.json', dict(date=date, n=len(y), level_n=len(levels),
                   response=str(response_path), response_sha256=sha256(response_path),
                   reference_cutoff=saved['train_stop_day'], train_last=saved['train_last'],
                   no_future_data=str(saved['train_last']) < date,
                   periodogram_sha256=sha256(folder/'periodogram.npy')))
    shutil.copy2(__file__, out/Path(__file__).name)
    return config


def execute_case(args):
    entry_started = time.perf_counter()
    config = json.loads((args.out/'config.json').read_text())
    if config.get('version') not in (2, 3):
        raise ValueError('Revised initialization requires a new benchmark directory')
    config.update(repetition=args.rep, seed=config['seed'], slots=1,
                  case_folder=str(args.out/'cases'/args.date/f'{args.method}_r{args.rep}'))
    if args.method == 'spark_g1':
        config['groups'] = 1
    folder = Path(config['case_folder'])
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'completed.json').exists():
        print(f'Already completed: {folder}', flush=True)
        return
    inputs = args.out/'inputs'/args.date
    metadata = json.loads((inputs/'prepared.json').read_text())
    write_json(folder/'progress.json', dict(stage='initialization', time=time.time()))
    warm_started = time.perf_counter()
    spark_method=uses_spark(args.method)
    warm_seconds=0.
    if not spark_method:
        warm(config['source'],config['cpus'][0])
        warm_seconds=time.perf_counter()-warm_started
    init_started = time.perf_counter()
    spark = None
    if spark_method:
        config['slots'] = 1 if args.method in ('dc_serial', 'spark_g1') else config['parallel_slots']
        selected = config['cpus'][:config['slots']]
        os.sched_setaffinity(0, set(selected))
        # A Python '-m pyspark.daemon' worker prepends its cwd. Use the frozen
        # package as cwd, and launch Java before creating driver JAX threads.
        os.chdir(config['source'])
        os.environ['PYTHONPATH'] = config['source']+os.pathsep+str(Path(__file__).resolve().parents[2])
        from pyspark.sql import SparkSession
        spark = (SparkSession.builder.master(f'local[{config["slots"]}]')
                 .appName('dts-fair-computation-timing')
                 .config('spark.ui.enabled', 'false')
                 .config('spark.ui.showConsoleProgress', 'false')
                 .config('spark.driver.host', '127.0.0.1')
                 .config('spark.driver.bindAddress', '127.0.0.1')
                 .config('spark.driver.maxResultSize', '2g')
                 .config('spark.python.worker.reuse', 'true')
                 .config('spark.pyspark.python', sys.executable)
                 .config('spark.executorEnv.PYTHONPATH', os.environ['PYTHONPATH'])
                 .config('spark.executorEnv.OPENBLAS_NUM_THREADS', '1')
                 .config('spark.executorEnv.OMP_NUM_THREADS', '1')
                 .config('spark.executorEnv.MKL_NUM_THREADS', '1')
                 .config('spark.executorEnv.NUMBA_CACHE_DIR', os.environ['NUMBA_CACHE_DIR'])
                 .config('spark.executorEnv.MPLCONFIGDIR', os.environ['MPLCONFIGDIR'])
                 .getOrCreate())
        spark.sparkContext.setLogLevel('ERROR')
        src, cpus = config['source'], selected
        try:
            warm_records = spark.sparkContext.parallelize(range(config['slots']), config['slots']).barrier().mapPartitions(
                lambda it: warm_partition(it, src, cpus)).collect()
            if len({x['pid'] for x in warm_records}) != config['slots']:
                raise ValueError('Not every parallel worker was initialized')
            write_json(folder/'startup_workers.json',warm_records)
        except BaseException:
            spark.stop()
            raise
    startup_seconds = time.perf_counter()-init_started
    runtime_startup_seconds = startup_seconds
    try:
        if spark_method:
            warm_started=time.perf_counter()
            warm(config['source'],config['cpus'][0])
            warm_seconds=time.perf_counter()-warm_started
            os.sched_setaffinity(0,set(selected))
        probe,coordinates,covariance,sampling_quality=backend(config['source'])
        from scripts.liquidity.compare_csi300_boundary_arima101 import fit_arma
        from dts.aggregation import consensus
    except BaseException:
        if spark is not None:spark.stop()
        raise
    driver_import_seconds = 0.
    if config.get('detailed_timing'):
        process_started = float(os.environ.get('DTS_TIMING_PROCESS_STARTED', entry_started))
        driver_import_seconds = max(0.,entry_started-process_started)
        startup_seconds = time.perf_counter()-process_started
    try:
        fit_started = time.perf_counter()
        write_json(folder/'progress.json', dict(stage='fft', time=time.time()))
        started = time.perf_counter()
        if uses_spark(args.method):
            from dts.dfft_blocks import spark_fft_contiguous_blocks
            from scripts.liquidity.compare_csi300_boundary_spark import (
                fft_partitions_for, load_input_block, frequency_chunks, cache_frequency_group)
            n, groups = metadata['n'], config['groups']
            partitions = fft_partitions_for(n, 256)
            (folder/'frequencies').mkdir(exist_ok=True)
            response = metadata['response']
            blocks = spark.sparkContext.parallelize(range(partitions), partitions).map(
                lambda b: load_input_block(b, response, n, partitions))
            if config.get('fft_backend', 'numpy') == 'jax':
                transformed = spark_fft_contiguous_blocks(blocks, n, partitions, backend='jax')
            else:
                transformed = spark_fft_contiguous_blocks(blocks, n, partitions)
            chunks = transformed.flatMap(lambda x: frequency_chunks(x, n, partitions, groups)).groupByKey(groups)
            destination, reference = str(folder), str(inputs)
            checks = chunks.map(lambda x: cache_frequency_group(x, destination, reference, n, groups)).collect()
            if sum(x['ordinates'] for x in checks) != n//2-1:
                raise ValueError('A Fourier ordinate was omitted or repeated')
            fft_error = max(x['max_absolute_error'] for x in checks)
            q = pg = None  # The driver never fits the full target for DC.
            omega = None
        else:
            response = metadata['response'] if args.method != 'arima101' else inputs/'levels.npy'
            y = np.load(response)
            q, pg, omega = spectrum(y)
            n = len(y)
            fft_error = 0.
            del y
        fft_seconds = time.perf_counter()-started
        write_json(folder/'progress.json', dict(stage='optimization', time=time.time()))
        map_started = time.perf_counter()
        if args.method == 'arima011':
            fitted = probe.fit_null_spectrum(q, pg, [-.1, 0.])
            if not fitted['null_success']:
                raise ValueError('Unresolved ARIMA(0,1,1) MAP')
            map_seconds = time.perf_counter()-map_started
            result = dict(estimator='Whittle MAP', map=fitted, retained=0)
        elif args.method == 'arima101':
            fitted = fit_arma(q, pg, n)
            map_seconds = time.perf_counter()-map_started
            result = dict(estimator='Whittle MAP', map=fitted, retained=0)
        else:
            global_fit = full_data_initialization(args.method, probe, q, pg,
                                                  config.get('optimization_policy', 'robust'))
            global_map_seconds = time.perf_counter()-map_started if global_fit is not None else 0.
            result = dict(estimator='MCMC', global_map_seconds=global_map_seconds,
                          initialization='full-data MAP' if global_fit is not None
                                         else config['dc_initialization'])
            if global_fit is not None:
                result['common_full_map'] = global_fit
            (folder/'shards').mkdir(exist_ok=True)
            write_json(folder/'progress.json', dict(stage='mcmc', time=time.time()))
            action_started = time.perf_counter()
            if args.method == 'full':
                records = [fit_and_sample(q, pg, omega, 1, 0, config, global_fit['theta'], existing_fit=global_fit)]
            else:
                worker_config = dict(config)
                records = spark.sparkContext.parallelize(range(config['groups']), config['groups']).map(
                    lambda group: spark_group(group, worker_config)).collect()
            action_seconds = time.perf_counter()-action_started
            if config.get('detailed_timing'):
                from scripts.liquidity._computation_timing import critical_path_breakdown
                result['worker_wall_breakdown'] = critical_path_breakdown(
                    records, action_seconds, config['slots'] > 1)
            write_json(folder/'progress.json', dict(stage='consensus', time=time.time()))
            started = time.perf_counter()
            sampled = [np.load(folder/'shards'/f'group{x["group"]:03d}_draws.npy') for x in records]
            if args.method == 'full':
                combined = sampled[0]
            else:
                matrices = [np.load(folder/'shards'/f'group{x["group"]:03d}_covariance.npy') for x in records]
                combined = consensus(sampled, ridge=0., covariances=matrices)
            native = coordinates.from_coordinates(combined)
            quality = sampling_quality(native[None, ...])
            np.save(folder/'posterior_draws.npy', native)
            aggregation_seconds = time.perf_counter()-started
            result.update(action_seconds=action_seconds, aggregation_seconds=aggregation_seconds,
                          shard_map_seconds=[x['map_seconds'] for x in records],
                          shard_sampling_seconds=[x['sampling_seconds'] for x in records],
                          shard_covariance_seconds=[x['covariance_seconds'] for x in records],
                          acceptance=float(np.mean([x['acceptance'] for x in records])),
                          retained=len(native), quality=quality,
                          posterior_mean=native.mean(axis=0).tolist(),
                          posterior_sd=native.std(axis=0, ddof=1).tolist())
            map_seconds = global_map_seconds
        fit_seconds = time.perf_counter()-fit_started
        result.update(date=args.date, method=args.method, repetition=args.rep,
                      n=int(n), groups=1 if args.method=='full' else config['groups'] if uses_spark(args.method) else 0,
                      slots=config['slots'], fft_seconds=fft_seconds, map_seconds=map_seconds,
                      fit_seconds=fit_seconds, startup_seconds=startup_seconds,
                      all_in_seconds=fit_seconds+startup_seconds,
                      iterations=config['iterations'] if result['estimator']=='MCMC' else 0,
                      burn=config['burn'] if result['estimator']=='MCMC' else 0,
                      fft_max_absolute_error=fft_error)
        if config.get('detailed_timing'):
            result.update(driver_import_seconds=driver_import_seconds,
                          driver_warm_seconds=warm_seconds,
                          runtime_startup_seconds=runtime_startup_seconds,
                          driver_pid=os.getpid(), finished_at=time.time(),
                          interference_policy=config.get('interference_policy','exclude'))
        write_json(folder/'completed.json', result)
        write_json(folder/'progress.json', dict(stage='completed', time=time.time()))
        print(json.dumps({k:result[k] for k in ['date','method','repetition','n','fit_seconds','startup_seconds']}) ,flush=True)
    finally:
        if spark is not None:
            spark.stop()


def process_ticks():
    values = {}
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            raw=(p/'stat').read_text()
            comm=raw[raw.index('(')+1:raw.rindex(')')]
            fields=raw[raw.rindex(')')+2:].split()
            values[int(p.name)]=(int(fields[1]), int(fields[11])+int(fields[12]), p.stat().st_uid, comm)
        except (OSError, ValueError, IndexError):
            pass
    return values


def is_compute_process(command):
    # Firefox's "RDD Process" is a decoder, not the R interpreter.
    return command in {'R', 'Rscript', 'R.bin'} or command.startswith(
        ('python', 'java', 'julia', 'matlab', 'ray::', 'spark', 'ffmpeg'))


def descendant_pids(current, root):
    descendants = {root}
    while True:
        expanded = descendants | {pid for pid, value in current.items() if value[0] in descendants}
        if expanded == descendants:
            return descendants
        descendants = expanded


def competing_activity(previous, current, seconds, excluded=()):
    active = []
    ticks_per_second = os.sysconf('SC_CLK_TCK')
    for pid, (ppid, ticks, uid, comm) in current.items():
        if pid in excluded or pid not in previous:
            continue
        cores = (ticks-previous[pid][1])/ticks_per_second/seconds
        if (cores > .25 and is_compute_process(comm)) or (uid != os.getuid() and cores > 1.):
            active.append(dict(pid=pid, uid=uid, command=comm, cpu_cores=cores))
    return active


def wait_for_quiet(folder, quiet_seconds, timeout):
    """Wait for measured inactivity; a timestamp or lock alone is insufficient."""
    if quiet_seconds <= 0:
        return
    started = previous_time = quiet_since = time.monotonic()
    previous = process_ticks()
    last_report = started-30
    while True:
        time.sleep(2)
        now = time.monotonic()
        current = process_ticks()
        excluded = descendant_pids(current, os.getpid())
        active = competing_activity(previous, current, now-previous_time, excluded)
        if active:
            quiet_since = now
        record = dict(wait_seconds=now-started, quiet_seconds=now-quiet_since,
                      required_quiet_seconds=quiet_seconds, active=active, finished=False)
        write_json(Path(folder)/'quiet_host.json', record)
        with (Path(folder)/'quiet_host_events.jsonl').open('a') as handle:
            handle.write(json.dumps(record)+'\n')
        if now-last_report >= 30:
            print(f'Quiet-host gate: {record["quiet_seconds"]:.0f}/{quiet_seconds}s quiet; '
                  f'{len(active)} competing processes; waited {now-started:.0f}s', flush=True)
            last_report = now
        if now-quiet_since >= quiet_seconds:
            write_json(Path(folder)/'quiet_host.json', dict(record, finished=True))
            return
        if now-started > timeout:
            raise RuntimeError('Quiet-host gate timed out; completed cases remain resumable')
        previous, previous_time = current, now


def stop_case(process):
    """Terminate only this benchmark's process group and known descendants."""
    descendants = descendant_pids(process_ticks(), process.pid)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    for pid in descendants-{process.pid}:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def monitor_process(process, log_path, max_seconds, interference_policy='exclude'):
    if interference_policy not in {'exclude','record'}:
        raise ValueError('Unknown interference policy')
    previous=process_ticks(); previous_time=time.monotonic(); start=previous_time
    interference=[]; checks=[]
    while process.poll() is None:
        time.sleep(2)
        current=process_ticks(); now=time.monotonic()
        descendants=descendant_pids(current, process.pid)|{os.getpid()}
        active=competing_activity(previous,current,now-previous_time,descendants)
        if active:interference.extend(active)
        checks.append(dict(elapsed=now-start,load1=os.getloadavg()[0],competing_compute_processes=active))
        write_json(log_path, dict(checks=checks,interference=interference,finished=False,policy=interference_policy))
        if now-start>max_seconds or (interference_policy=='exclude' and len(interference)>=3):
            reason='time-budget' if now-start>max_seconds else 'competing-computation'
            write_json(log_path,dict(checks=checks,interference=interference,finished=False,
                                     policy=interference_policy,stopped_for=reason))
            stop_case(process)
            raise RuntimeError('Timing case stopped: time budget exceeded or competing computation detected')
        previous,previous_time=current,now
    write_json(log_path,dict(checks=checks,interference=interference,finished=True,exit_code=process.returncode,policy=interference_policy))
    return process.returncode


def controller(args):
    from tqdm import tqdm
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out/'controller.lock').open('a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError as exc:
        raise RuntimeError('Another controller is already using this output directory') from exc
    config=prepare(args)
    if args.prepare_only:
        print(f'Prepared and verified inputs/source: {args.out}', flush=True)
        return
    source_script=Path(__file__).resolve()
    jobs=[(date,rep,method) for date in config['dates'] for rep in range(1,config['repetitions']+1) for method in config['methods']]
    # Rotate order within repetitions, preserving the same dates for all methods.
    rng=np.random.default_rng(config['seed'])
    jobs=[jobs[i] for i in rng.permutation(len(jobs))]
    started=time.monotonic()
    for index,(date,rep,method) in enumerate(tqdm(jobs, desc='Matched timing cases'),1):
        folder=args.out/'cases'/date/f'{method}_r{rep}'
        if (folder/'completed.json').exists():continue
        folder.mkdir(parents=True,exist_ok=True)
        if config.get('detailed_timing'):
            wait_for_quiet(folder,config['quiet_seconds'],config['quiet_timeout'])
        print(f'Case {index}/{len(jobs)}: {date} {method}, repetition {rep}',flush=True)
        command=[sys.executable,str(source_script),'--out',str(args.out),'--worker',
                 '--date',date,'--method',method,'--rep',str(rep)]
        with (folder/'run.log').open('w') as log:
            environment = dict(os.environ, DTS_TIMING_PROCESS_STARTED=str(time.perf_counter()))
            process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=environment)
            write_json(folder/'process.json',dict(pid=process.pid,started_at=time.time(),command=command))
            try:
                code=monitor_process(process,folder/'resource_monitor.json',
                    min(args.case_max_seconds,args.max_seconds-(time.monotonic()-started)),
                    config.get('interference_policy','exclude'))
            except BaseException:
                stop_case(process)
                raise
        if code:
            raise RuntimeError(f'Benchmark case failed ({code}): {folder}/run.log')
        if config.get('detailed_timing'):
            monitoring=json.loads((folder/'resource_monitor.json').read_text())
            if monitoring['interference'] and config.get('interference_policy','exclude')=='exclude':
                result=json.loads((folder/'completed.json').read_text())
                write_json(folder/'excluded.json',dict(reason='Competing computation during timing',result=result))
                (folder/'completed.json').unlink()
                raise RuntimeError(f'Timing excluded due to competing computation: {folder}')
    print(f'All cases completed: {args.out}',flush=True)


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True)
    base=Path.home()/'running/dts/artifacts/liquidity'
    p.add_argument('--source',type=Path,default=base/'csi300_computation_independent_maps_20261102_v1/source')
    p.add_argument('--production',type=Path,default=base/'csi300_boundary_spark_g20_full1000_onechain_v5')
    p.add_argument('--reference',type=Path,default=base/'csi300_boundary_full1000_v1')
    p.add_argument('--dates',nargs='+',default=['20211013','20260724'])
    p.add_argument('--repetitions',type=int,default=3)
    p.add_argument('--iterations',type=int,default=2000)
    p.add_argument('--burn',type=int,default=500)
    p.add_argument('--groups',type=int,default=20)
    p.add_argument('--slots',type=int,default=20)
    p.add_argument('--cpus',type=int,nargs='+',default=list(range(26,46)))
    p.add_argument('--seed',type=int,default=20261030)
    p.add_argument('--serial-dc',action='store_true')
    p.add_argument('--optimization-policy',choices=list(OPTIMIZATION_POLICIES),default='robust',
                   help='Initialization budget for a separate timing experiment; default preserves production fitting')
    p.add_argument('--fft-backend',choices=['numpy','jax'],default='numpy',
                   help='Local kernels in the same Spark block FFT; JAX compilation is included in FFT timing')
    p.add_argument('--methods',nargs='+',choices=['arima011','arima101','full','dc_parallel','dc_serial','spark_g1'],
                   default=['arima011','arima101','full','dc_parallel'],
                   help='Methods to measure; use full dc_serial dc_parallel for the MCMC table')
    p.add_argument('--max-seconds',type=int,default=7200)
    p.add_argument('--case-max-seconds',type=int,default=1800)
    p.add_argument('--detailed-timing',action='store_true',
                   help='Record explicit compilation/MCMC stages and include complete initialization in startup')
    p.add_argument('--quiet-seconds',type=int,default=30)
    p.add_argument('--quiet-timeout',type=int,default=43200)
    p.add_argument('--interference-policy',choices=['exclude','record'],default='exclude',
                   help='Exclude competing-load measurements, or retain and label monitored shared-host measurements')
    p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--worker',action='store_true')
    p.add_argument('--method',choices=['arima011','arima101','full','dc_parallel','dc_serial','spark_g1'])
    p.add_argument('--date')
    p.add_argument('--rep',type=int)
    return p.parse_args()


if __name__=='__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Benchmark interrupted; completed cases remain resumable')
    signal.signal(signal.SIGTERM, interrupted)
    arguments=parse_args()
    execute_case(arguments) if arguments.worker else controller(arguments)
