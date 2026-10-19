#!/usr/bin/env python3
"""Re-fit saved CSI 300 training histories with repository Spark Spectral DC.

Reuse the application response/cutoff exactly; run dts.dfft_blocks on the full
response, then independent exact-target MH on systematic frequency groups.
Boundary-aware coordinates and covariance regularization improve consensus.
One chain per shard is the production default; independent diagnostic chains
are requested explicitly. All output is checkpointed separately from the
existing Ray experiment.
"""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
os.environ.setdefault('SPARK_LOCAL_HOSTNAME', 'localhost')
os.environ.setdefault('SPARK_LOCAL_IP', '127.0.0.1')
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ.setdefault('MPLCONFIGDIR', '/tmp/dts-csi300-spark-mpl')
os.environ.setdefault('NUMBA_CACHE_DIR', '/tmp/dts-csi300-spark-numba')
os.environ.setdefault('NUMBA_NUM_THREADS','16')
import numpy as np
import pandas as pd
from tqdm import tqdm
from dts.dfft_blocks import spark_fft_contiguous_blocks

RUN_ARGUMENTS = ['reference', 'out', 'series', 'groups', 'chains', 'iterations',
                 'burn', 'seed', 'spark_slots', 'fft_partitions', 'dates', 'sampler_backend',
                 'reuse_from', 'max_iterations', 'extension_iterations', 'ess_target',
                 'reference_dates', 'reference_chains', 'reference_iterations',
                 'reference_burn', 'reference_max_iterations','history_workers',
                 'parameterization','mh_kernel','consensus_covariance','reuse_sampling_from']
AFFINITY_PREVIOUS_SHA256 = '19d64364e4d597b50c1818e721c9ec1f8bbf84aa907699ee13b9c652ea269f35'
SINGLE_CHAIN_PREVIOUS_SHA256 = 'eb3c9f0391d6246f505f71cbb468e870c2940db15540efa64da2c88b9e562978'
CHECKPOINT_PREVIOUS_SHA256 = '4a3592fd9baa850bb8e8a0a2fde3ddf635cc612746c63993fbb667d8749f61cb'
VALIDATION_PREVIOUS_SHA256 = 'f59f5691bea8d3c78ee8e6af8059015fa72a0de4587a3b358587b5e216ac21e8'
CONCURRENCY_PREVIOUS_SHA256 = 'aa35ea174943d042c454b5043ebe27cb8464e662ff0a91278f9834c130212067'
COORDINATE_PREVIOUS_SHA256 = 'cf147ae87b0639161954599f933997e0d9441ea03778140804c60a1c5bd864c4'
SAMPLING_PREVIOUS_SHA256 = '931383e8fa6ae22514695ba91ef07d911ec96402a05ba9e9d54a3341557b9125'
COVARIANCE_PREVIOUS_SHA256 = '3c6a8e478f38c1cfa36847a9df363ba8d6b1c3153a63a5c6f2f0e45161ea352c'
AGGREGATION_PREVIOUS_SHA256 = '566a3f87cc1f7bcbeb24b2ce1e615447a5e3b49d450c4323405ca7dd1962e3aa'
SAMPLING_CONTRACT_SHA256 = '15f68a27c063c945fee408d19e5c71adf1d516641e96228377b1719a59d21971'


def compatible_coordinate_update(previous,current):
    """Old native-coordinate fits keep their original target and MH kernel."""
    left,right=dict(previous),dict(current)
    old_sources,new_sources=left.pop('source_sha256'),right.pop('source_sha256')
    left.pop('source_history',None);right.pop('source_history',None)
    left.setdefault('parameterization','native');left.setdefault('mh_kernel','random-walk')
    for value in [left,right]:
        value.setdefault('consensus_covariance','empirical');value.setdefault('reuse_sampling_from',None)
    if left!=right or right['parameterization']!='native' or right['mh_kernel']!='random-walk':return False
    runner='scripts/liquidity/compare_csi300_boundary_spark.py'
    sampling='scripts/liquidity/_spectral_sampling.py'
    aggregation='dts/aggregation.py'
    return (old_sources.get(runner)==COORDINATE_PREVIOUS_SHA256
            and old_sources.get(sampling)==SAMPLING_PREVIOUS_SHA256
            and old_sources.get(aggregation,AGGREGATION_PREVIOUS_SHA256)==AGGREGATION_PREVIOUS_SHA256
            and {k:v for k,v in old_sources.items() if k not in {runner,sampling,aggregation}}
            == {k:v for k,v in new_sources.items() if k not in {runner,sampling,aggregation}})


def compatible_covariance_update(previous,current):
    """Existing empirical-weight runs keep their original aggregation results."""
    left,right=dict(previous),dict(current)
    old_sources,new_sources=left.pop('source_sha256'),right.pop('source_sha256')
    left.pop('source_history',None);right.pop('source_history',None)
    for value in [left,right]:
        value.setdefault('consensus_covariance','empirical');value.setdefault('reuse_sampling_from',None)
    runner='scripts/liquidity/compare_csi300_boundary_spark.py';aggregation='dts/aggregation.py'
    return (left==right and right['consensus_covariance']=='empirical'
            and old_sources.get(runner)==COVARIANCE_PREVIOUS_SHA256
            and old_sources.get(aggregation)==AGGREGATION_PREVIOUS_SHA256
            and {k:v for k,v in old_sources.items() if k not in {runner,aggregation}}
            == {k:v for k,v in new_sources.items() if k not in {runner,aggregation}})


def compatible_validation_update(previous,current):
    """Only the reviewed scale-aware checksum/production guard can reuse v2 fits."""
    left,right=dict(previous),dict(current)
    old_sources,new_sources=left.pop('source_sha256'),right.pop('source_sha256')
    left.pop('source_history',None);right.pop('source_history',None)
    left.setdefault('history_workers',0);right.setdefault('history_workers',0)
    key='scripts/liquidity/compare_csi300_boundary_spark.py'
    return (left==right and old_sources.get(key)==VALIDATION_PREVIOUS_SHA256
            and {k:v for k,v in old_sources.items() if k!=key}
            == {k:v for k,v in new_sources.items() if k!=key})


def compatible_concurrency_update(previous,current):
    left,right=dict(previous),dict(current)
    old_sources,new_sources=left.pop('source_sha256'),right.pop('source_sha256')
    left.pop('source_history',None);right.pop('source_history',None)
    left.setdefault('history_workers',0);right.setdefault('history_workers',0)
    key='scripts/liquidity/compare_csi300_boundary_spark.py'
    return (left==right and old_sources.get(key)==CONCURRENCY_PREVIOUS_SHA256
            and {k:v for k,v in old_sources.items() if k!=key}
            == {k:v for k,v in new_sources.items() if k!=key})


def native_targets_agree(native,optimized):
    # A fixed absolute tolerance rejects ordinary summation roundoff for
    # millions of ordinates. This is separate from posterior accuracy gates.
    return bool(np.isfinite(native) and np.isfinite(optimized)
                and np.isclose(native,-optimized,rtol=1e-12,atol=1e-7))


class PosteriorValidationError(ValueError):
    pass


def validate_production_reference(config,record):
    if config['series']<=10 or record['date'] not in config['reference_dates']:return
    required=['sampling_passed','mode_compatible','consensus_stable','posterior_accuracy_passed']
    if config.get('consensus_covariance')=='regularized':required.append('covariance_estimation_passed')
    if not all(record.get(k) is True for k in required):
        raise PosteriorValidationError(f"Full run stopped: reference validation failed on {record['date']} "
            f"for G={config['groups']}. Review dc_vs_full.json and resolve the sampling/aggregation error before a full run.")


def compatible_worker_affinity_update(previous, current):
    """Allow only this recorded scheduling fix with unchanged fit settings."""
    left, right = dict(previous), dict(current)
    old_sources, new_sources = left.pop('source_sha256'), right.pop('source_sha256')
    left.pop('source_history', None)
    right.pop('source_history', None)
    key = 'scripts/liquidity/compare_csi300_boundary_spark.py'
    return (left == right and old_sources.get(key) == AFFINITY_PREVIOUS_SHA256
            and {k: v for k, v in old_sources.items() if k != key}
            == {k: v for k, v in new_sources.items() if k != key})


def compatible_single_chain_update(previous, current):
    """Retain existing diagnostic runs only when their settings are unchanged."""
    left, right = dict(previous), dict(current)
    old_sources, new_sources = left.pop('source_sha256'), right.pop('source_sha256')
    left.pop('source_history', None)
    right.pop('source_history', None)
    key = 'scripts/liquidity/compare_csi300_boundary_spark.py'
    return (left == right and previous.get('chains', 0) >= 2
            and old_sources.get(key) == SINGLE_CHAIN_PREVIOUS_SHA256
            and {k: v for k, v in old_sources.items() if k != key}
            == {k: v for k, v in new_sources.items() if k != key})


def compatible_checkpoint_update(previous, current):
    """Allow this checkpoint repair with identical numerical code and settings."""
    left, right = dict(previous), dict(current)
    old_sources, new_sources = left.pop('source_sha256'), right.pop('source_sha256')
    left.pop('source_history', None)
    right.pop('source_history', None)
    key = 'scripts/liquidity/compare_csi300_boundary_spark.py'
    return (left == right and old_sources.get(key) == CHECKPOINT_PREVIOUS_SHA256
            and {k: v for k, v in old_sources.items() if k != key}
            == {k: v for k, v in new_sources.items() if k != key})


def sampling_quality(draws):
    """Check ESS for production; compare independent chains when available.

    The existing estimator splits each chain to estimate bulk/tail ESS. Its
    one-chain split Rhat is not reported as an independent-chain diagnostic.
    """
    from scripts.liquidity.compare_csi300_boundary_random_days import chain_quality
    draws = np.asarray(draws)
    if draws.ndim != 3 or draws.shape[0] < 1 or draws.shape[1] < 4:
        raise ValueError('At least one chain with four retained draws is required')
    result = chain_quality(draws)
    result['independent_chains_checked'] = bool(draws.shape[0] >= 2)
    if draws.shape[0] == 1:
        result['rhat_max'] = None
        result['passed'] = bool(result['ess_bulk_min'] >= 400 and result['ess_tail_min'] >= 400)
    return result


def release_worker_threads(allowed):
    """Let the scheduler place all initialized worker threads on usable CPUs."""
    if hasattr(os, 'sched_setaffinity'):
        for thread in Path('/proc/self/task').iterdir():
            try:
                os.sched_setaffinity(int(thread.name), set(allowed))
            except ProcessLookupError:
                pass


def write_json(path, value):
    temporary = path.with_name(path.name+f'.{os.getpid()}.partial')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def save_npz(path, **values):
    temporary = path.with_name(path.name+f'.{os.getpid()}.partial')
    with temporary.open('wb') as handle:
        np.savez_compressed(handle, **values)
    temporary.replace(path)


def save_npy(path, value):
    temporary = path.with_name(path.name+f'.{os.getpid()}.partial')
    with temporary.open('wb') as handle:
        np.save(handle, value)
    temporary.replace(path)


def fft_partitions_for(n, maximum):
    """Preserve n by choosing a power-of-two decomposition that divides it."""
    if n < 2 or n % 2 or maximum < 1 or maximum & (maximum-1):
        raise ValueError('Even n and a positive power-of-two maximum are required')
    p = maximum
    while n % p:
        p //= 2
    return p


def select_dates(available, references, count, seed):
    """Include full-MCMC checks, then sample remaining dates without outcomes."""
    if not 1 <= count <= len(available):
        raise ValueError('Series count exceeds the saved evaluation')
    fixed = sorted(set(references))
    if len(fixed) > count:
        fixed = fixed[:count]
    pool = sorted(set(available)-set(fixed))
    extra = np.random.default_rng(seed).choice(pool, count-len(fixed), replace=False)
    return sorted(fixed+extra.tolist())


def prepare_response(source, saved_folder, folder):
    from scripts.liquidity.compare_csi300_boundary_random_days import standardized_training_changes
    saved = json.loads((saved_folder/'fit.json').read_text())
    path = folder/'response.npy'
    if path.exists():
        # Older versions wrote directly to the destination. Discard only a
        # truncated legacy file; new writes become visible after completion.
        try:
            response = np.load(path, mmap_mode='r')
        except (OSError, ValueError, EOFError):
            path.replace(path.with_name(path.name+'.interrupted'))
        else:
            if response.shape != (saved['n'],) or not np.isfinite(response).all():
                raise ValueError('Invalid saved response')
            return saved
    if not path.exists():
        raw = np.load(source/'spread.npy', mmap_mode='r')
        response = standardized_training_changes(raw[:saved['train_stop_day']],
                    np.load(saved_folder/'clock_mean.npy'), np.load(saved_folder/'clock_scale.npy'))
        if saved['earliest_change_removed']:
            response = response[saved['earliest_change_removed']:]
        response /= saved['scale']
        if len(response) != saved['n'] or len(response) < 10_000_000:
            raise ValueError('Reconstructed training response/count differs from saved application')
        save_npy(path, response)
    response = np.load(path, mmap_mode='r')
    if response.shape != (saved['n'],) or not np.isfinite(response).all():
        raise ValueError('Invalid saved response')
    return saved


def load_input_block(block, path, n, p):
    response = np.load(path, mmap_mode='r')
    width = n//p
    return block, np.ascontiguousarray(response[block*width:(block+1)*width])


def frequency_chunks(record, n, p, groups):
    """Pass array chunks directly from block-cyclic FFT to systematic groups."""
    _, (offset, transformed) = record
    block_size = n//p
    for radix in range(p):
        start = radix*block_size+offset
        lo, hi = max(1-start, 0), min(n//2-start, transformed.shape[1])
        if hi <= lo:
            continue
        indices = np.arange(start+lo, start+hi, dtype=np.int64)
        values = np.abs(transformed[radix, lo:hi])**2/(2*np.pi*n)
        for group in range(groups):
            shift = (group-(int(indices[0])-1)) % groups
            if shift < len(indices):
                yield group, (indices[shift::groups], values[shift::groups])


def cache_frequency_group(item, folder, saved_folder, n, groups):
    group, chunks = item
    indices, values = zip(*list(chunks))
    indices, pg = np.concatenate(indices), np.concatenate(values)
    order = np.argsort(indices)
    indices, pg = indices[order], pg[order]
    path = Path(folder)/'frequencies'/f'shard{group:03d}.npz'
    omega = 2*np.pi*indices/n
    error = validate_frequency_group(group, indices, pg, omega, 4*np.sin(omega/2)**2,
                                     saved_folder, n, groups)
    save_npz(path, indices=indices, periodogram=pg, omega=omega,
             q=4*np.sin(omega/2)**2)
    write_json(path.with_suffix('.json'), error)
    return error


def validate_frequency_group(group, indices, pg, omega, q, saved_folder, n, groups):
    """Validate a new or interrupted FFT checkpoint against the original FFT."""
    np.testing.assert_array_equal(indices, np.arange(group+1, n//2, groups, dtype=np.int64))
    original = np.load(Path(saved_folder)/'periodogram.npy', mmap_mode='r')[indices-1]
    np.testing.assert_allclose(pg, original, rtol=1e-8, atol=1e-12)
    np.testing.assert_array_equal(omega, 2*np.pi*indices/n)
    np.testing.assert_array_equal(q, 4*np.sin(omega/2)**2)
    error = dict(group=int(group), ordinates=len(indices),
                 max_absolute_error=float(np.max(np.abs(pg-original))),
                 max_relative_error=float(np.max(np.abs(pg-original)/np.maximum(np.abs(original), 1e-12))),
                 checked_against_saved_full_fft=True)
    return error


def recover_frequency_checks(folder, saved_folder, n, groups):
    """Reuse complete FFT groups and recover sidecars without repeating FFTs."""
    paths = [folder/'frequencies'/f'shard{g:03d}.npz' for g in range(groups)]
    if not all(path.exists() for path in paths):
        return None
    checks = []
    for group, path in enumerate(paths):
        sidecar = path.with_suffix('.json')
        if sidecar.exists():
            checks.append(json.loads(sidecar.read_text()))
        else:
            with np.load(path) as saved:
                check = validate_frequency_group(group, saved['indices'], saved['periodogram'],
                    saved['omega'], saved['q'], saved_folder, n, groups)
            write_json(sidecar, check)
            checks.append(check)
    return checks


def saved_shard_quality(path):
    """A saved sample file contains the metadata needed to finish its commit."""
    if not path.exists():
        return None
    sidecar = path.with_suffix('.json')
    if sidecar.exists():
        return json.loads(sidecar.read_text())
    with np.load(path) as saved:
        if 'quality_json' not in saved:
            # Legacy files without a sidecar cannot recover acceptance/timing.
            return None
        quality = json.loads(str(saved['quality_json'].item()))
    write_json(sidecar, quality)
    return quality


def record_execution(out, dates, slots, elapsed, status, completed_before):
    """Preserve prior timings when a resume only rebuilds the final report."""
    path = out/'execution.json'
    history = []
    if path.exists():
        previous = json.loads(path.read_text())
        history = previous.get('attempts', [{**previous, 'legacy': True}])
    completed_after = sum((out/'days'/date/'completed.json').exists() for date in dates)
    history.append(dict(seconds=elapsed, status=status, completed_before=completed_before,
                        completed_after=completed_after, finished_at=time.time()))
    write_json(path, dict(series=len(dates), spark_slots=slots, status=status,
                          seconds=sum(x['seconds'] for x in history),
                          last_attempt_seconds=elapsed, attempts=history))


def polish_map(fit, q, pg, groups):
    from scripts.liquidity.probe_csi300_fractional_changes import refine_spectrum_map
    return refine_spectrum_map(fit, q, pg, groups)


def reuse_training_cache(reuse, folder, saved_folder, config, n):
    """Copy only validated response/FFT files, never fits or posterior draws."""
    if reuse is None:return
    reuse=Path(reuse)
    previous=json.loads((reuse/'run_config.json').read_text())
    for name in ['input_manifest_sha256','reference_protocol_sha256']:
        if previous[name]!=config[name]:raise ValueError(f'Incompatible reusable input: {name}')
    key='scripts/liquidity/compare_csi300_boundary_random_days.py'
    if previous['source_sha256'].get(key)!=config['source_sha256'].get(key):
        raise ValueError('Training response construction changed; do not reuse its cache')
    old=reuse/'days'/folder.name
    copied=[]
    response=old/'response.npy'
    if response.exists() and not (folder/'response.npy').exists():
        values=np.load(response,mmap_mode='r')
        if values.shape!=(n,) or not np.isfinite(values).all():raise ValueError('Invalid reusable response')
        temporary=folder/'response.npy.reuse.partial'
        shutil.copyfile(response,temporary);temporary.replace(folder/'response.npy')
        copied.append('response.npy')
    if previous['groups']==config['groups']:
        for group in range(config['groups']):
            src=old/'frequencies'/f'shard{group:03d}.npz'
            dst=folder/'frequencies'/src.name
            if not src.exists() or dst.exists():continue
            with np.load(src) as frequency:
                check=validate_frequency_group(group,frequency['indices'],frequency['periodogram'],
                    frequency['omega'],frequency['q'],saved_folder,n,config['groups'])
            temporary=dst.with_suffix('.reuse.partial')
            shutil.copyfile(src,temporary);temporary.replace(dst)
            write_json(dst.with_suffix('.json'),check);copied.append(str(dst.relative_to(folder)))
    write_json(folder/'reused_training.json',dict(source=str(old),copied=copied,
        posterior_samples_reused=False,all_copied_frequencies_validated=True))


def sampling_contract_hash():
    source=Path(__file__).read_text();tree=ast.parse(source)
    functions={node.name:node for node in tree.body if isinstance(node,ast.FunctionDef)}
    names=['run_shard','reference_components','prepare_full_map','sampling_quality','prepare_response']
    values=[ast.get_source_segment(source,functions[name]) for name in names]
    for node in ast.walk(functions['fit_history']):
        if isinstance(node,ast.Dict):
            for key,value in zip(node.keys,node.values):
                if isinstance(key,ast.Constant) and key.value=='day_seed':values.append(ast.dump(value))
    return hashlib.sha256('\n'.join(values).encode()).hexdigest()


def validate_sampling_reuse(previous,current):
    """Only aggregation may differ when reusing already sampled histories."""
    keys=['reference','input','input_manifest_sha256','reference_protocol_sha256',
          'model','numerical_protocol','groups','chains','seed','sampler_backend',
          'parameterization','mh_kernel','iterations','burn','max_iterations',
          'extension_iterations','ess_target','reference_chains','reference_iterations',
          'reference_burn','reference_max_iterations']
    if any(previous.get(key)!=current.get(key) for key in keys):
        raise ValueError('Sample reuse requires identical inputs, target, sampler, seed and sampling settings')
    runner='scripts/liquidity/compare_csi300_boundary_spark.py'
    old_sources,new_sources=previous['source_sha256'],current['source_sha256']
    if old_sources.get(runner)!=new_sources.get(runner):
        if (old_sources.get(runner)!=COVARIANCE_PREVIOUS_SHA256
                or sampling_contract_hash()!=SAMPLING_CONTRACT_SHA256):
            raise ValueError('Unreviewed sampling implementation change; do not reuse posterior draws')
    excluded={runner,'dts/aggregation.py','scripts/liquidity/_spectral_covariance.py'}
    if ({k:v for k,v in old_sources.items() if k not in excluded}
            !={k:v for k,v in new_sources.items() if k not in excluded}):
        raise ValueError('Sampling/likelihood sources changed; do not reuse posterior draws')


def reuse_sampled_history(parents,folder,config):
    """Copy validated same-target samples, preserving the parent experiment."""
    if not parents or (folder/'reused_sampling.json').exists():return
    for parent in parents:
        parent=Path(parent)
        previous=json.loads((parent/'run_config.json').read_text())
        if folder.name not in previous['dates']:continue
        validate_sampling_reuse(previous,config)
        old=parent/'days'/folder.name
        completed=json.loads((old/'completed.json').read_text())
        if not completed.get('sampling_passed') or not completed.get('mode_compatible'):
            raise ValueError('Only complete, sampling-validated histories can reuse posterior draws')
        if not json.loads((old/'G1_quality.json').read_text())['passed']:
            raise ValueError('A validated full-data reference is required for sample reuse')
        for group in range(config['groups']):
            path=old/'shards'/f'G{config["groups"]}_shard{group:03d}.npz'
            quality=saved_shard_quality(path)
            if not quality or not quality.get('passed'):
                raise ValueError('Every reused shard must pass its sampling diagnostics')
            with np.load(path) as saved:
                if saved['coordinate_draws'].shape!=saved['draws'].shape:
                    raise ValueError('Invalid reused coordinate draws')
        copied={}
        files=[old/name for name in ['response.npy','full_map.json','G1_draws.npy','G1_quality.json']]
        files+=list((old/'frequencies').glob('*'))+list((old/'shards').glob('*'))
        for source in files:
            if not source.is_file() or source.name.endswith('.partial'):continue
            relative=source.relative_to(old);target=folder/relative;target.parent.mkdir(parents=True,exist_ok=True)
            digest=hashlib.sha256(source.read_bytes()).hexdigest()
            if target.exists():
                if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:
                    raise ValueError(f'Conflicting sample reuse file: {relative}')
            else:
                temporary=target.with_name(target.name+'.reuse.partial')
                shutil.copyfile(source,temporary);temporary.replace(target)
            copied[str(relative)]=digest
        write_json(folder/'reused_sampling.json',dict(source=str(old),
            parent_config_sha256=hashlib.sha256((parent/'run_config.json').read_bytes()).hexdigest(),
            sampling_contract_sha256=sampling_contract_hash(),copied_sha256=copied,
            target_and_sampler_identical=True,consensus_recomputed=True))
        return


def prepare_full_map(saved_folder, folder):
    from scripts.liquidity.probe_csi300_fractional_changes import fit_spectrum
    path=folder/'full_map.json'
    if path.exists():return json.loads(path.read_text())
    old=json.loads((saved_folder/'fit.json').read_text())
    fit=fit_spectrum(np.load(saved_folder/'q.npy',mmap_mode='r'),
                     np.load(saved_folder/'periodogram.npy',mmap_mode='r'),starts=[old['theta']])
    if not fit['passed']:raise ValueError(f'Unresolved corrected full-data MAP: {fit}')
    fit['previous_theta']=old['theta']
    write_json(path,fit)
    return fit


def reference_components(fit):
    """Keep resolved competing regions in an exact MH mixture proposal."""
    selected=[]
    for candidate in fit['map_candidates']:
        if not candidate['passed']:continue
        point=np.asarray(candidate['theta'])
        if not any(np.linalg.norm((point-np.asarray(x['theta']))/[.05,2.,.05,.1])<1 for x in selected):
            selected.append(candidate)
    if not selected:raise ValueError('No resolved full-data mode')
    return selected


def run_reference_chain(chain,folder,saved_folder,config):
    release_worker_threads(config['worker_cpu_ids'])
    from numba import set_num_threads
    set_num_threads(min(16,len(config['worker_cpu_ids'])//config['reference_chains'] or 1))
    from scripts.liquidity._spectral_sampling import sampling_blocks
    folder,saved_folder=Path(folder),Path(saved_folder)
    path=folder/'reference'/f'chain{chain:03d}.npz'
    if path.exists():
        with np.load(path) as saved:return json.loads(str(saved['quality_json'].item()))
    components=reference_components(json.loads((folder/'full_map.json').read_text()))
    covariance=np.asarray(components[0]['covariance'])*(2.38**2/4)
    covariances=np.array([np.asarray(x['covariance'])*(2.38**2/4) for x in components])
    centers=np.array([x['theta'] for x in components])
    initial=centers[chain%len(centers)][None,:]
    pg=np.load(saved_folder/'periodogram.npy',mmap_mode='r')
    reference=dict(q=np.load(saved_folder/'q.npy',mmap_mode='r'),n=config['n'],centers=centers,
        chols=np.linalg.cholesky(covariances),precisions=np.linalg.inv(covariances),
        log_determinants=np.linalg.slogdet(covariances)[1])
    draws,acceptance,total,history=sampling_blocks(path.with_suffix('.checkpoint.npz'),
        initial,covariance,pg,None,1,1,config['reference_iterations'],config['reference_burn'],
        config['reference_max_iterations'],config['extension_iterations'],400,
        config['day_seed']+10_000_000+chain*100_000,sampling_quality,reference=reference)
    quality=sampling_quality(draws)
    quality.update(chain=chain,iterations=total,acceptance=acceptance,history=history,
        initial_mode=initial[0].tolist(),kernel='fixed RWMH + 10% exact mixture-independence MH',
        all_frequencies_used=True)
    save_npz(path,draws=draws,quality_json=json.dumps(quality))
    write_json(path.with_suffix('.json'),quality)
    return quality


def balanced_draws(shards):
    """Align chains of unequal lengths using evenly spaced retained draws."""
    chains=shards[0].shape[0]
    if any(x.shape[0]!=chains for x in shards):raise ValueError('Inconsistent chain counts')
    count=min(x.shape[1] for x in shards)
    return [x[:,np.linspace(0,x.shape[1]-1,count,dtype=int)] for x in shards]


def finish_reference(folder, chains):
    values=[np.load(folder/'reference'/f'chain{c:03d}.npz')['draws'] for c in range(chains)]
    draws=np.concatenate(balanced_draws(values),axis=0)
    quality=sampling_quality(draws)
    quality['competing_mode_starts_checked']=True
    save_npy(folder/'G1_draws.npy',draws)
    write_json(folder/'G1_quality.json',quality)
    return quality


def run_shard(group, folder, config):
    folder = Path(folder)
    path = folder/'shards'/f'G{config["groups"]}_shard{group:03d}.npz'
    sidecar = path.with_suffix('.json')
    cached = saved_shard_quality(path)
    if cached is not None:
        return cached
    # Bound pool creation and compilation on one CPU, then release ALL threads.
    # Re-pinning only the main thread leaves reused JAX pools on old CPUs.
    allowed = config['worker_cpu_ids']
    if hasattr(os, 'sched_setaffinity'):
        os.sched_setaffinity(0, {allowed[(config['day_seed']//10_000+group) % len(allowed)]})
    sys.path.insert(0, str(ROOT))
    from dts.mcmc import sampler, whittle_log_posterior_jit
    from scripts.liquidity.probe_csi300_fractional_changes import fit_spectrum, target
    from scripts.liquidity._spectral_sampling import sampling_blocks
    parameterization=config.get('parameterization','native')
    kernel=config.get('mh_kernel','random-walk')
    started = time.monotonic()
    try:
        with np.load(folder/'frequencies'/f'shard{group:03d}.npz') as frequency:
            q, pg, omega = frequency['q'], frequency['periodogram'], frequency['omega']
        full=json.loads((folder/'full_map.json').read_text())
        starts=[x['theta'] for x in reference_components(full)]
        fit = fit_spectrum(q, pg, groups=config['groups'],starts=starts)
        if fit['covariance'] is None or fit['newton_distance'] > .01:
            raise ValueError(f'Unresolved shard MAP: {fit}')
        native_center=np.asarray(fit['theta'])
        center=native_center.copy()
        covariance = np.asarray(fit['covariance'])*(2.38**2/4)
        quality_function=sampling_quality
        coordinate_fit=None
        if parameterization=='ma-sqrtlambda':
            from scripts.liquidity._spectral_coordinates import (
                coordinate_map,coordinate_objective,coordinate_sampling_quality,from_coordinates)
            coordinate_fit=coordinate_map(native_center,q,pg,config['groups'])
            center=np.asarray(coordinate_fit['theta'])
            covariance=np.asarray(coordinate_fit['covariance'])*(2.38**2/4)
            quality_function=coordinate_sampling_quality
        chol = np.linalg.cholesky(covariance)
        warm = float(whittle_log_posterior_jit(native_center, 0, 1, pg, True, omega,
                     config['groups'], 0, 1.0, 3, False, None, None, 0.0, 1.0))
        optimized=float(target(native_center, q, pg, config['groups'])[0])
        target_error = abs(warm+optimized)
        if not native_targets_agree(warm,optimized):
            raise ValueError(f'Native posterior target mismatch: {target_error}')
        release_worker_threads(allowed)
        initial_states=[]
        for chain in range(config['chains']):
            seed = config['day_seed']+group*1000+chain
            initial = center+chol@np.random.default_rng(seed).normal(size=4)
            objective=coordinate_objective if parameterization=='ma-sqrtlambda' else target
            if not np.isfinite(objective(initial, q, pg, config['groups'])[0]):
                initial = center.copy()
            initial_states.append(initial)
        draws,acceptances,total,history=sampling_blocks(path.with_suffix('.checkpoint.npz'),
            initial_states,covariance,pg,omega,config['groups'],config['chains'],
            config['iterations'],config['burn'],config['max_iterations'],
            config['extension_iterations'],config['ess_target'],
            config['day_seed']+group*1000,quality_function,backend=config['sampler_backend'],
            parameterization=parameterization,kernel=kernel,proposal_center=center)
        coordinate_draws=draws
        delta=native_center-np.asarray(full['theta'])
        mode_distance=float(delta@np.linalg.solve(np.asarray(full['covariance']),delta)/config['groups'])
        quality = quality_function(draws)
        quality.update(group=int(group), ordinates=len(pg), acceptance=acceptances,
                       map=fit['theta'],coordinate_map=coordinate_fit,
                       parameterization=parameterization,mh_kernel=kernel,
                       newton_distance=fit['newton_distance'],
                       newton_polish_steps=fit.get('newton_polish_steps', 0),
                       native_target_error=target_error,
                       iterations=total, burn=config['burn'],sampling_history=history,
                       retained=draws.shape[1],mode_mahalanobis_sq=mode_distance,
                       mode_matches_full=bool(mode_distance<1000),
                       full_frequency_starts=fit['full_frequency_starts'],
                       negative_log_posterior=fit['negative_log_posterior'],
                       map_candidates=[{k:x[k] for k in ['theta','negative_log_posterior','passed','search_expansions']}
                                       for x in fit['map_candidates']],
                       sampler_backend=config['sampler_backend'],
                       status='sampled', seconds=time.monotonic()-started)
        if parameterization=='ma-sqrtlambda':draws=from_coordinates(draws)
        save_npz(path, draws=draws,coordinate_draws=coordinate_draws,
                 map=np.asarray(fit['theta']), proposal_covariance=covariance,proposal_center=center,
                 quality_json=json.dumps(quality))
        write_json(sidecar, quality)
        return quality
    except Exception as exc:
        failure = dict(group=int(group), status='failed', passed=False,
                       error=f'{type(exc).__name__}: {exc}', seconds=time.monotonic()-started)
        write_json(folder/'shards'/f'G{config["groups"]}_shard{group:03d}_failure.json', failure)
        return failure


def monitored_action(action, folder, pattern, total, description,show_progress=True):
    """Run Spark while keeping actual checkpoint progress visible."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(action)
        with tqdm(total=total, desc=description, leave=False,disable=not show_progress) as progress:
            while not future.done():
                count = min(total, len(list(folder.glob(pattern))))
                progress.update(max(0, count-progress.n))
                time.sleep(1)
            result = future.result()
            progress.update(total-progress.n)
    return result


def run_covariance(item,folder,config):
    group,quality=item
    if quality['status']!='sampled':return quality
    try:
        from scripts.liquidity._spectral_covariance import ensure_covariance
        return {**quality,**ensure_covariance(group,folder,config)}
    except Exception as exc:
        failure={**quality,'status':'failed','passed':False,'covariance_passed':False,
                 'error_stage':'covariance','error':f'{type(exc).__name__}: {exc}'}
        destination=Path(folder)/'covariances';destination.mkdir(exist_ok=True)
        write_json(destination/f'shard{group:03d}_failure.json',failure)
        return failure


def finish_day(reference, out, date, records, config, fft_checks, fft_seconds, sampling_seconds,
               covariance_seconds=0.):
    from dts.aggregation import consensus
    from scripts.liquidity.compare_csi300_boundary_random_days import (
        posterior_comparison, forecasts_on_day)
    folder = out/'days'/date
    parameterization=config.get('parameterization','native')
    covariance_method=config.get('consensus_covariance','empirical')
    quality = pd.DataFrame(records).sort_values('group')
    quality.to_csv(folder/'shard_diagnostics.csv', index=False)
    sampled = quality[quality.status.eq('sampled')]
    summary = dict(date=date, groups=config['groups'], chains=config['chains'],
                   parameterization=parameterization,mh_kernel=config.get('mh_kernel','random-walk'),
                   consensus_covariance=covariance_method,covariance_seconds=covariance_seconds,
                   sampler_backend=config['sampler_backend'],
                   iterations=config['iterations'], burn=config['burn'],
                   n=config['n'], fft_partitions=config['actual_fft_partitions'],
                   fft_seconds=fft_seconds, sampling_seconds=sampling_seconds,
                   sampled_shards=len(sampled), failed_shards=int(quality.status.eq('failed').sum()),
                   passed_shards=int(quality.passed.sum()),
                   fft_max_absolute_error=max(x['max_absolute_error'] for x in fft_checks),
                   independent_chains_checked=bool(config['chains'] >= 2),
                   sampling_passed=False, posterior_accuracy_passed=None, forecast_scored=False,
                   mode_compatible=False,consensus_stable=False,
                   covariance_estimation_passed=covariance_method=='empirical',
                   extended_shards=int((sampled.iterations>config['iterations']).sum()) if len(sampled) else 0)
    if len(sampled) == config['groups']:
        draw_key='coordinate_draws' if parameterization=='ma-sqrtlambda' else 'draws'
        quality_function=sampling_quality
        if parameterization=='ma-sqrtlambda':
            from scripts.liquidity._spectral_coordinates import coordinate_sampling_quality,from_coordinates
            quality_function=coordinate_sampling_quality
        shards = [np.load(folder/'shards'/f'G{config["groups"]}_shard{g:03d}.npz')[draw_key]
                  for g in range(config['groups'])]
        aligned=balanced_draws(shards)
        aligned_quality=[quality_function(x) for x in aligned]
        quality['aligned_ess_bulk_min']=[x['ess_bulk_min'] for x in aligned_quality]
        quality['aligned_ess_tail_min']=[x['ess_tail_min'] for x in aligned_quality]
        quality['aligned_passed']=[x['passed'] for x in aligned_quality]
        quality.to_csv(folder/'shard_diagnostics.csv',index=False)
        shape = aligned[0].shape
        covariances=None
        if covariance_method=='regularized':
            covariances=[np.load(folder/'covariances'/f'shard{g:03d}.npz')['covariance']
                         for g in range(config['groups'])]
            summary['covariance_estimation_passed']=bool(quality.covariance_passed.all())
            summary['covariance_shrinkage_mean']=float(quality.shrinkage_strength.mean())
            summary['covariance_agreement_max_z']=float(quality.covariance_agreement_max_z.max())
            summary['quadrature_order_max']=int(quality.quadrature_order.max())
        combined = consensus([d.reshape(-1, 4) for d in aligned], ridge=1e-12,
                             covariances=covariances).reshape(shape)
        if parameterization=='ma-sqrtlambda':
            save_npy(folder/f'G{config["groups"]}_coordinate_draws.npy',combined)
            combined=from_coordinates(combined)
        save_npy(folder/f'G{config["groups"]}_draws.npy', combined)
        combined_quality = sampling_quality(combined)
        combined_quality['shards_passed'] = bool(quality.passed.all() and quality.aligned_passed.all())
        write_json(folder/'consensus_quality.json', combined_quality)
        summary['sampling_passed'] = bool(combined_quality['passed'] and combined_quality['shards_passed'])
        summary['mode_compatible']=bool(quality.mode_matches_full.all())
        summary['mode_distance_max']=float(quality.mode_mahalanobis_sq.max())
        summary['consensus_rhat_max'] = combined_quality['rhat_max']
        summary['consensus_ess_bulk_min'] = combined_quality['ess_bulk_min']
        summary['consensus_ess_tail_min'] = combined_quality['ess_tail_min']
        names = ['theta', 'log_lambda', 'log_sigma2', 'd']
        flat = combined.reshape(-1, 4)
        pd.DataFrame(dict(parameter=names, mean=flat.mean(axis=0), sd=flat.std(axis=0, ddof=1),
                          lower=np.quantile(flat, .025, axis=0), median=np.median(flat, axis=0),
                          upper=np.quantile(flat, .975, axis=0))).to_csv(folder/'posterior_summary.csv', index=False)
        previous = np.load(reference/'days'/date/'G2_draws.npy')
        write_json(folder/'dc_vs_previous_G2.json', posterior_comparison(previous, combined))
        from scripts.liquidity.probe_csi300_fractional_changes import target
        full_fit=json.loads((folder/'full_map.json').read_text())
        mean=flat.mean(axis=0)
        value=float(target(mean,np.load(reference/'days'/date/'q.npy',mmap_mode='r'),
            np.load(reference/'days'/date/'periodogram.npy',mmap_mode='r'),1)[0])
        summary['consensus_mean_objective_gap']=value-full_fit['negative_log_posterior']
        summary['consensus_stable']=bool(np.isfinite(value) and abs(mean[0])<1
            and -1<=summary['consensus_mean_objective_gap']<=50)
        full_path = folder/'G1_draws.npy'
        if full_path.exists():
            comparison = posterior_comparison(np.load(full_path), combined)
            write_json(folder/'dc_vs_full.json', comparison)
            reference_quality=json.loads((folder/'G1_quality.json').read_text())
            summary['reference_sampling_passed']=reference_quality['passed']
            summary['posterior_accuracy_passed'] = bool(comparison['passed'] and reference_quality['passed'])
        reference_ok=summary['posterior_accuracy_passed'] is not False
        if (summary['sampling_passed'] and summary['mode_compatible'] and summary['consensus_stable']
                and summary['covariance_estimation_passed'] and reference_ok):
            saved_folder = reference/'days'/date
            fit = {**json.loads((saved_folder/'fit.json').read_text()),**full_fit}
            for name in ['profile.npy', 'clock_mean.npy', 'clock_scale.npy']:
                target = folder/name
                if not target.exists():
                    shutil.copyfile(saved_folder/name,target)
            tasks = pd.read_csv(reference/'forecast_tasks.csv', dtype={'date': str})
            tasks = tasks[tasks.date.eq(date)]
            forecast_config = dict(posterior_parameters=64, paths_per_parameter=8)
            forecasts_on_day(Path(config['input']), folder, tasks, fit, combined,
                             forecast_config, config['day_seed']+7)
            summary['forecast_scored'] = True
    write_json(folder/'completed.json', summary)
    return summary


def summarize(out, reference, dates, execution_seconds):
    records = [json.loads((out/'days'/date/'completed.json').read_text()) for date in dates]
    diagnostics = pd.DataFrame(records)
    diagnostics.to_csv(out/'daily_diagnostics.csv', index=False)
    forecast_dates = diagnostics.loc[diagnostics.forecast_scored, 'date'].tolist()
    groups = records[0]['groups']
    chains = records[0]['chains']
    sampler_backend = records[0].get('sampler_backend', 'vanilla')
    tables = []
    for date in forecast_dates:
        old = pd.read_parquet(reference/'days'/date/'forecasts.parquet')
        new = pd.read_parquet(out/'days'/date/'forecasts.parquet')
        old = old[old.model.eq('SpectralDC_ARTFIMA')].copy()
        old['model'] = 'SpectralDC_ARTFIMA_G2'
        new.loc[new.model.eq('SpectralDC_ARTFIMA'), 'model'] = f'Spark_SpectralDC_ARTFIMA_G{groups}'
        tables.extend([old, new])
    scores = None
    if tables:
        forecasts = pd.concat(tables, ignore_index=True)
        forecasts.to_parquet(out/'forecasts.parquet', index=False)
        scored = forecasts[np.isfinite(forecasts.actual)]
        scores = scored.groupby(['model', 'horizon']).agg(n=('se', 'size'), days=('date', 'nunique'),
            mse=('se', 'mean'), mae=('ae', 'mean'), crps=('crps', 'mean'), coverage95=('cover95', 'mean')).reset_index()
        scores['rmse'] = np.sqrt(scores.pop('mse'))
        scores.to_csv(out/'scores.csv', index=False)
    report = [f'CSI 300 Spark Spectral DC: {groups} frequency shards per training history',
              f'{len(dates)} expanding daily histories of the same aggregate liquidity series.',
              ('Repository block-record Spark FFT; native Whittle target in atanh-MA/sqrt-lambda coordinates; induced prior 1/G; fixed MAP-based MH.'
               if records[0].get('parameterization')=='ma-sqrtlambda' else
               'Repository block-record Spark FFT and native dts.mcmc.sampler; fractional prior 1/G; fixed MAP-Hessian RWMH.'),
              f'Sampler backend: {sampler_backend}.',
              f"Consensus covariance estimate: {records[0].get('consensus_covariance','empirical')}.",
              'Exact original training lengths retained; P chosen as the largest power-of-two divisor <= requested P.',
              'Every positive-frequency ordinate is checked against the saved full-series FFT.',
              ('Sampling gate: all shards and consensus Rhat <=1.01; bulk/tail ESS >=400.'
               if chains >= 2 else
               'Production sampling gate: all shards and consensus bulk/tail ESS >=400. One chain per shard; independent-chain convergence is not checked; Rhat is blank.'),
              'Accuracy gate: full-reference mean shift <0.1 full SD and SD ratio within 0.9-1.1.',
              'MAP search compares competing modes on all assigned frequencies; active tail search bounds expand.',
              'Low-ESS chains continue in checkpointed blocks with the same fixed proposal; burn-in is applied once.',
              'Full-data references use fresh competing-mode starts and exact-target MH mode-jump diagnostics.',
              'Forecasts require sampling, mode compatibility, consensus stability and any available corrected full-reference accuracy checks.',
              'The archived G=2 forecast row retains earlier inference and is a legacy comparison, not a corrected full-posterior benchmark.',
              f'Execution wall seconds: {execution_seconds:.1f}', '', diagnostics.to_string(index=False)]
    if scores is not None:
        report.extend(['', 'Matched-date forecast scores:', scores.to_string(index=False)])
    (out/'REPORT.txt').write_text('\n'.join(report)+'\n')
    print(diagnostics.to_string(index=False), flush=True)
    if scores is not None:
        print(scores.to_string(index=False), flush=True)


def parse_arguments(argv=None):
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=Path)
    known, _ = pre.parse_known_args(argv)
    defaults = {}
    if known.config:
        saved = json.loads(known.config.read_text())
        defaults = {k: saved[k] for k in RUN_ARGUMENTS if k in saved}
        if 'sampler_backend' not in saved and 'source_sha256' in saved:
            defaults['sampler_backend'] = 'vanilla'
        if 'parameterization' not in saved and 'source_sha256' in saved:
            defaults.update(parameterization='native',mh_kernel='random-walk')
        if 'consensus_covariance' not in saved and 'source_sha256' in saved:
            defaults['consensus_covariance']='empirical'
    base = Path.home()/'running/dts/artifacts/liquidity'
    parser = argparse.ArgumentParser(description=__doc__, parents=[pre])
    parser.add_argument('--reference', type=Path, default=base/'csi300_boundary_full1000_v1')
    parser.add_argument('--out', type=Path, default=base/'csi300_boundary_spark_g20_test10_v5')
    parser.add_argument('--series', type=int, default=10)
    parser.add_argument('--groups', type=int, default=20)
    parser.add_argument('--chains', type=int, default=1,
                        help='Chains per shard: 1 for production; use 4 for independent-chain diagnostics')
    parser.add_argument('--iterations', type=int, default=35000)
    parser.add_argument('--burn', type=int, default=5000)
    parser.add_argument('--seed', type=int, default=20261012)
    parser.add_argument('--spark-slots', type=int, default=100)
    parser.add_argument('--history-workers',type=int,default=0,
                        help='Independent histories in flight; 0 uses max(1, Spark slots / G)')
    parser.add_argument('--fft-partitions', type=int, default=256)
    parser.add_argument('--sampler-backend', choices=['compiled', 'vanilla'], default='compiled')
    parser.add_argument('--parameterization',choices=['native','ma-sqrtlambda'],default='ma-sqrtlambda',
                        help='Sampling/consensus coordinates; physical model and prior are unchanged')
    parser.add_argument('--mh-kernel',choices=['random-walk','mixture'],
                        help='Default: mixture in boundary-aware coordinates, random-walk in native coordinates')
    parser.add_argument('--consensus-covariance',choices=['empirical','regularized'],
                        help='Default: regularized in boundary-aware coordinates, empirical in native coordinates')
    parser.add_argument('--reuse-from',type=Path,help='Reuse only validated responses/FFTs from an earlier run')
    parser.add_argument('--reuse-sampling-from',nargs='+',
                        help='Reuse validated identical-target draws and recompute consensus in a new output folder')
    parser.add_argument('--max-iterations',type=int,default=215000)
    parser.add_argument('--extension-iterations',type=int,default=30000)
    parser.add_argument('--ess-target',type=float,default=20000,
                        help='Sampling effort target; final bulk/tail ESS gate remains 400')
    parser.add_argument('--reference-dates',nargs='+',help='Fresh full-MCMC checks; default first/middle/last selected dates')
    parser.add_argument('--reference-chains',type=int,default=4)
    parser.add_argument('--reference-iterations',type=int,default=8000)
    parser.add_argument('--reference-burn',type=int,default=2000)
    parser.add_argument('--reference-max-iterations',type=int,default=24000)
    parser.add_argument('--dates', nargs='+')
    parser.set_defaults(**defaults)
    args = parser.parse_args(argv)
    args.mh_kernel=args.mh_kernel or ('mixture' if args.parameterization=='ma-sqrtlambda' else 'random-walk')
    args.consensus_covariance=args.consensus_covariance or ('regularized' if args.parameterization=='ma-sqrtlambda' else 'empirical')
    if args.parameterization=='native' and args.mh_kernel!='random-walk':
        parser.error('The mixture kernel requires --parameterization ma-sqrtlambda')
    if args.parameterization=='native' and args.consensus_covariance!='empirical':
        parser.error('Covariance regularization requires --parameterization ma-sqrtlambda')
    if args.groups < 2 or args.chains < 1 or args.iterations-args.burn < 4 or args.burn < 0:
        parser.error('Requires distributed groups, >=1 chain and >=4 retained iterations')
    if args.spark_slots < 1 or args.series < 1 or args.history_workers<0:
        parser.error('Positive Spark slots and series count required')
    if (args.max_iterations<args.iterations or args.extension_iterations<4 or args.ess_target<400
            or args.reference_chains<2 or args.reference_burn<0
            or args.reference_iterations-args.reference_burn<4
            or args.reference_max_iterations<args.reference_iterations):
        parser.error('Invalid extension/reference settings; effort target must be >=400')
    fft_partitions_for(2, args.fft_partitions)
    return args


def fit_history(date,args,source,config,spark,show_progress=True):
    folder = args.out/'days'/date
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'completed.json').exists():
        return json.loads((folder/'completed.json').read_text())
    (folder/'frequencies').mkdir(exist_ok=True)
    (folder/'shards').mkdir(exist_ok=True)
    saved_folder = args.reference/'days'/date
    write_json(folder/'progress.json', dict(stage='prepare', time=time.time()))
    n=json.loads((saved_folder/'fit.json').read_text())['n']
    reuse_sampled_history(args.reuse_sampling_from,folder,config)
    reuse_training_cache(args.reuse_from,folder,saved_folder,config,n)
    saved = prepare_response(source, saved_folder, folder)
    n = saved['n']
    p = fft_partitions_for(n, args.fft_partitions)
    daily = {**config, 'n': n, 'actual_fft_partitions': p,
             'day_seed': args.seed+saved['train_stop_day']*10_000}
    print(f'{date}: n={n:,}, FFT P={p}, G={args.groups}, {n//2//args.groups:,} frequencies/shard', flush=True)
    write_json(folder/'progress.json',dict(stage='corrected_full_map',time=time.time()))
    prepare_full_map(saved_folder,folder)
    if date in args.reference_dates and not (folder/'G1_quality.json').exists():
        (folder/'reference').mkdir(exist_ok=True)
        write_json(folder/'progress.json',dict(stage='full_reference',time=time.time()))
        folder_text,saved_text=str(folder),str(saved_folder)
        result=spark.sparkContext.parallelize(range(args.reference_chains),args.reference_chains).map(
            lambda chain:run_reference_chain(chain,folder_text,saved_text,daily))
        monitored_action(result.collect,folder/'reference','chain*.json',args.reference_chains,
                         f'{date} full-reference chains',show_progress)
        reference_quality=finish_reference(folder,args.reference_chains)
        print(f'{date} corrected full-reference: {reference_quality}',flush=True)
    write_json(folder/'progress.json', dict(stage='distributed_fft', time=time.time()))
    t = time.monotonic()
    fft_checks = recover_frequency_checks(folder, saved_folder, n, args.groups)
    if fft_checks is None:
        response_path = str(folder/'response.npy')
        blocks = spark.sparkContext.parallelize(range(p), p).map(
            lambda b: load_input_block(b, response_path, n, p))
        transformed = spark_fft_contiguous_blocks(blocks, n, p)
        groups = transformed.flatMap(lambda record: frequency_chunks(record, n, p, args.groups)).groupByKey(args.groups)
        folder_text, saved_text = str(folder), str(saved_folder)
        result = groups.map(lambda item: cache_frequency_group(item, folder_text, saved_text, n, args.groups))
        fft_checks = monitored_action(result.collect, folder/'frequencies', 'shard*.json',
                                      args.groups, f'{date} FFT groups',show_progress)
    fft_seconds = time.monotonic()-t
    write_json(folder/'fft_validation.json', dict(n=n, fft_partitions=p, groups=args.groups,
               ordinates=sum(x['ordinates'] for x in fft_checks), all_frequencies_checked=True,
               max_absolute_error=max(x['max_absolute_error'] for x in fft_checks), seconds=fft_seconds))
    write_json(folder/'progress.json', dict(stage='spark_mcmc', time=time.time()))
    t = time.monotonic()
    folder_text = str(folder)
    result = spark.sparkContext.parallelize(range(args.groups), args.groups).map(
        lambda group: run_shard(group, folder_text, daily))
    records = monitored_action(result.collect, folder/'shards', '*.json', args.groups,
                               f'{date} MCMC shards',show_progress)
    sampling_seconds = time.monotonic()-t
    covariance_seconds=0.
    if config.get('consensus_covariance')=='regularized':
        write_json(folder/'progress.json',dict(stage='covariance',time=time.time()))
        t=time.monotonic();(folder/'covariances').mkdir(exist_ok=True)
        result=spark.sparkContext.parallelize([(int(r['group']),r) for r in records],args.groups).map(
            lambda item:run_covariance(item,folder_text,daily))
        records=monitored_action(result.collect,folder/'covariances','shard*.json',args.groups,
                                 f'{date} consensus covariances',show_progress)
        covariance_seconds=time.monotonic()-t
    write_json(folder/'progress.json', dict(stage='consensus', time=time.time()))
    finished = finish_day(args.reference, args.out, date, records, daily, fft_checks, fft_seconds,
                          sampling_seconds,covariance_seconds)
    write_json(folder/'progress.json', dict(stage='completed', time=time.time()))
    print(json.dumps(finished), flush=True)
    return finished


def execute_histories(dates,references,workers,fit,validate,cancel):
    """Validate reference fits concurrently, then release the remaining fits."""
    first=sorted(set(references));remaining=[d for d in dates if d not in first]
    with tqdm(total=len(dates),desc='Spark liquidity histories') as progress:
        executor=ThreadPoolExecutor(max_workers=workers)
        try:
            for batch in [first,remaining]:
                futures={executor.submit(fit,date,workers==1):date for date in batch}
                for future in as_completed(futures):
                    validate(future.result());progress.update(1)
        except BaseException:
            cancel();executor.shutdown(wait=False,cancel_futures=True)
            raise
        else:executor.shutdown(wait=True)


def main():
    args = parse_arguments()
    args.reference, args.out = args.reference.expanduser().absolute(), args.out.expanduser().absolute()
    if args.reuse_from:args.reuse_from=Path(args.reuse_from).expanduser().absolute()
    if args.reuse_sampling_from:
        args.reuse_sampling_from=[str(Path(value).expanduser().absolute()) for value in args.reuse_sampling_from]
        if str(args.out) in args.reuse_sampling_from:
            raise ValueError('Reuse sampled histories in a new output folder')
    original = json.loads((args.reference/'protocol.json').read_text())
    input_config = json.loads((args.reference/'run_config.json').read_text())
    source = Path(input_config['input'])
    dates = args.dates or select_dates(original['dates'], original['references'], args.series, args.seed)
    if len(dates) != args.series or len(set(dates)) != len(dates) or not set(dates) <= set(original['dates']):
        raise ValueError('Dates must identify the requested number of distinct saved training histories')
    dates = sorted(dates)
    args.reference_dates=args.reference_dates or sorted({dates[0],dates[len(dates)//2],dates[-1]})
    if not set(args.reference_dates)<=set(dates):raise ValueError('Reference dates must be in the requested sample')
    cpu_ids = sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else list(range(os.cpu_count()))
    config = {k: str(getattr(args, k)) if isinstance(getattr(args, k), Path) else getattr(args, k)
              for k in RUN_ARGUMENTS}
    config.update(dates=dates, input=str(source), worker_cpu_ids=cpu_ids,
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__), ROOT/'dts/dfft_blocks.py', ROOT/'dts/mcmc.py', ROOT/'dts/aggregation.py',
             ROOT/'scripts/liquidity/probe_csi300_fractional_changes.py',
             ROOT/'scripts/liquidity/_spectral_sampling.py',
             ROOT/'scripts/liquidity/run_csi300_fractional_changes_dc.py',
             ROOT/'scripts/liquidity/compare_csi300_boundary_random_days.py']+
            ([ROOT/'scripts/liquidity/_spectral_coordinates.py'] if args.parameterization=='ma-sqrtlambda' else [])+
            ([ROOT/'scripts/liquidity/_spectral_covariance.py'] if args.consensus_covariance=='regularized' else [])},
        input_manifest_sha256=hashlib.sha256((source/'input_manifest.json').read_bytes()).hexdigest(),
        reference_protocol_sha256=hashlib.sha256((args.reference/'protocol.json').read_bytes()).hexdigest(),
        model='ARTFIMA(0,d,lambda,1) on identical training-clock-standardized changes',
        inference=('Spark block FFT; systematic frequency groups; native DTS sampler; prior 1/G; fixed MAP-Hessian RWMH; joint consensus'
                   if args.parameterization=='native' else
                   'Spark block FFT; systematic frequency groups; native Whittle target in atanh-MA/sqrt-lambda coordinates; induced prior 1/G; fixed MAP-based MH; joint covariance consensus'),
        fft_policy='Largest power-of-two divisor <= fft_partitions; no observations discarded',
        numerical_protocol=('multimode-map-and-checkpointed-ess-v2' if args.parameterization=='native'
                            else 'boundary-coordinates-and-covariance-ess-v3'))
    if args.consensus_covariance=='regularized':
        config['covariance_protocol']='dimensionless-batch-shrinkage-to-converged-gh-v1'
        config['inference']+='; covariance regularization with checked quadrature anchor'
    args.out.mkdir(parents=True, exist_ok=True)
    config_path = args.out/'run_config.json'
    if config_path.exists():
        previous = json.loads(config_path.read_text())
        config['source_history'] = previous.get('source_history', [])
        if previous != config:
            if compatible_worker_affinity_update(previous, config):
                reason = 'Release all initialized JAX threads from stale CPU affinity; same target, sampler and seeds'
                archive_name = 'worker_affinity'
            elif compatible_single_chain_update(previous, config):
                reason = 'One-chain production default and dynamic reporting; existing independent-chain settings, target, sampler and seeds unchanged'
                archive_name = 'single_chain_default'
            elif compatible_checkpoint_update(previous, config):
                reason = 'Atomic response/consensus writes and recovery of checkpoint metadata; same numerical target, sampler, settings and seeds'
                archive_name = 'atomic_checkpoints'
            elif compatible_validation_update(previous,config):
                reason = 'Scale-aware target checksum and full-run validation guard; same fits, likelihood, sampler and seeds'
                archive_name = 'validation_guard'
            elif compatible_concurrency_update(previous,config):
                reason = 'Concurrent independent histories and distributed worker bootstrapping; same targets, proposals and seeds'
                archive_name = 'history_concurrency'
            elif compatible_coordinate_update(previous,config):
                reason = 'Optional boundary coordinates and MH kernel; native target, kernel, checkpoints and seeds unchanged'
                archive_name = 'optional_coordinates'
            elif compatible_covariance_update(previous,config):
                reason = 'Optional covariance regularization; existing empirical weights and posterior draws unchanged'
                archive_name = 'optional_covariance'
            else:
                raise ValueError('Use a new output directory after incompatible configuration/source changes')
            archive = args.out/'attempts'/archive_name
            archive.mkdir(parents=True, exist_ok=True)
            write_json(archive/'run_config.json', previous)
            config['source_history'] = config['source_history']+[dict(
                reason=reason,
                previous=previous['source_sha256'], current=config['source_sha256'])]
    else:
        config['source_history'] = []
    write_json(config_path, config)
    tasks = pd.read_csv(args.reference/'forecast_tasks.csv', dtype={'date': str})
    tasks[tasks.date.isin(dates)].to_csv(args.out/'forecast_tasks.csv', index=False)
    print(f'{len(dates)} histories; G={args.groups}; {args.chains} chains; {args.iterations}/{args.burn} iterations/burn', flush=True)
    from pyspark.sql import SparkSession
    slots = min(args.spark_slots, len(cpu_ids))
    history_workers=args.history_workers or max(1,slots//args.groups)
    print(f'{slots} Spark slots; up to {history_workers} independent histories after reference validation',flush=True)
    completed_before = sum((args.out/'days'/date/'completed.json').exists() for date in dates)
    started = time.monotonic()
    spark = (SparkSession.builder.master(f'local[{slots}]').appName('csi300-spectral-dc-pilot')
             .config('spark.driver.host', '127.0.0.1').config('spark.driver.bindAddress', '127.0.0.1')
             .config('spark.ui.enabled', 'false').config('spark.ui.showConsoleProgress', 'false')
             .config('spark.scheduler.mode','FAIR')
             .config('spark.driver.maxResultSize', '4g').config('spark.python.worker.reuse', 'true')
             .config('spark.pyspark.python', sys.executable)
             .config('spark.executorEnv.PYTHONPATH', str(ROOT))
             .config('spark.executorEnv.NUMBA_CACHE_DIR', os.environ['NUMBA_CACHE_DIR'])
             .config('spark.executorEnv.NUMBA_NUM_THREADS',os.environ['NUMBA_NUM_THREADS'])
             .config('spark.executorEnv.MPLCONFIGDIR', os.environ['MPLCONFIGDIR'])
             .config('spark.executorEnv.OPENBLAS_NUM_THREADS', '1')
             .config('spark.executorEnv.OMP_NUM_THREADS', '1').config('spark.executorEnv.MKL_NUM_THREADS', '1')
             .getOrCreate())
    spark.sparkContext.setLogLevel('ERROR')
    status = 'interrupted'
    try:
        execute_histories(dates,args.reference_dates,history_workers,
            lambda date,show:fit_history(date,args,source,config,spark,show),
            lambda record:validate_production_reference(config,record),
            spark.sparkContext.cancelAllJobs)
        status = 'completed'
    except PosteriorValidationError:
        status='validation_failed'
        ready=[d for d in dates if (args.out/'days'/d/'completed.json').exists()]
        if ready:summarize(args.out,args.reference,ready,time.monotonic()-started)
        raise
    finally:
        try:
            spark.stop()
        finally:
            record_execution(args.out, dates, slots, time.monotonic()-started, status, completed_before)
    elapsed = time.monotonic()-started
    summarize(args.out, args.reference, dates, elapsed)
    print(f'Finished: {args.out}', flush=True)


if __name__ == '__main__':
    main()
