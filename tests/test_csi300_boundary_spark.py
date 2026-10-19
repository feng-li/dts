import numpy as np

from scripts.liquidity.compare_csi300_boundary_spark import (
    AFFINITY_PREVIOUS_SHA256, SINGLE_CHAIN_PREVIOUS_SHA256, CHECKPOINT_PREVIOUS_SHA256,
    compatible_worker_affinity_update, compatible_single_chain_update,
    compatible_checkpoint_update, recover_frequency_checks, save_npy, save_npz,
    saved_shard_quality, prepare_response, run_shard, record_execution,
    fft_partitions_for, frequency_chunks, parse_arguments, sampling_quality,
    select_dates, summarize)


def test_fft_partition_choice_preserves_saved_lengths():
    assert fft_partitions_for(10_238_352, 256) == 16
    assert fft_partitions_for(17_317_660, 256) == 4
    assert fft_partitions_for(24_300_012, 256) == 4


def test_block_spectrum_groups_preserve_all_positive_frequencies():
    n, p, groups = 360, 8, 7
    spectrum = np.fft.fft(np.random.default_rng(7).normal(size=n))
    width = n//p
    chunks = {group: [] for group in range(groups)}
    for block in range(p):
        lo, hi = block*width//p, (block+1)*width//p
        matrix = np.array([spectrum[r*width+lo:r*width+hi] for r in range(p)])
        for group, data in frequency_chunks((block, (lo, matrix)), n, p, groups):
            chunks[group].append(data)
    for group, pieces in chunks.items():
        indices = np.concatenate([x[0] for x in pieces])
        values = np.concatenate([x[1] for x in pieces])
        order = np.argsort(indices)
        expected = np.arange(group+1, n//2, groups)
        np.testing.assert_array_equal(indices[order], expected)
        np.testing.assert_allclose(values[order], np.abs(spectrum[expected])**2/(2*np.pi*n))


def test_date_selection_keeps_reference_dates_and_is_reproducible():
    dates = [str(x) for x in range(100)]
    selected = select_dates(dates, ['0', '50', '99'], 10, 20261012)
    assert len(selected) == 10 and {'0', '50', '99'} <= set(selected)
    assert selected == select_dates(dates, ['0', '50', '99'], 10, 20261012)


def test_resume_accepts_only_recorded_scheduling_update():
    key = 'scripts/liquidity/compare_csi300_boundary_spark.py'
    old = dict(groups=200, iterations=15000,
               source_sha256={key: AFFINITY_PREVIOUS_SHA256, 'dts/mcmc.py': 'same'})
    current = {**old, 'source_sha256': {key: 'current', 'dts/mcmc.py': 'same'},
               'source_history': []}
    assert compatible_worker_affinity_update(old, current)
    assert not compatible_worker_affinity_update(old, {**current, 'iterations': 4000})
    assert not compatible_worker_affinity_update(old, {**current, 'source_sha256':
               {key: 'current', 'dts/mcmc.py': 'changed'}})


def test_production_defaults_to_one_chain_and_saved_diagnostics_remain_explicit(tmp_path):
    import json
    import pytest
    assert parse_arguments([]).chains == 1
    assert parse_arguments([]).groups == 20
    assert parse_arguments([]).parameterization == 'ma-sqrtlambda'
    assert parse_arguments([]).mh_kernel == 'mixture'
    assert parse_arguments([]).consensus_covariance == 'regularized'
    assert parse_arguments([]).sampler_backend == 'compiled'
    assert parse_arguments(['--sampler-backend', 'vanilla']).sampler_backend == 'vanilla'
    assert parse_arguments(['--chains', '4']).chains == 4
    path = tmp_path/'run_config.json'
    path.write_text(json.dumps(dict(chains=4)))
    assert parse_arguments(['--config', str(path)]).chains == 4
    assert parse_arguments(['--config', str(path), '--chains', '1']).chains == 1
    path.write_text(json.dumps(dict(chains=4, source_sha256={})))
    assert parse_arguments(['--config', str(path)]).sampler_backend == 'vanilla'
    assert parse_arguments(['--config', str(path)]).parameterization == 'native'
    assert parse_arguments(['--config', str(path)]).mh_kernel == 'random-walk'
    assert parse_arguments(['--config', str(path)]).consensus_covariance == 'empirical'
    with pytest.raises(SystemExit):
        parse_arguments(['--chains', '0'])


def test_coordinate_upgrade_only_reuses_unchanged_native_targets():
    from scripts.liquidity.compare_csi300_boundary_spark import (
        COORDINATE_PREVIOUS_SHA256, SAMPLING_PREVIOUS_SHA256, compatible_coordinate_update)
    runner='scripts/liquidity/compare_csi300_boundary_spark.py'
    sampling='scripts/liquidity/_spectral_sampling.py'
    previous=dict(groups=2,chains=1,source_sha256={runner:COORDINATE_PREVIOUS_SHA256,
        sampling:SAMPLING_PREVIOUS_SHA256,'dts/mcmc.py':'unchanged'})
    current={**previous,'parameterization':'native','mh_kernel':'random-walk',
        'source_sha256':{runner:'new',sampling:'new','dts/mcmc.py':'unchanged'}}
    assert compatible_coordinate_update(previous,current)
    assert not compatible_coordinate_update(previous,{**current,'parameterization':'ma-sqrtlambda'})
    assert not compatible_coordinate_update(previous,{**current,'mh_kernel':'mixture'})
    assert not compatible_coordinate_update(previous,{**current,'source_sha256':{
        **current['source_sha256'],'dts/mcmc.py':'changed'}})


def test_one_chain_reports_ess_without_claiming_independent_chain_convergence():
    draws = np.random.default_rng(10).normal(size=(1, 8192, 4))
    quality = sampling_quality(draws)
    assert quality['passed'] and quality['ess_bulk_min'] >= 400
    assert quality['ess_tail_min'] >= 400
    assert quality['rhat_max'] is None and not quality['independent_chains_checked']
    draws[0, :, 0] = np.linspace(-5, 5, draws.shape[1])
    assert not sampling_quality(draws)['passed']


def test_multiple_chain_diagnostics_keep_the_previous_gate():
    from scripts.liquidity.compare_csi300_boundary_random_days import chain_quality
    draws = np.random.default_rng(11).normal(size=(4, 2048, 4))
    previous = chain_quality(draws)
    current = sampling_quality(draws)
    assert current.pop('independent_chains_checked')
    assert current == previous


def test_resume_rejects_relabelling_four_chain_draws_as_one_chain():
    key = 'scripts/liquidity/compare_csi300_boundary_spark.py'
    previous = dict(chains=4, groups=200, source_sha256={
        key: SINGLE_CHAIN_PREVIOUS_SHA256, 'dts/mcmc.py': 'same'})
    current = {**previous, 'source_sha256': {key: 'current', 'dts/mcmc.py': 'same'}}
    assert compatible_single_chain_update(previous, current)
    assert not compatible_single_chain_update(previous, {**current, 'chains': 1})
    assert not compatible_single_chain_update(previous, {**current, 'source_sha256':
        {key: 'current', 'dts/mcmc.py': 'changed'}})


def test_single_chain_report_uses_actual_history_and_shard_counts(tmp_path):
    import json
    folder = tmp_path/'days/20211013'
    folder.mkdir(parents=True)
    record = dict(date='20211013', groups=10, chains=1, sampling_passed=False,
                  forecast_scored=False, consensus_rhat_max=None)
    (folder/'completed.json').write_text(json.dumps(record))
    summarize(tmp_path, tmp_path/'unused', ['20211013'], 5.)
    report = (tmp_path/'REPORT.txt').read_text()
    assert '10 frequency shards' in report and '1 expanding daily histories' in report
    assert 'independent-chain convergence is not checked' in report


def test_atomic_array_write_preserves_previous_file_when_interrupted(tmp_path, monkeypatch):
    import pytest
    path = tmp_path/'response.npy'
    save_npy(path, np.arange(8.))
    def interrupted(handle, value):
        handle.write(b'incomplete')
        raise KeyboardInterrupt
    monkeypatch.setattr(np, 'save', interrupted)
    with pytest.raises(KeyboardInterrupt):
        save_npy(path, np.ones(8))
    np.testing.assert_array_equal(np.load(path), np.arange(8.))


def test_truncated_legacy_response_is_rebuilt_without_changing_raw_data(tmp_path, monkeypatch):
    import json
    import scripts.liquidity.compare_csi300_boundary_random_days as old
    source, saved, folder = [tmp_path/x for x in ['source', 'saved', 'output']]
    for path in (source, saved, folder):
        path.mkdir()
    raw = np.arange(12.).reshape(2, 6)
    np.save(source/'spread.npy', raw)
    np.save(saved/'clock_mean.npy', np.zeros(6))
    np.save(saved/'clock_scale.npy', np.ones(6))
    fit = dict(n=10_000_000, train_stop_day=2, earliest_change_removed=0, scale=2.)
    (saved/'fit.json').write_text(json.dumps(fit))
    (folder/'response.npy').write_bytes(b'incomplete')
    monkeypatch.setattr(old, 'standardized_training_changes',
                        lambda *args: np.full(10_000_000, 4.))
    assert prepare_response(source, saved, folder) == fit
    assert (folder/'response.npy.interrupted').read_bytes() == b'incomplete'
    response = np.load(folder/'response.npy', mmap_mode='r')
    assert response.shape == (fit['n'],) and np.all(response == 2.)
    np.testing.assert_array_equal(np.load(source/'spread.npy'), raw)
    monkeypatch.setattr(old, 'standardized_training_changes',
                        lambda *args: (_ for _ in ()).throw(AssertionError('recomputed')))
    assert prepare_response(source, saved, folder) == fit


def test_fft_cache_recovers_interrupted_metadata_and_keeps_every_ordinate(tmp_path):
    import json
    folder, reference = tmp_path/'output', tmp_path/'reference'
    (folder/'frequencies').mkdir(parents=True)
    reference.mkdir()
    n, groups = 64, 3
    pg = np.abs(np.fft.fft(np.random.default_rng(12).normal(size=n))[1:n//2])**2/(2*np.pi*n)
    np.save(reference/'periodogram.npy', pg)
    for group in range(groups):
        indices = np.arange(group+1, n//2, groups)
        omega = 2*np.pi*indices/n
        save_npz(folder/'frequencies'/f'shard{group:03d}.npz', indices=indices,
                 periodogram=pg[indices-1], omega=omega, q=4*np.sin(omega/2)**2)
    before = [(p, p.read_bytes()) for p in (folder/'frequencies').glob('*.npz')]
    checks = recover_frequency_checks(folder, reference, n, groups)
    assert sum(x['ordinates'] for x in checks) == n//2-1
    assert all(x['checked_against_saved_full_fft'] for x in checks)
    for path, original in before:
        assert path.read_bytes() == original
        assert json.loads(path.with_suffix('.json').read_text()) == checks[int(path.stem[-3:])]
    assert recover_frequency_checks(folder, reference, n, groups) == checks
    (folder/'frequencies/shard001.npz').unlink()
    assert recover_frequency_checks(folder, reference, n, groups) is None


def test_interrupted_fft_checkpoint_is_not_certified_when_it_differs(tmp_path):
    import pytest
    folder, reference = tmp_path/'output', tmp_path/'reference'
    (folder/'frequencies').mkdir(parents=True)
    reference.mkdir()
    np.save(reference/'periodogram.npy', np.ones(3))
    for group in range(2):
        indices = np.arange(group+1, 4, 2)
        omega = 2*np.pi*indices/8
        save_npz(folder/'frequencies'/f'shard{group:03d}.npz', indices=indices,
                 periodogram=np.zeros(len(indices)), omega=omega, q=4*np.sin(omega/2)**2)
    with pytest.raises(AssertionError):
        recover_frequency_checks(folder, reference, 8, 2)
    assert not list((folder/'frequencies').glob('*.json'))


def test_saved_sampling_recovers_sidecar_and_skips_sampler(tmp_path):
    import json
    folder = tmp_path/'day'
    (folder/'shards').mkdir(parents=True)
    path = folder/'shards/G200_shard007.npz'
    quality = dict(group=7, status='sampled', passed=False, seconds=12., acceptance=[.3])
    save_npz(path, draws=np.zeros((1, 10, 4)), quality_json=json.dumps(quality))
    assert saved_shard_quality(path) == quality
    path.with_suffix('.json').unlink()
    # No worker settings/frequencies are supplied: reuse returns before fitting.
    assert run_shard(7, str(folder), dict(groups=200)) == quality
    assert json.loads(path.with_suffix('.json').read_text()) == quality
    path.unlink()
    assert saved_shard_quality(path) is None  # A sidecar alone is not a checkpoint.


def test_checkpoint_update_keeps_only_identical_numerical_runs():
    key = 'scripts/liquidity/compare_csi300_boundary_spark.py'
    previous = dict(chains=1, groups=200, sampler_backend='compiled', source_sha256={
        key: CHECKPOINT_PREVIOUS_SHA256, 'dts/mcmc.py': 'same'})
    current = {**previous, 'source_sha256': {key: 'current', 'dts/mcmc.py': 'same'}}
    assert compatible_checkpoint_update(previous, current)
    assert not compatible_checkpoint_update(previous, {**current, 'chains': 4})
    assert not compatible_checkpoint_update(previous, {**current, 'sampler_backend': 'vanilla'})
    assert not compatible_checkpoint_update(previous, {**current, 'source_sha256':
        {key: 'current', 'dts/mcmc.py': 'changed'}})


def test_resume_preserves_previous_execution_timing(tmp_path):
    import json
    (tmp_path/'execution.json').write_text(json.dumps(dict(series=2, spark_slots=100, seconds=50.)))
    (tmp_path/'days/20211013').mkdir(parents=True)
    (tmp_path/'days/20211013/completed.json').write_text('{}')
    record_execution(tmp_path, ['20211013', '20231124'], 100, 12., 'interrupted', 0)
    record_execution(tmp_path, ['20211013', '20231124'], 100, 3., 'completed', 1)
    result = json.loads((tmp_path/'execution.json').read_text())
    assert result['seconds'] == 65. and result['last_attempt_seconds'] == 3.
    assert len(result['attempts']) == 3 and result['attempts'][0]['legacy']
    assert result['attempts'][-1]['completed_before'] == result['attempts'][-1]['completed_after'] == 1


def test_sampling_extension_resume_preserves_rng_and_applies_burn_once(tmp_path,monkeypatch):
    import pytest
    import scripts.liquidity._spectral_sampling as recovery
    pg=np.ones(31)*.15;omega=2*np.pi*np.arange(1,32)/64
    def quality(draws):
        enough=draws.shape[1]>=96
        return dict(passed=enough,ess_bulk_min=800 if enough else 100,
                    ess_tail_min=800 if enough else 100)
    args=dict(initial=np.array([[.1,-4.,-.1,.2]]),covariance=np.eye(4)*.001,
        pg=pg,omega=omega,groups=200,chains=1,iterations=64,burn=16,
        maximum=128,extension=32,ess_target=800,seed=5,quality_function=quality)
    uninterrupted=recovery.sampling_blocks(tmp_path/'full.npz',**args)
    original=recovery.atomic_npz
    calls=[]
    def interrupt_second(path,**values):
        calls.append(path)
        if len(calls)==2:raise KeyboardInterrupt
        original(path,**values)
    monkeypatch.setattr(recovery,'atomic_npz',interrupt_second)
    with pytest.raises(KeyboardInterrupt):recovery.sampling_blocks(tmp_path/'resume.npz',**args)
    monkeypatch.setattr(recovery,'atomic_npz',original)
    resumed=recovery.sampling_blocks(tmp_path/'resume.npz',**args)
    np.testing.assert_array_equal(uninterrupted[0],resumed[0])
    assert uninterrupted[1:]==resumed[1:]
    assert resumed[0].shape==(1,112,4) and resumed[2]==128


def test_balancing_uses_extended_history_instead_of_dropping_it():
    from scripts.liquidity.compare_csi300_boundary_spark import balanced_draws
    short=np.arange(40.).reshape(1,10,4)
    long=np.arange(120.).reshape(1,30,4)
    aligned=balanced_draws([short,long])
    np.testing.assert_array_equal(aligned[0],short)
    np.testing.assert_array_equal(aligned[1][:,-1],long[:,-1])
    assert aligned[1].shape==short.shape


def test_mode_jump_density_uses_complete_mixture():
    from scipy.special import logsumexp
    from scipy.stats import multivariate_normal
    from scripts.liquidity._spectral_sampling import mixture_log_density
    centers=np.array([[0.,-10.,0.,-.08],[-.99,-4.,0.,.94]])
    covariances=np.array([np.diag([.1,2.,.2,.1]),np.diag([.01,.1,.2,.05])])
    point=np.array([-.2,-6.,-.1,.2])
    expected=logsumexp([multivariate_normal.logpdf(point,c,cov) for c,cov in zip(centers,covariances)])-np.log(2)
    actual=mixture_log_density(point,centers,np.linalg.inv(covariances),np.linalg.slogdet(covariances)[1])
    np.testing.assert_allclose(actual,expected,rtol=1e-12)


def test_parallel_full_reference_visits_complete_grid_and_matches_native_target():
    from scripts.liquidity._spectral_sampling import full_reference_target
    from scripts.liquidity.probe_csi300_fractional_changes import target
    n=512;omega=2*np.pi*np.arange(1,n//2)/n;q=4*np.sin(omega/2)**2
    pg=np.random.default_rng(9).exponential(.2,size=len(q))
    for point in [np.array([.2,-10.,-.1,-.08]),np.array([-.998,-4.6,-.01,.94])]:
        np.testing.assert_allclose(full_reference_target(point,q,pg,n),target(point,q,pg,1)[0],rtol=1e-10)


def test_reuse_copies_only_validated_training_files(tmp_path):
    import json
    from scripts.liquidity.compare_csi300_boundary_spark import reuse_training_cache
    old,folder,saved=tmp_path/'old',tmp_path/'new/20211013',tmp_path/'saved'
    (old/'days/20211013/frequencies').mkdir(parents=True)
    (folder/'frequencies').mkdir(parents=True);saved.mkdir()
    config=dict(groups=2,input_manifest_sha256='same',reference_protocol_sha256='same',
        source_sha256={'scripts/liquidity/compare_csi300_boundary_random_days.py':'same'})
    (old/'run_config.json').write_text(json.dumps(config))
    n=64;pg=np.ones(n//2-1)*.2
    np.save(saved/'periodogram.npy',pg)
    np.save(old/'days/20211013/response.npy',np.arange(n,dtype=float))
    np.save(old/'days/20211013/G200_draws.npy',np.ones((1,10,4)))
    for group in range(2):
        indices=np.arange(group+1,n//2,2);omega=2*np.pi*indices/n
        save_npz(old/'days/20211013/frequencies'/f'shard{group:03d}.npz',indices=indices,
            omega=omega,periodogram=pg[indices-1],q=4*np.sin(omega/2)**2)
    reuse_training_cache(old,folder,saved,config,n)
    assert not (folder/'G200_draws.npy').exists()
    assert not (folder/'response.npy').is_symlink()
    assert recover_frequency_checks(folder,saved,n,2) is not None


def test_target_checksum_accepts_large_sum_roundoff_but_rejects_target_changes():
    from scripts.liquidity.compare_csi300_boundary_spark import native_targets_agree
    assert native_targets_agree(-2_170_000.,2_170_000.+2.6e-7)
    assert not native_targets_agree(-2_170_000.,2_170_000.+.01)
    assert not native_targets_agree(float('-inf'),float('inf'))


def test_full_run_stops_on_failed_approximation_without_loosening_gate():
    import pytest
    from scripts.liquidity.compare_csi300_boundary_spark import validate_production_reference,PosteriorValidationError
    config=dict(series=1000,groups=200,reference_dates=['20211013'])
    record=dict(date='20211013',sampling_passed=True,mode_compatible=True,
                consensus_stable=True,posterior_accuracy_passed=False)
    with pytest.raises(PosteriorValidationError,match='reference validation failed'):
        validate_production_reference(config,record)
    validate_production_reference({**config,'series':3},record)
    validate_production_reference(config,{**record,'posterior_accuracy_passed':True})


def test_validation_guard_update_keeps_only_unchanged_numerical_sources():
    from scripts.liquidity.compare_csi300_boundary_spark import compatible_validation_update,VALIDATION_PREVIOUS_SHA256
    key='scripts/liquidity/compare_csi300_boundary_spark.py'
    previous=dict(series=3,groups=200,source_sha256={key:VALIDATION_PREVIOUS_SHA256,'dts/mcmc.py':'same'})
    current={**previous,'source_sha256':{key:'current','dts/mcmc.py':'same'}}
    assert compatible_validation_update(previous,current)
    assert not compatible_validation_update(previous,{**current,'groups':2})
    assert not compatible_validation_update(previous,{**current,'source_sha256':{key:'current','dts/mcmc.py':'changed'}})


def test_independent_histories_run_concurrently_after_reference_validation():
    from threading import Barrier
    from scripts.liquidity.compare_csi300_boundary_spark import execute_histories
    barrier=Barrier(2);started=[];validated=[]
    def fit(date,show):
        started.append(date)
        if date!='ref':barrier.wait(timeout=5)
        return dict(date=date,show=show)
    execute_histories(['a','ref','b'],['ref'],2,fit,validated.append,lambda:None)
    assert started[0]=='ref' and validated[0]['date']=='ref'
    assert {r['date'] for r in validated}=={'a','b','ref'}
    assert all(not r['show'] for r in validated if r['date']!='ref')


def test_failed_reference_prevents_submission_of_other_histories():
    import pytest
    from scripts.liquidity.compare_csi300_boundary_spark import execute_histories,PosteriorValidationError
    started=[]
    def fit(date,show):started.append(date);return dict(date=date)
    def validate(record):raise PosteriorValidationError('failed reference')
    with pytest.raises(PosteriorValidationError):
        execute_histories(['a','ref','b'],['ref'],2,fit,validate,lambda:None)
    assert started==['ref']


def test_reference_fits_can_share_slots_but_keep_the_validation_barrier():
    from threading import Barrier
    from scripts.liquidity.compare_csi300_boundary_spark import execute_histories
    barrier=Barrier(2);validated=[]
    def fit(date,show):
        if date.startswith('ref'):barrier.wait(timeout=5)
        else:assert set(validated)=={'ref1','ref2'}
        return dict(date=date)
    execute_histories(['ref1','other','ref2'],['ref1','ref2'],2,fit,
                      lambda result:validated.append(result['date']),lambda:None)
    assert validated[-1]=='other'


def test_parallel_update_does_not_accept_different_sampling_settings():
    from scripts.liquidity.compare_csi300_boundary_spark import compatible_concurrency_update,CONCURRENCY_PREVIOUS_SHA256
    key='scripts/liquidity/compare_csi300_boundary_spark.py'
    previous=dict(series=1,groups=2,chains=1,source_sha256={key:CONCURRENCY_PREVIOUS_SHA256,'dts/mcmc.py':'same'})
    current={**previous,'history_workers':0,'source_sha256':{key:'current','dts/mcmc.py':'same'}}
    assert compatible_concurrency_update(previous,current)
    assert not compatible_concurrency_update(previous,{**current,'chains':4})
    assert parse_arguments(['--groups','2','--spark-slots','100']).history_workers==0
