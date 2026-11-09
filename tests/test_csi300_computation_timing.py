"""Numerical identity and wall-time accounting for the timing experiment."""
import os

import numpy as np
import pytest

from scripts.liquidity._computation_timing import (
    coordinate_step_timing, critical_path_breakdown)
from scripts.liquidity.benchmark_csi300_computation import (
    competing_activity, descendant_pids, uses_spark)


def test_instrumentation_preserves_production_sampler_draws_and_random_stream():
    from scripts.liquidity import _spectral_coordinates as coordinates

    omega = 2*np.pi*np.arange(1, 64)/128
    pg = np.linspace(.5, 1.5, len(omega))
    center = coordinates.to_coordinates(np.array([-.5, -4., 0., .1]))
    arguments = (center, np.eye(4)*1e-6, pg, omega, 20, 80, 20, 415)
    expected = coordinates.coordinate_sampler(*arguments, kernel='mixture', center=center)
    original = coordinates._coordinate_steps
    with coordinate_step_timing(coordinates) as times:
        actual = coordinates.coordinate_sampler(*arguments, kernel='mixture', center=center)
    assert coordinates._coordinate_steps is original
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)
    assert times['loop_compile_seconds'] >= 0
    assert times['compiled_sampling_seconds'] > 0
    assert times['sampling_started_at'] < times['sampling_finished_at']


def test_timing_hook_is_restored_after_an_exception():
    from scripts.liquidity import _spectral_coordinates as coordinates

    original = coordinates._coordinate_steps
    with pytest.raises(RuntimeError, match='test interruption'):
        with coordinate_step_timing(coordinates):
            raise RuntimeError('test interruption')
    assert coordinates._coordinate_steps is original


def test_parallel_accounting_uses_terminal_worker_not_sum_of_workers():
    workers = [dict(group=0, worker_finished_at=10., compiled_sampling_seconds=7.),
               dict(group=1, worker_finished_at=11., compiled_sampling_seconds=5.,
                    covariance_seconds=2.)]
    result = critical_path_breakdown(workers, 12., parallel=True)
    assert result['terminal_groups'] == [1]
    assert result['compiled_sampling_seconds'] == 5.
    assert result['scheduling_wait_other_seconds'] == 5.
    assert sum(v for k, v in result.items() if k.endswith('_seconds')) == 12.


def test_sequential_accounting_includes_every_worker():
    workers = [dict(group=0, compiled_sampling_seconds=7.),
               dict(group=1, compiled_sampling_seconds=5., covariance_seconds=2.)]
    result = critical_path_breakdown(workers, 16., parallel=False)
    assert result['compiled_sampling_seconds'] == 12.
    assert result['scheduling_wait_other_seconds'] == 2.


def test_invalid_wall_time_allocation_is_rejected():
    with pytest.raises(ValueError, match='exceed'):
        critical_path_breakdown([dict(group=0, compiled_sampling_seconds=2.)], 1., False)


def test_activity_monitor_ignores_browser_and_benchmark_descendants():
    ticks = os.sysconf('SC_CLK_TCK')
    uid = os.getuid()
    before = {pid: (0, 0, uid, comm) for pid, comm in
              [(1, 'RDD Process'), (2, 'python3.12'), (3, 'ffmpeg'), (4, 'R')]}
    current = {pid: (0, ticks, uid, value[3]) for pid, value in before.items()}
    active = competing_activity(before, current, 1., excluded={2})
    assert {item['pid'] for item in active} == {3, 4}


def test_descendant_search_handles_deep_process_trees():
    processes = {pid: (pid-1, 0, 1000, 'python') for pid in range(2, 25)}
    assert descendant_pids(processes, 1) == set(range(1, 25))


@pytest.mark.parametrize('method, expected', [
    ('full', False), ('arima011', False), ('spark_g1', True),
    ('dc_parallel', True), ('dc_serial', True)])
def test_spark_control_uses_same_distributed_pipeline(method, expected):
    assert uses_spark(method) is expected


def test_report_distinguishes_practical_and_internal_speedups():
    from scripts.liquidity.report_csi300_timing_comparison import comparisons

    rows=[dict(date='20211013',n=100,method=method,fit_seconds=seconds,
               all_in_seconds=seconds+10,cores=cores) for method,seconds,cores in
          [('full',100.,1),('dc_parallel',10.,20),('dc_serial',150.,1),('spark_g1',120.,1)]]
    gain=comparisons(rows)[0]
    assert gain['full_to_parallel_fit']==10.
    assert gain['full_to_parallel_all_in']==5.5
    assert gain['dc_serial_to_parallel_fit']==15.
    assert gain['spark_g1_to_parallel_fit']==12.
    assert gain['parallel_efficiency_same_dc']==.75


def test_cpu_binding_covers_reused_worker_threads():
    import subprocess
    import sys

    code='''
import os, threading
from pathlib import Path
from scripts.liquidity.benchmark_csi300_computation import pin_cpu
event=threading.Event()
thread=threading.Thread(target=event.wait)
thread.start()
try:
    cpu=min(os.sched_getaffinity(0))
    pin_cpu(cpu)
    assert all(os.sched_getaffinity(int(p.name)) == {cpu} for p in Path('/proc/self/task').iterdir())
finally:
    event.set()
    thread.join()
'''
    subprocess.run([sys.executable,'-c',code],check=True)


def test_profile_worker_can_import_pickled_helpers_from_frozen_cwd(tmp_path,monkeypatch):
    import runpy
    import subprocess
    import sys
    from pathlib import Path

    folder=Path(__file__).resolve().parents[1]/'scripts/liquidity'
    monkeypatch.syspath_prepend(str(folder))
    configure=runpy.run_path(str(folder/'profile_csi300_timing.py'))['configure_spark_environment']
    pythonpath=configure(tmp_path)
    assert pythonpath.split(os.pathsep)[:2]==[str(tmp_path),str(folder)]
    subprocess.run([sys.executable,'-c',
        'import benchmark_csi300_computation as b; assert callable(b.write_json); assert callable(b.pin_cpu)'],
        cwd=tmp_path,env=dict(os.environ,PYTHONPATH=pythonpath),check=True)


@pytest.mark.parametrize('policy', ['exclude','record'])
def test_monitor_retains_observations_under_explicit_shared_host_policy(monkeypatch,tmp_path,policy):
    import json
    from types import SimpleNamespace
    from scripts.liquidity import benchmark_csi300_computation as benchmark

    polls=iter([None,None,None,0])
    process=SimpleNamespace(pid=10,poll=lambda:next(polls),returncode=0)
    ticks=os.sysconf('SC_CLK_TCK')
    snapshots=iter([{20:(0,index*2*ticks,os.getuid(),'python')} for index in range(4)])
    clock=iter([0.,2.,4.,6.])
    stopped=[]
    monkeypatch.setattr(benchmark,'process_ticks',lambda:next(snapshots))
    monkeypatch.setattr(benchmark.time,'monotonic',lambda:next(clock))
    monkeypatch.setattr(benchmark.time,'sleep',lambda seconds:None)
    monkeypatch.setattr(benchmark,'stop_case',lambda child:stopped.append(child.pid))
    if policy=='exclude':
        with pytest.raises(RuntimeError,match='competing computation'):
            benchmark.monitor_process(process,tmp_path/'monitor.json',60,policy)
        assert stopped==[10]
    else:
        assert benchmark.monitor_process(process,tmp_path/'monitor.json',60,policy)==0
        record=json.loads((tmp_path/'monitor.json').read_text())
        assert len(record['interference'])==3 and record['finished']
        assert record['policy']=='record' and not stopped


@pytest.mark.parametrize('cached_wrong_package',[False,True])
def test_backend_uses_frozen_source_or_rejects_stale_package(tmp_path,cached_wrong_package):
    import subprocess,sys
    from pathlib import Path

    for name in ('frozen','outside'):
        root=tmp_path/name
        for package in ('scripts','scripts/liquidity','dts'):
            folder=root/package;folder.mkdir(parents=True,exist_ok=True)
            (folder/'__init__.py').write_text('')
        for module in ('probe_csi300_fractional_changes','_spectral_coordinates','_spectral_covariance'):
            (root/'scripts/liquidity'/f'{module}.py').write_text('marker=1\n')
        (root/'scripts/liquidity/compare_csi300_boundary_spark.py').write_text('def sampling_quality(x): return x\n')
        (root/'dts/mcmc.py').write_text('marker=1\n')
    controller=Path(__file__).resolve().parents[1]/'scripts/liquidity/benchmark_csi300_computation.py'
    code='''
import runpy,sys
from pathlib import Path
load=runpy.run_path(sys.argv[1])['backend']
frozen,outside,cached=sys.argv[2:]
sys.path.insert(0,outside)
sys.path.append(frozen)
if cached=='True':
    import scripts.liquidity.probe_csi300_fractional_changes
    try:load(frozen)
    except RuntimeError as exc:assert 'outside frozen source' in str(exc)
    else:raise AssertionError('A cached outside package was silently accepted')
else:
    probe,*_=load(frozen)
    assert Path(probe.__file__).resolve().is_relative_to(Path(frozen))
    assert sys.path[0]==frozen
'''
    subprocess.run([sys.executable,'-c',code,str(controller),str(tmp_path/'frozen'),
                    str(tmp_path/'outside'),str(cached_wrong_package)],check=True)


@pytest.mark.parametrize('stopped_for,policy,retry',[
    (None,'record',False),(None,'exclude',False),
    ('time-budget','exclude',False),('competing-computation','exclude',True)])
def test_retry_does_not_hide_startup_or_sampling_errors(tmp_path,monkeypatch,stopped_for,policy,retry):
    import json,runpy
    from pathlib import Path

    folder=Path(__file__).resolve().parents[1]/'scripts/liquidity'
    monkeypatch.syspath_prepend(str(folder))
    archive=runpy.run_path(str(folder/'run_csi300_timing_comparison.py'))['archive_interrupted']
    phase=dict(name='test',out=str(tmp_path/'phase'))
    case=tmp_path/'phase/cases/trial';case.mkdir(parents=True)
    (case/'resource_monitor.json').write_text(json.dumps(dict(
        policy=policy,interference=[dict(pid=42)],stopped_for=stopped_for)))
    archived=archive(phase,tmp_path)
    assert bool(archived)==retry
    assert case.exists()!=retry


@pytest.mark.parametrize('iterations',[35000,65000])
def test_completion_checks_the_saved_budget_instead_of_a_fixed_chain_length(tmp_path,monkeypatch,iterations):
    import json,runpy
    from pathlib import Path

    folder=Path(__file__).resolve().parents[1]/'scripts/liquidity'
    monkeypatch.syspath_prepend(str(folder))
    verify=runpy.run_path(str(folder/'run_csi300_timing_comparison.py'))['verify_phase']
    (tmp_path/'config.json').write_text(json.dumps(dict(iterations=iterations,burn=5000)))
    case=tmp_path/'cases/full';case.mkdir(parents=True)
    (case/'resource_monitor.json').write_text(json.dumps(dict(finished=True,exit_code=0)))
    result=dict(iterations=iterations,burn=5000,method='full',action_seconds=2.,
                worker_wall_breakdown=dict(compiled_sampling_seconds=2.))
    (case/'completed.json').write_text(json.dumps(result))
    phase=dict(out=str(tmp_path),kind='fit',expected_cases=1)
    assert verify(phase)==1
    result['iterations']=35000 if iterations==65000 else 65000
    (case/'completed.json').write_text(json.dumps(result))
    with pytest.raises(ValueError,match='Sampling budget'):
        verify(phase)


def test_report_labels_measured_budget_and_rejects_mixed_budgets():
    from scripts.liquidity.report_csi300_timing_comparison import sampling_budget_text

    assert sampling_budget_text([dict(iterations=65000,burn=5000)]).startswith('65,000 iterations, 5,000 burn-in')
    assert sampling_budget_text([dict(iterations=35000,burn=5000)]).startswith('35,000 iterations, 5,000 burn-in')
    with pytest.raises(ValueError,match='common fitting'):
        sampling_budget_text([dict(iterations=35000,burn=5000),dict(iterations=65000,burn=5000)])


def test_profile_completion_requires_the_requested_length_and_matching_draws(tmp_path,monkeypatch):
    import json,runpy
    from pathlib import Path

    folder=Path(__file__).resolve().parents[1]/'scripts/liquidity'
    monkeypatch.syspath_prepend(str(folder))
    verify=runpy.run_path(str(folder/'run_csi300_timing_comparison.py'))['verify_phase']
    (tmp_path/'config.json').write_text(json.dumps(dict(reference_iterations=65000,
        reference_burn=5000,lengths=[2000,65000])))
    case=tmp_path/'cases/profile';case.mkdir(parents=True)
    (case/'resource_monitor.json').write_text(json.dumps(dict(finished=True,exit_code=0)))
    phase=dict(out=str(tmp_path),kind='profile',expected_cases=1)
    for iterations,replay in [(35000,True),(65000,False),(65000,True)]:
        (case/'completed.json').write_text(json.dumps(dict(iterations=iterations,burn=5000,
            all_reference_replays_equal=replay)))
        if iterations==65000 and replay:
            assert verify(phase)==1
        else:
            with pytest.raises(ValueError):verify(phase)


@pytest.mark.parametrize('reason,expected_calls',[(None,1),('time-budget',1),('competing-computation',2)])
def test_followup_retries_only_excluded_competing_load(tmp_path,monkeypatch,reason,expected_calls):
    import json
    from types import SimpleNamespace
    from scripts.liquidity import investigate_csi300_timing as investigation

    case=tmp_path/'out/cases/test';case.mkdir(parents=True)
    (case/'resource_monitor.json').write_text(json.dumps(dict(policy='exclude',stopped_for=reason)))
    calls=[]
    def launch(*args,**kwargs):
        calls.append(args)
        code=1 if len(calls)==1 else 0
        return SimpleNamespace(pid=42,wait=lambda:code)
    monkeypatch.setattr(investigation.subprocess,'Popen',launch)
    result=investigation.measured_command(['python','worker.py','--out',str(tmp_path/'out')],
        tmp_path/'run.log',tmp_path/'process.json')
    assert len(calls)==expected_calls
    assert result==(0 if expected_calls==2 else 1)
    assert case.exists()==(expected_calls==1)


def test_tex_publisher_rejects_shorter_run_results():
    from scripts.liquidity.update_csi300_timing_tex import render_tex

    with pytest.raises(ValueError,match='Only measured 65,000'):
        render_tex('',[dict(iterations=35000,burn=5000)],None,None,None)


def test_longest_only_protocol_has_no_smaller_history_in_any_phase(tmp_path,monkeypatch):
    import json,runpy
    from pathlib import Path

    folder=Path(__file__).resolve().parents[1]/'scripts/liquidity'
    monkeypatch.syspath_prepend(str(folder))
    commands=runpy.run_path(str(folder/'run_csi300_timing_comparison.py'))['commands']
    config=dict(python='python',source='source',production='production',reference='reference',
        iterations=65000,burn=5000,groups=20,slots=20,cpus=list(range(26,46)),seed=1,
        optimization_policy='robust',fft_backend='numpy',quiet_seconds=60,quiet_timeout=86400,
        case_max_seconds=10800,dates=['20260724'],repetitions=3,profile_lengths=[2000,65000])
    phases=commands(tmp_path,config)
    assert [x['name'] for x in phases]==['history_20260724','sampling_only','controls']
    assert [x['expected_cases'] for x in phases]==[6,12,2]
    assert '20211013' not in json.dumps(phases)
