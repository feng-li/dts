#!/usr/bin/env python3
"""Prepare evidence for the co-author timing questions after measured runs.

An optional short-chain comparison is queued after the main runner exits.
Numerical experiments never overlap. The remaining work reads saved results.
"""
from __future__ import annotations

import argparse
import csv
import fcntl
import importlib.metadata
import json
import os
from pathlib import Path
from statistics import median
import subprocess
import sys
import time
import signal
import shutil


def save(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def csv_file(path,rows):
    if not rows:return
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def middle(values):
    return median(values) if values else None


def fitted_cases(root,short=False):
    pattern='short_fits/cases/*/*/completed.json' if short else 'histories/*/cases/*/*/completed.json'
    paths=list(root.glob(pattern))
    if not short:paths+=list(root.glob('controls/cases/*/*/completed.json'))
    return [(path,json.loads(path.read_text())) for path in sorted(paths)]


def socket_layouts(root):
    config=json.loads((root/'config.json').read_text())
    date=config['dates'][0]
    host=json.loads((root/'histories'/date/'host.json').read_text())
    physical={}
    for line in host['cpu_topology'].splitlines():
        if not line or line.startswith('#'):continue
        cpu,core,socket=map(int,line.split(','))
        physical.setdefault((socket,core),cpu)
    sockets=sorted({socket for socket,_ in physical})
    if len(sockets)<2:return None
    selected=[]
    for socket in sockets[:2]:
        candidates=sorted(cpu for (node,core),cpu in physical.items() if node==socket and cpu!=0)
        if len(candidates)<10:raise ValueError('Insufficient physical cores for balanced socket test')
        selected.extend(candidates[:10])
    return dict(one_socket=config['cpus'][:20],balanced_sockets=selected)


def measured_command(command,log_path,metadata_path):
    while True:
        with log_path.open('a') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            save(metadata_path,dict(pid=child.pid,command=command,started_at=time.time()))
            try:code=child.wait()
            except BaseException:
                try:os.killpg(child.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                child.wait(timeout=30)
                raise
        if not code:return code
        destination=Path(command[command.index('--out')+1])
        archived=[]
        for path in destination.glob('cases/**/resource_monitor.json'):
            state=json.loads(path.read_text())
            if (path.with_name('completed.json').exists() or
                state.get('policy','exclude')!='exclude' or
                state.get('stopped_for')!='competing-computation'):continue
            target=destination/'excluded'/(path.parent.name+'_'+str(time.time_ns()))
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(path.parent),target);archived.append(str(target))
        if not archived:return code
        print('Excluded competing-load attempt; retrying after the quiet-host check',flush=True)


def layout_controls(root):
    out=root/'reply_support';layouts=socket_layouts(root)
    if layouts is None:return
    save(out/'socket_layouts.json',layouts)
    config=json.loads((root/'config.json').read_text())
    date=config['dates'][0]
    profiler=out/'diagnostic_code/profile_csi300_timing.py'
    for pair in range(1,4):
        names=['one_socket','balanced_sockets'] if pair%2 else ['balanced_sockets','one_socket']
        for name in names:
            destination=out/'socket_controls'/f'pair{pair}'/name
            command=[config['python'],str(profiler),'--benchmark',str(root/'histories'/date),
                '--out',str(destination),'--date',date,'--lengths',str(config['iterations']),'--modes','dc_parallel',
                '--repetitions','1','--cpus',*map(str,layouts[name]),
                '--quiet-seconds',str(config['quiet_seconds']),
                '--interference-policy',config.get('interference_policy','exclude'),
                '--case-max-seconds',str(config['case_max_seconds'])]
            save(out/'investigation_status.json',dict(status='socket_layout_control',pair=pair,layout=name,pid=os.getpid(),updated_at=time.time()))
            code=measured_command(command,out/'socket_controls.log',out/'socket_control_process.json')
            if code:raise RuntimeError('Socket placement control failed; inspect socket_controls.log')


def analyze(root):
    out=root/'reply_support';out.mkdir(exist_ok=True)
    date=json.loads((root/'config.json').read_text())['dates'][0]
    cases=fitted_cases(root)
    questions=[]
    for path,result in cases:
        workers=[json.loads(p.read_text()) for p in sorted(path.parent.glob('shards/group*.json'))]
        if len(workers)!=result['groups']:raise ValueError('Incomplete worker metadata')
        if any(len(w['thread_cpus'])!=1 for w in workers):raise ValueError('A numerical worker used multiple cores')
        if result['method']=='dc_parallel' and len({w['thread_cpus'][0] for w in workers})!=20:
            raise ValueError('Parallel workers did not have distinct cores')
        if result['method']!='dc_parallel':continue
        w=result['worker_wall_breakdown']
        local=sum(w[key] for key in ('native_map_seconds','coordinate_map_seconds',
                    'loop_compile_seconds','compiled_sampling_seconds','covariance_seconds'))
        durations=[item['worker_finished_at']-item['worker_started_at'] for item in workers]
        ends=[item['worker_finished_at'] for item in workers]
        questions.append(dict(date=result['date'],repetition=result['repetition'],
            fit_seconds=result['fit_seconds'],total_seconds=result['all_in_seconds'],
            startup_seconds=result['startup_seconds'],fft_preparation_seconds=result['fft_seconds'],
            action_residual_seconds=w['scheduling_wait_other_seconds'],
            aggregation_seconds=result['aggregation_seconds'],
            local_compiled_mcmc_seconds=w['compiled_sampling_seconds'],
            local_covariance_seconds=w['covariance_seconds'],
            nonlocal_upper_bound_fit_seconds=result['fit_seconds']-local,
            nonlocal_upper_bound_fit_percent=100*(result['fit_seconds']-local)/result['fit_seconds'],
            nonlocal_upper_bound_total_percent=100*(result['all_in_seconds']-local)/result['all_in_seconds'],
            worker_min_seconds=min(durations),worker_median_seconds=median(durations),
            worker_max_seconds=max(durations),max_to_median_worker_ratio=max(durations)/median(durations),
            tail_after_median_finish_seconds=max(ends)-median(ends),
            tail_percent_of_fit=100*(max(ends)-median(ends))/result['fit_seconds'],
            median_mcmc_cpu_to_wall=median(x['sampling_cpu_seconds']/x['compiled_sampling_seconds'] for x in workers)))
    csv_file(out/'parallel_explanation_cases.csv',questions)
    summaries=[]
    for scope,items in [('production',cases),('short',fitted_cases(root,True))]:
        for date,method in sorted({(r['date'],r['method']) for _,r in items}):
            selected=[r for _,r in items if r['date']==date and r['method']==method]
            summaries.append(dict(scope=scope,date=date,method=method,n=selected[0]['n'],
                iterations=selected[0]['iterations'],repetitions=len(selected),
                fit_seconds=median(r['fit_seconds'] for r in selected),
                total_seconds=median(r['all_in_seconds'] for r in selected),
                fit_min=min(r['fit_seconds'] for r in selected),fit_max=max(r['fit_seconds'] for r in selected)))
    csv_file(out/'sampling_length_fit_comparison.csv',summaries)

    # Same group, proposal stream, target and compiled loop under serial and
    # concurrent execution. Draw equality is tested, not assumed.
    import numpy as np
    serial=[]
    base=root/'controls/cases'/date/'dc_serial_r1'
    parallel=root/'histories'/date/'cases'/date/'dc_parallel_r1'
    if (base/'completed.json').exists() and (parallel/'completed.json').exists():
        for group in range(20):
            filename=f'group{group:03d}'
            a=json.loads((base/'shards'/f'{filename}.json').read_text())
            b=json.loads((parallel/'shards'/f'{filename}.json').read_text())
            da=np.load(base/'shards'/f'{filename}_draws.npy')
            db=np.load(parallel/'shards'/f'{filename}_draws.npy')
            with np.load(base/'frequencies'/f'shard{group:03d}.npz') as fa, np.load(parallel/'frequencies'/f'shard{group:03d}.npz') as fb:
                same_frequencies=all(np.array_equal(fa[key],fb[key]) for key in ('q','periodogram','omega'))
            serial.append(dict(group=group,serial_mcmc_seconds=a['compiled_sampling_seconds'],
                parallel_mcmc_seconds=b['compiled_sampling_seconds'],
                parallel_to_serial_time_ratio=b['compiled_sampling_seconds']/a['compiled_sampling_seconds'],
                serial_cpu_to_wall=a['sampling_cpu_seconds']/a['compiled_sampling_seconds'],
                parallel_cpu_to_wall=b['sampling_cpu_seconds']/b['compiled_sampling_seconds'],
                identical_frequencies=same_frequencies,bitwise_equal_draws=bool(np.array_equal(da,db)),
                max_draw_difference=float(np.max(abs(da-db)))))
        csv_file(out/'serial_parallel_kernel_comparison.csv',serial)

    frames=[]
    for path in out.glob('resources_*.jsonl'):
        with path.open() as handle:
            for line in handle:
                if line.strip():frames.append(json.loads(line))
    memory=[]
    for path,result in cases:
        start=json.loads(path.with_name('process.json').read_text())['started_at']
        end=result['finished_at']
        workers=[json.loads(p.read_text()) for p in sorted(path.parent.glob('shards/group*.json'))]
        pids={w['pid'] for w in workers}
        observations=[f for f in frames if start<=f['time']<=end]
        numerical=[p for f in observations for p in f['processes'] if p['pid'] in pids]
        if not numerical:continue
        freq=[]
        for f in observations:
            if 'monotonic' not in f:continue
            for w in workers:
                if w['sampling_started_at']<=f['monotonic']<=w['sampling_finished_at']:
                    value=f['scaling_frequency_khz'].get(str(w['thread_cpus'][0]))
                    if value is not None:freq.append(value/1000.)
        memory.append(dict(date=result['date'],method=result['method'],repetition=result['repetition'],
            diagnostic_samples=len(observations),coverage_seconds=max(f['time'] for f in observations)-min(f['time'] for f in observations),
            elapsed_seconds=result['all_in_seconds'],
            max_worker_rss_gib=max(p['rss_kib'] for p in numerical)/1024**2,
            max_worker_swap_kib=max(p['swap_kib'] for p in numerical),
            min_host_available_gib=min(f['host_memory_kib']['MemAvailable'] for f in observations)/1024**2,
            host_swapin_pages_delta=max(f['host_swap_pages']['pswpin'] for f in observations)-min(f['host_swap_pages']['pswpin'] for f in observations),
            host_swapout_pages_delta=max(f['host_swap_pages']['pswpout'] for f in observations)-min(f['host_swap_pages']['pswpout'] for f in observations),
            kernel_scaling_frequency_mhz_median=middle(freq),frequency_samples=len(freq)))
    csv_file(out/'memory_and_frequency_observations.csv',memory)
    layouts=[]
    for path in sorted(out.glob('socket_controls/pair*/*/cases/*/completed.json')):
        result=json.loads(path.read_text())
        if not result['all_reference_replays_equal']:
            raise ValueError('Socket placement changed the reference MCMC replay')
        layouts.append(dict(pair=path.parents[3].name,layout=path.parents[2].name,
            sampling_wall_seconds=result['sampling_wall_seconds'],
            max_compile_seconds=result['max_compile_seconds'],
            reference_replay_equal=result['all_reference_replays_equal']))
    csv_file(out/'socket_layout_cases.csv',layouts)
    summary=dict(updated_at=time.time(),parallel_cases=questions,fit_comparisons=summaries,
        serial_parallel=serial,memory=memory,socket_layouts=layouts,
        qualifications=['Single shared-memory host, Spark local[20]; no per-iteration cross-shard communication.',
            'Nonlocal upper bounds include FFT arithmetic, RNG preparation, imports and other local work; they are not direct communication measurements.',
            'Scaling frequency is a sampled kernel-reported value, not a cycle-weighted measurement or proof of a causal bottleneck.',
            'Resource coverage is reported explicitly for each measured case.',
            'RSS is per process and shared pages may be counted in more than one process. Host swap counters can include unrelated applications.'])
    save(out/'investigation_results.json',summary)
    return summary


def wait_and_compare(root):
    out=root/'reply_support';out.mkdir(exist_ok=True)
    lock=(out/'investigation.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root_pid=json.loads((root/'process.json').read_text())['pid']
    while Path(f'/proc/{root_pid}/cmdline').exists():
        command=Path(f'/proc/{root_pid}/cmdline').read_bytes()
        if b'run_csi300_timing_comparison.py' not in command:break
        save(out/'investigation_status.json',dict(status='waiting_for_main_timing',pid=os.getpid(),root_pid=root_pid,updated_at=time.time()))
        time.sleep(15)
    if json.loads((root/'progress.json').read_text()).get('status')!='complete':
        raise RuntimeError('Main timing stopped before completion; no follow-up experiment was started')
    # Match current numerical software rather than comparing new runs with
    # old timings whose package versions may differ.
    expected=json.loads((out/'environment.json').read_text())['packages']
    for name,value in expected.items():
        try:actual=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:actual=None
        if actual!=value:raise RuntimeError(f'Numerical package version changed: {name}')
    phases=json.loads((root/'commands.json').read_text())
    date=json.loads((root/'config.json').read_text())['dates'][0]
    first=next(x for x in phases if x['name']=='history_'+date)
    command=list(first['command'])
    for key,value in [('--out',str(root/'short_fits')),('--iterations','2000'),('--burn','500')]:
        command[command.index(key)+1]=value
    save(out/'short_fit_command.json',dict(command=command,scope='Matched short-chain fitting control; same source, target and resources; after main timing'))
    save(out/'investigation_status.json',dict(status='short_chain_fits',pid=os.getpid(),updated_at=time.time()))
    code=measured_command(command,out/'short_fits.log',out/'short_fit_process.json')
    if code:raise RuntimeError('Short-chain fitting control failed; inspect short_fits.log')
    if len(fitted_cases(root,True))!=6:raise RuntimeError('Six matched short fitting cases are required')
    layout_controls(root)
    os.sched_setaffinity(0,{0})
    result=analyze(root)
    save(out/'investigation_status.json',dict(status='complete',updated_at=time.time(),parallel_cases=len(result['parallel_cases'])))


if __name__=='__main__':
    def interrupted(signum,frame):
        raise KeyboardInterrupt('Timing investigation interrupted; saved cases remain available')
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--wait-and-short-fit',action='store_true')
    args=parser.parse_args();root=args.input.resolve()
    wait_and_compare(root) if args.wait_and_short_fit else analyze(root)
