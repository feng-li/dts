#!/usr/bin/env python3
"""Run/resume the production-length timing procedure from its saved config."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import signal

from benchmark_csi300_computation import sha256, stop_case, write_json


FILES = ['benchmark_csi300_computation.py','_computation_timing.py',
         'profile_csi300_timing.py','report_csi300_timing_comparison.py',
         'run_csi300_timing_comparison.py']


def configure(args):
    path = args.config or args.out/'config.json'
    if path.exists():
        config = json.loads(path.read_text())
        root = Path(config['out'])
    else:
        root = args.out.resolve()
        root.mkdir(parents=True,exist_ok=True)
        config = dict(version=1,out=str(root),python=sys.executable,
                      source=str(args.source.resolve()),production=str(args.production.resolve()),
                      reference=str(args.reference.resolve()),dates=args.dates,
                      iterations=args.iterations,burn=args.burn,groups=20,slots=20,repetitions=3,
                      cpus=args.cpus,seed=20261030,optimization_policy='robust',fft_backend='numpy',
                      quiet_seconds=args.quiet_seconds,quiet_timeout=86400,
                      interference_policy=args.interference_policy,
                      artifact_marker=os.environ.get('DTS_ARTIFACT_MARKER'),
                      case_max_seconds=7200,isolation='Monitored shared host; no exclusive reservation',
                      profile_lengths=sorted({2000,args.iterations}),control_repetitions=1)
        if config['iterations']-config['burn'] < 512 or config['burn'] < 0:
            raise ValueError('At least 512 retained draws and nonnegative burn-in are required')
        code = root/'code/scripts/liquidity'
        code.mkdir(parents=True,exist_ok=True)
        for name in FILES:
            shutil.copy2(Path(__file__).with_name(name),code/name)
        config['script_sha256']={name:sha256(code/name) for name in FILES}
        write_json(root/'config.json',config)
    for name,expected in config['script_sha256'].items():
        if sha256(root/'code/scripts/liquidity'/name) != expected:
            raise ValueError(f'Frozen controller changed: {name}')
    return root,config


def commands(root,config):
    code=root/'code/scripts/liquidity'
    common=[config['python'],str(code/'benchmark_csi300_computation.py'),
            '--source',config['source'],'--production',config['production'],
            '--reference',config['reference'],'--iterations',str(config['iterations']),
            '--burn',str(config['burn']),'--groups',str(config['groups']),
            '--slots',str(config['slots']),'--cpus',*map(str,config['cpus']),
            '--seed',str(config['seed']),'--optimization-policy',config['optimization_policy'],
            '--fft-backend',config['fft_backend'],'--detailed-timing',
            '--quiet-seconds',str(config['quiet_seconds']),
            '--interference-policy',config.get('interference_policy','exclude'),
            '--quiet-timeout',str(config['quiet_timeout']),
            '--case-max-seconds',str(config['case_max_seconds']),'--max-seconds','86400']
    phases=[]
    for date in config['dates']:
        output=root/'histories'/date
        command=common+['--out',str(output),'--dates',date,'--repetitions',str(config['repetitions']),
                        '--methods','full','dc_parallel']
        phases.append(dict(name='history_'+date,kind='fit',out=str(output),command=command,
                           iterations=config['iterations'],burn=config['burn'],
                           expected_cases=config['repetitions']*2))
        if date == config['dates'][0]:
            output=root/'sampling_only'
            command=[config['python'],str(code/'profile_csi300_timing.py'),
                     '--benchmark',str(root/'histories'/date),'--out',str(output),'--date',date,
                     '--lengths',*map(str,config['profile_lengths']),
                     '--repetitions',str(config['repetitions']),
                     '--quiet-seconds',str(config['quiet_seconds']),
                     '--interference-policy',config.get('interference_policy','exclude'),
                     '--quiet-timeout',str(config['quiet_timeout']),
                     '--case-max-seconds',str(config['case_max_seconds'])]
            phases.append(dict(name='sampling_only',kind='profile',out=str(output),command=command,
                               iterations=config['iterations'],burn=config['burn'],
                               expected_cases=len(config['profile_lengths'])*config['repetitions']*2))
            output=root/'controls'
            command=common+['--out',str(output),'--dates',date,'--repetitions','1',
                            '--methods','spark_g1','dc_serial']
            phases.append(dict(name='controls',kind='fit',out=str(output),command=command,
                               iterations=config['iterations'],burn=config['burn'],expected_cases=2))
    return phases


def verify_phase(phase):
    root=Path(phase['out'])
    saved=json.loads((root/'config.json').read_text())
    iterations=phase.get('iterations',saved.get('iterations',saved.get('reference_iterations')))
    burn=phase.get('burn',saved.get('burn',saved.get('reference_burn')))
    if iterations is None or burn is None:
        raise ValueError('The saved sampling budget is missing')
    complete=list(root.glob('cases/**/completed.json'))
    if len(complete) != phase['expected_cases']:
        raise ValueError(f'Incomplete phase {phase["name"]}: {len(complete)} cases')
    for path in complete:
        monitoring=json.loads(path.with_name('resource_monitor.json').read_text())
        if (not monitoring.get('finished') or monitoring.get('exit_code') != 0 or
            (monitoring.get('interference') and monitoring.get('policy','exclude')!='record')):
            raise ValueError(f'Unaccepted timing: {path.parent}')
        result=json.loads(path.read_text())
        if phase['kind']=='fit':
            if result['iterations'] != iterations or result['burn'] != burn:
                raise ValueError('Sampling budget differs from production')
            if result.get('global_map_seconds',0) and result['method'] != 'full':
                raise ValueError('DC used a global full-data MAP')
            parts=result['worker_wall_breakdown']
            if abs(sum(v for k,v in parts.items() if k.endswith('_seconds'))-result['action_seconds'])>1e-4:
                raise ValueError('Wall-time decomposition does not reconcile')
        else:
            if result['iterations'] not in saved['lengths']:
                raise ValueError('Profile iteration budget differs from its saved configuration')
            expected_burn=burn if result['iterations']==iterations else min(500,result['iterations']//4)
            if result['burn'] != expected_burn:
                raise ValueError('Profile burn-in differs from its saved configuration')
            if result['iterations']==iterations and not result['all_reference_replays_equal']:
                raise ValueError('Compiled-loop replay does not reproduce reference draws')
    return len(complete)


def archive_interrupted(phase,root):
    """Preserve failed attempts before an interference-triggered retry."""
    archived=[]
    for case in Path(phase['out']).glob('cases/**/resource_monitor.json'):
        if case.with_name('completed.json').exists():
            continue
        state=json.loads(case.read_text())
        if state.get('policy','exclude')=='record' or state.get('stopped_for')!='competing-computation':
            continue
        destination=root/'excluded'/phase['name']/(case.parent.name+'_'+str(time.time_ns()))
        destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.move(str(case.parent),destination)
        archived.append(str(destination))
    return archived


def main(args):
    root,config=configure(args)
    lock=(root/'procedure.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    phases=commands(root,config)
    write_json(root/'commands.json',phases)
    write_json(root/'process.json',dict(pid=os.getpid(),started_at=time.time(),argv=sys.argv))
    environment=dict(os.environ,PYSPARK_PYTHON=config['python'],
                     OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',
                     SPARK_LOCAL_HOSTNAME='localhost',SPARK_LOCAL_IP='127.0.0.1')
    if args.prepare_only:
        for phase in phases:
            if phase['kind']=='fit':
                subprocess.run(phase['command']+['--prepare-only'],env=environment,check=True)
        return
    from tqdm import tqdm
    for phase in tqdm(phases,desc='Timing procedure phases'):
        write_json(root/'progress.json',dict(phase=phase['name'],status='running',updated_at=time.time()))
        print('\nPHASE '+phase['name'],flush=True)
        attempt=0
        while True:
            attempt+=1
            with (root/(phase['name']+'.log')).open('a') as log:
                log.write(f'\nAttempt {attempt} started {time.time()}\n');log.flush()
                process=subprocess.Popen(phase['command'],stdout=log,stderr=subprocess.STDOUT,env=environment,start_new_session=True)
                write_json(root/'phase_process.json',dict(phase=phase['name'],pid=process.pid,
                    started_at=time.time(),command=phase['command']))
                try:
                    code=process.wait()
                except BaseException:
                    stop_case(process)
                    raise
            if code==0:
                break
            archived=archive_interrupted(phase,root)
            if not archived:
                raise RuntimeError(f'Phase failed; inspect {root/(phase["name"]+".log")}')
            print('Interference: excluded attempt; waiting for quiet host before retry',flush=True)
        print(f'Verified {verify_phase(phase)} cases for {phase["name"]}',flush=True)
    command=[config['python'],str(root/'code/scripts/liquidity/report_csi300_timing_comparison.py'),
             '--input',str(root)]
    marker=root/'artifact_operation_started.json'
    if not marker.exists():
        output=''
        if config.get('artifact_marker'):
            marked=subprocess.run(['node',config['artifact_marker'],'--operation-kind','create',
                                   '--expected-output-count','1','--output-format','pdf'],
                                  env=environment,check=True,capture_output=True,text=True)
            output=marked.stdout
        write_json(marker,dict(returncode=0,output=output,external_marker=bool(config.get('artifact_marker'))))
    subprocess.run(command,env=environment,check=True)
    write_json(root/'progress.json',dict(status='complete',updated_at=time.time()))
    print(f'Completed timing procedure: {root}',flush=True)


if __name__=='__main__':
    def interrupted(signum, frame):
        raise KeyboardInterrupt('Timing procedure interrupted; rerun with --config to resume')
    signal.signal(signal.SIGTERM, interrupted)
    base=Path.home()/'running/dts/artifacts/liquidity'
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=base/'csi300_timing_65000_g20_longest_20261109_v1')
    parser.add_argument('--config',type=Path)
    parser.add_argument('--source',type=Path,default=base/'csi300_computation_independent_maps_20261102_v1/source')
    parser.add_argument('--production',type=Path,default=base/'csi300_boundary_spark_g20_full1000_onechain_v5')
    parser.add_argument('--reference',type=Path,default=base/'csi300_boundary_full1000_v1')
    parser.add_argument('--dates',nargs='+',default=['20260724'],
                        help='Forecast dates whose saved training histories are timed')
    parser.add_argument('--iterations',type=int,default=65000,
                        help='Iterations per chain for a new config; a saved config remains authoritative')
    parser.add_argument('--burn',type=int,default=5000)
    parser.add_argument('--cpus',type=int,nargs='+',default=list(range(26,46)))
    parser.add_argument('--quiet-seconds',type=int,default=60)
    parser.add_argument('--interference-policy',choices=['exclude','record'],default='exclude')
    parser.add_argument('--prepare-only',action='store_true')
    main(parser.parse_args())
