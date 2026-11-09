#!/usr/bin/env python3
"""Resume the timing study, diagnostics, audit and guarded TeX update in order."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def save(path,value):
    temporary=path.with_suffix('.partial');temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def main(root):
    root=root.resolve();config=json.loads((root/'config.json').read_text())
    pipeline=json.loads((root/'pipeline_config.json').read_text())
    lock=(root/'pipeline.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.sched_setaffinity(0,{0})
    for name,expected in pipeline['script_sha256'].items():
        if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'Frozen pipeline code changed: {name}')
    save(root/'pipeline_process.json',dict(pid=os.getpid(),started_at=time.time()))
    if not (root/'progress.json').exists():save(root/'progress.json',dict(status='queued'))
    out=root/'reply_support';out.mkdir(exist_ok=True)
    stop=out/f'observer_stop_{os.getpid()}'
    observer_command=[config['python'],str(root/'pipeline_code/observe_csi300_timing_resources.py'),
        '--input',str(root),'--root-pid',str(os.getpid()),'--cpus',*map(str,pipeline['observer_cpus']),
        '--stop-file',str(stop),'--interval','10']
    environment=dict(os.environ,PYSPARK_PYTHON=config['python'],OPENBLAS_NUM_THREADS='1',
        OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1')
    with (out/'observer.log').open('a') as log:
        observer=subprocess.Popen(observer_command,env=environment,stdout=log,stderr=subprocess.STDOUT)
    stages=[('timing',[config['python'],str(root/'code/scripts/liquidity/run_csi300_timing_comparison.py'),'--config',str(root/'config.json')]),
            ('diagnostics',[config['python'],str(out/'investigate_csi300_timing.py'),'--input',str(root),'--wait-and-short-fit'])]
    try:
        for name,command in stages:
            save(root/'pipeline_status.json',dict(status='running',phase=name,updated_at=time.time(),command=command))
            subprocess.run(command,env=environment,check=True)
        stop.touch();observer.wait(timeout=30)
        if observer.returncode:raise RuntimeError('The resource observer failed')
        for name,command in [
            ('audit',[config['python'],str(out/'audit_dts_timing_results.py')]),
            ('tex_update',[config['python'],str(root/'pipeline_code/update_csi300_timing_tex.py'),'--input',str(root)])]:
            save(root/'pipeline_status.json',dict(status='running',phase=name,updated_at=time.time(),command=command))
            subprocess.run(command,env=environment,check=True)
        save(root/'pipeline_status.json',dict(status='computed_and_tex_updated',visual_review='pending',updated_at=time.time()))
    except BaseException as exc:
        save(root/'pipeline_status.json',dict(status='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',error=str(exc),updated_at=time.time()))
        raise
    finally:
        stop.touch()
        if observer.poll() is None:
            try:observer.wait(timeout=30)
            except subprocess.TimeoutExpired:observer.terminate();observer.wait(timeout=10)


if __name__=='__main__':
    def interrupted(signum,frame):
        raise KeyboardInterrupt('Pipeline interrupted; completed cases remain resumable')
    signal.signal(signal.SIGTERM,interrupted)
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    main(parser.parse_args().input)
