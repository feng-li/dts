#!/usr/bin/env python3
"""Observe our benchmark's CPU binding, RSS and swapping without changing it.

Only processes descended from the verified timing runner are inspected in
detail. Hardware and host swap counters are diagnostic context, not timings.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def info(pid):
    path=Path('/proc')/str(pid)
    raw=(path/'stat').read_text()
    fields=raw[raw.rindex(')')+2:].split()
    status=dict(line.split(':',1) for line in (path/'status').read_text().splitlines() if ':' in line)
    return path,fields,status


def memory_value(status,name):
    return int(status.get(name,'0 kB').split()[0])


def collect_process(pid):
    path,fields,status=info(pid)
    command=(path/'cmdline').read_bytes()
    name=status.get('Name','').strip()
    role='runtime'
    if b'benchmark_csi300_computation.py' in command and b'--worker' in command:
        role='numerical-worker'
    elif b'profile_csi300_timing.py' in command and b'--case-json' in command:
        role='profile-worker'
    elif b'pyspark' in command and 'python' in name:
        role='spark-python'
    affinity=set()
    if 'python' in name:
        for task in (path/'task').iterdir():
            try:affinity.add(tuple(sorted(os.sched_getaffinity(int(task.name)))))
            except ProcessLookupError:pass
    return dict(pid=pid,ppid=int(fields[1]),name=name,role=role,
                cpu_seconds=(int(fields[11])+int(fields[12]))/os.sysconf('SC_CLK_TCK'),
                major_faults=int(fields[9]),rss_kib=memory_value(status,'VmRSS'),
                peak_rss_kib=memory_value(status,'VmHWM'),swap_kib=memory_value(status,'VmSwap'),
                threads=int(status.get('Threads','0')),thread_affinity_sets=sorted(affinity))


def snapshot(root_pid,cpus):
    parents={}
    for p in Path('/proc').iterdir():
        if p.name.isdigit():
            try:
                if p.stat().st_uid==os.getuid():
                    raw=(p/'stat').read_text();fields=raw[raw.rindex(')')+2:].split()
                    parents[int(p.name)]=int(fields[1])
            except (OSError,ValueError):pass
    family={root_pid}
    while True:
        extended=family|{pid for pid,parent in parents.items() if parent in family}
        if extended==family:break
        family=extended
    processes=[]
    for pid in sorted(family):
        try:processes.append(collect_process(pid))
        except (OSError,ValueError):pass
    mem={line.split(':')[0]:int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()
         if line.split(':')[0] in {'MemTotal','MemAvailable','SwapTotal','SwapFree'}}
    swap={line.split()[0]:int(line.split()[1]) for line in Path('/proc/vmstat').read_text().splitlines()
          if line.split()[0] in {'pswpin','pswpout'}}
    frequencies={}
    for cpu in cpus:
        path=Path(f'/sys/devices/system/cpu/cpu{cpu}/cpufreq/scaling_cur_freq')
        try:frequencies[cpu]=int(path.read_text())
        except OSError:pass
    return dict(time=time.time(),monotonic=time.perf_counter(),processes=processes,host_memory_kib=mem,
                host_swap_pages=swap,scaling_frequency_khz=frequencies)


def main(root,interval,root_pid=None,cpus=None,stop_file=None):
    pid=root_pid if root_pid is not None else json.loads((root/'process.json').read_text())['pid']
    command=Path(f'/proc/{pid}/cmdline').read_bytes()
    if b'run_csi300_timing_comparison' not in command:
        raise ValueError('The recorded PID is not the timing runner')
    config=json.loads((root/'config.json').read_text())
    out=root/'reply_support';out.mkdir(exist_ok=True)
    os.sched_setaffinity(0,{0})
    metadata=dict(root_pid=pid,observer_pid=os.getpid(),interval_seconds=interval,
                  started_at=time.time(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope='Read-only diagnostic snapshots; timestamps record the actual resource coverage')
    (out/f'resource_observer_{pid}_config.json').write_text(json.dumps(metadata,indent=2)+'\n')
    count=0
    with (out/f'resources_{pid}.jsonl').open('a',buffering=1) as handle:
        while Path(f'/proc/{pid}').exists():
            if stop_file is not None and stop_file.exists():break
            try:
                if Path(f'/proc/{pid}/cmdline').read_bytes()!=command:break
            except OSError:break
            record=snapshot(pid,cpus or config['cpus'])
            record['phase']=json.loads((root/'progress.json').read_text())
            handle.write(json.dumps(record)+'\n');count+=1
            (out/'resource_observer_status.json').write_text(json.dumps(
                dict(root_pid=pid,observer_pid=os.getpid(),samples=count,last_sample=record['time'],running=True),indent=2)+'\n')
            time.sleep(interval)
    (out/'resource_observer_status.json').write_text(json.dumps(
        dict(root_pid=pid,observer_pid=os.getpid(),samples=count,finished_at=time.time(),running=False),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--interval',type=float,default=10.)
    parser.add_argument('--root-pid',type=int)
    parser.add_argument('--cpus',type=int,nargs='+')
    parser.add_argument('--stop-file',type=Path)
    args=parser.parse_args()
    main(args.input.resolve(),args.interval,args.root_pid,args.cpus,args.stop_file)
