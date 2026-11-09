#!/usr/bin/env python3
"""Publish measured paper-budget timings after the complete study passes audit."""
from __future__ import annotations

import argparse
import csv
import difflib
import hashlib
import json
from pathlib import Path
import re
from statistics import median
import subprocess
import time


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as handle:
        return list(csv.DictReader(handle))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paragraph(text,prefix,value):
    pattern=re.compile(r'^'+re.escape(prefix)+r'[^\n]*$',re.M)
    text,count=pattern.subn(lambda match:value,text)
    if count!=1:raise ValueError(f'The expected paragraph changed: {prefix}')
    return text


def profile_frequencies(root):
    frames=[json.loads(line) for path in sorted((root/'reply_support').glob('resources_*.jsonl'))
            for line in path.read_text().splitlines() if line.strip()]
    result=[]
    for path in sorted((root/'sampling_only/cases').glob('*i65000*/completed.json')):
        d=read(path);start=read(path.with_name('process.json'))['started_at'];end=path.stat().st_mtime
        observations=[f for f in frames if start<=f['time']<=end]
        medians=[];count=0
        for w in d['workers']:
            values=[f['scaling_frequency_khz'][str(w['cpu'])]/1000 for f in observations
                    if w['sampling_started_at']<=f['monotonic']<=w['sampling_finished_at']
                    and str(w['cpu']) in f['scaling_frequency_khz']]
            if len(values)<3:raise ValueError(f'Insufficient clock coverage: {path}/{w["group"]}')
            count+=len(values);medians.append(median(values))
        result.append(dict(case=path.parent.name,mode=d['mode'],sampling_wall_seconds=d['sampling_wall_seconds'],
                           sampled_workers=len(medians),frequency_samples=count,
                           median_worker_frequency_mhz=median(medians)))
    if len(result)!=6:raise ValueError('Six paper-budget MCMC-only profiles are required')
    with (root/'reply_support/sampling_only_frequency_observations.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(result[0]));writer.writeheader();writer.writerows(result)
    return result


def longest_history_template(template):
    """Restrict the presentation as well as the measurements to the longest history."""
    old=next(x for x in template.splitlines() if x.startswith('We compare full-data'))
    template=paragraph(template,'We compare full-data',
        'We compare full-data and spectral divide-and-conquer MCMC using 24,300,012 training observations preceding the forecast date 24 July 2026. Both methods use'+old.split('Both methods use',1)[1])
    label=r'\label{tab:liquidity-production-timing}'
    begin=template.index(label);start=template.index(r'\multicolumn{5}',begin)
    end=template.index(r'\bottomrule',start)
    old_rows=[line for line in template[start:end].splitlines()
              if line.startswith(('Full-data ARTFIMA','Spectral DC ARTFIMA'))]
    selected=[next(line for line in reversed(old_rows) if line.startswith('Full-data')),
              next(line for line in old_rows if 'Spark $G=1$' in line),
              next(line for line in old_rows if 'sequential' in line),
              next(line for line in reversed(old_rows) if 'parallel' in line)]
    template=template[:start]+r'\multicolumn{5}{@{}l}{24,300,012 observations}\\[2pt]'+'\n'+'\n'.join(selected)+'\n'+template[end:]
    label=r'\label{tab:liquidity-timing-components}'
    begin=template.index(r'\begin{tabular}',template.index(label));end=template.index(r'\bottomrule',begin)
    names=['Startup','FFT and frequency preparation','MAP and proposal fitting',
           'Compiled MCMC execution','Covariance regularisation']
    block=r'\begin{tabular}{@{}lrr@{}}\toprule'+'\n'+r'Component & Full data & DC\\\midrule'+'\n'
    block+='\n'.join(name+r' & --- & ---\\' for name in names)+'\n'
    template=template[:begin]+block+template[end:]
    begin=template.index(r'\subsection*{Earlier approximation-quality results}')
    end=template.index(r'\begingroup\small',begin)
    template=template[:begin]+template[end:]
    template=template.replace('\\clearpage\n','')
    return template


def render_tex(template,primary,profile,diagnostic,frequencies):
    if {(int(x['iterations']),int(x['burn'])) for x in primary}!={(65000,5000)}:
        raise ValueError('Only measured 65,000-iteration/5,000-burn-in results may update this TeX')
    dates=sorted({x['date'] for x in primary})
    single=dates==['20260724']
    if single:template=longest_history_template(template)
    elif dates!=['20211013','20260724']:raise ValueError('Unexpected training histories')
    first=dates[0]
    n_label=f"{int(primary[0]['n'])/1e6:.2f}"
    data={(x['date'],x['method']):x for x in primary}
    get=lambda date,mode,key:float(data[(date,mode)][key])
    text=template.replace('35,000','65,000')
    label=r'\label{tab:liquidity-production-timing}'
    begin=text.index(label);end=text.index(r'\end{table}',begin)
    block=text[begin:end];lines=[];date=None;checked=0
    for line in block.splitlines():
        if '10,238,352 observations' in line:date='20211013'
        if '24,300,012 observations' in line:date='20260724'
        if line.startswith(('Full-data ARTFIMA','Spectral DC ARTFIMA')):
            name=line.split('&')[0].strip()
            mode='full' if name.startswith('Full-data') else 'spark_g1' if 'Spark' in name else 'dc_serial' if 'sequential' in name else 'dc_parallel'
            d=data[(date,mode)]
            line=' & '.join([name,d['groups'],d['cores'],f"{float(d['fit_seconds']):.1f}",
                             f"{float(d['all_in_seconds']):.1f}"])+r'\\'+('[4pt]' if '[4pt]' in line else '')
            checked+=1
        lines.append(line)
    if checked!=(4 if single else 6):raise ValueError('Primary table changed')
    text=text[:begin]+'\n'.join(lines)+'\n'+text[end:]
    label=r'\label{tab:liquidity-timing-components}'
    begin=text.index(label);end=text.index(r'\end{table}',begin)
    keys={'Startup':'startup','FFT and frequency preparation':'fft','MAP and proposal fitting':'optimization',
          'Compiled MCMC execution':'mcmc','Covariance regularisation':'covariance'}
    lines=[];checked=0
    for line in text[begin:end].splitlines():
        name=line.split('&')[0].strip()
        if name in keys:
            values=['---' if mode=='full' and keys[name]=='covariance' else f'{get(date,mode,keys[name]):.2f}'
                    for date in dates for mode in ('full','dc_parallel')]
            line=' & '.join([name,*values])+r'\\';checked+=1
        lines.append(line)
    if checked!=5:raise ValueError('Component table changed')
    text=text[:begin]+'\n'.join(lines)+'\n'+text[end:]
    fit_gain=[get(d,'full','fit_seconds')/get(d,'dc_parallel','fit_seconds') for d in dates]
    total_gain=[get(d,'full','all_in_seconds')/get(d,'dc_parallel','all_in_seconds') for d in dates]
    serial_gain=get(first,'dc_serial','fit_seconds')/get(first,'dc_parallel','fit_seconds')
    comparison=(f'Relative to native full-data MCMC, parallel Spectral DC achieves speed-ups of ${fit_gain[0]:.2f}$ excluding startup and ${total_gain[0]:.2f}$ including startup. Sequential-to-parallel Spectral DC gives a fitting speed-up of ${serial_gain:.2f}$, corresponding to ${100*serial_gain/20:.1f}\\%$ parallel efficiency on 20 cores. This latter comparison has only one sequential repetition.' if single else
        f'Relative to native full-data MCMC, parallel Spectral DC achieves speed-ups of ${fit_gain[0]:.2f}$ and ${fit_gain[1]:.2f}$ excluding startup, or ${total_gain[0]:.2f}$ and ${total_gain[1]:.2f}$ including startup, for the two histories. On the smaller history, sequential-to-parallel Spectral DC gives a fitting speed-up of ${serial_gain:.2f}$, corresponding to ${100*serial_gain/20:.1f}\\%$ parallel efficiency on 20 cores. This latter comparison has only one sequential repetition.')
    text=paragraph(text,'Relative to native full-data MCMC,',
        comparison)
    short={x['method']:x for x in diagnostic['fit_comparisons'] if x['scope']=='short'}
    full,dc=(float(short[x]['total_seconds']) for x in ('full','dc_parallel'))
    old=next(x for x in text.splitlines() if x.startswith('Fitting includes FFT'))
    text=paragraph(text,'Fitting includes FFT',old.split('In additional matched')[0]+
        f'In additional matched 2,000-iteration runs with 500 burn-in iterations, median startup-inclusive times were {full:.1f} seconds for full-data MCMC and {dc:.1f} seconds for parallel DC, a speed-up of ${full/dc:.2f}$; startup and covariance costs are less well amortised by short chains.')
    percentages=[100*get(d,'dc_parallel','fft')/get(d,'dc_parallel','fit_seconds') for d in dates]
    pct=(f'The FFT/preparation cost is about ${percentages[0]:.0f}\\%$ of parallel fitting time for this history.' if single else
         f'The FFT/preparation cost is about ${percentages[0]:.0f}\\%$ and ${percentages[1]:.0f}\\%$ of parallel fitting time for the two histories.')
    text,count=re.subn(r'The FFT/preparation cost is about [^\n]*?for the two histories\.',lambda m:pct,text)
    if count!=1:raise ValueError('FFT percentage sentence changed')
    p=next(x for x in profile if int(x['iterations'])==65000)
    full,dc=float(p['full_seconds']),float(p['parallel_seconds'])
    text=paragraph(text,'Three profiling repetitions per method',
        f'Three fresh timing executions per method on the {n_label}-million-observation history replay the same fitted proposals and random streams. The actual loop shapes are compiled before timing, parallel workers are synchronised, and numerical completion is awaited. The timer excludes FFT, optimisation, proposal preparation, compilation, covariance calculation and consensus. With 65,000 iterations, median MCMC-only times are {full:.2f} seconds for the full-data chain and {dc:.2f} seconds for all 20 parallel shard chains, giving ${full:.2f}/{dc:.2f}\\simeq{full/dc:.2f}$ times speed-up. These profiling executions are separate from the fitting repetitions in Table~\\ref{{tab:liquidity-timing-components}}.')
    freq=[median(x['median_worker_frequency_mhz'] for x in frequencies if x['mode']==mode)/1000
          for mode in ('full','dc_parallel')]
    layouts=diagnostic['socket_layouts']
    one=median(x['sampling_wall_seconds'] for x in layouts if x['layout']=='one_socket')
    both=median(x['sampling_wall_seconds'] for x in layouts if x['layout']=='balanced_sockets')
    if len(layouts)!=6:raise ValueError('All six socket controls must be complete')
    text=paragraph(text,'During these exact profiles,',
        f'During these exact profiles, sampled kernel-reported CPU frequencies were approximately {freq[0]:.2f}~GHz for the full-data chain and {freq[1]:.2f}~GHz per parallel worker. A simple frequency-proportional throughput calculation gives $20({freq[1]:.2f}/{freq[0]:.2f})\\simeq{20*freq[1]/freq[0]:.2f}$. This is illustrative, not an isolated causal measurement. In three additional paired controls, placing ten workers on each socket changed median MCMC-only time from {one:.2f} to {both:.2f} seconds, a ${100*(1-both/one):.1f}\\%$ reduction, with bitwise-identical saved draws. These placement controls do not measure complete fitting time; the primary comparison retains the one-socket layout. The measurements do not separately identify cache, bandwidth and power contributions.')
    critical=diagnostic['parallel_cases']
    limits=lambda key:(min(x[key] for x in critical),max(x[key] for x in critical))
    f0,f1=limits('nonlocal_upper_bound_fit_percent');t0,t1=limits('nonlocal_upper_bound_total_percent')
    s0,s1=limits('tail_after_median_finish_seconds');q0,q1=limits('tail_percent_of_fit')
    text=paragraph(text,'There is no inter-shard communication',
        f'There is no inter-shard communication during MCMC iterations. Assigning all elapsed fitting time outside the terminal worker\'s local MAP fitting, compilation, MCMC and covariance calculations to communication gives a conservative upper bound of ${f0:.1f}$--${f1:.1f}\\%$. This remainder also includes FFT arithmetic and input preparation, and is not a direct communication measurement. Including startup gives a bound of ${t0:.1f}$--${t1:.1f}\\%$. Consensus waits for every shard, but the last worker finished {s0:.1f}--{s1:.1f} seconds after the median worker, or ${q0:.1f}$--${q1:.1f}\\%$ of fitting time. These measurements concern a shared-memory host, not a networked cluster.')
    memory=diagnostic['memory']
    if len(memory)!=sum(int(x['repetitions']) for x in primary):
        raise ValueError('Resource observations are required for all fitting cases')
    if any(x['max_worker_swap_kib'] or x['host_swapin_pages_delta'] or x['host_swapout_pages_delta'] for x in memory):
        raise ValueError('Swap activity requires review before publication')
    rss=max(x['max_worker_rss_gib'] for x in memory);available=min(x['min_host_available_gib'] for x in memory)
    old=next(x for x in text.splitlines() if x.startswith('The largest observed numerical-worker'))
    text=paragraph(text,'The largest observed numerical-worker',
        f'The largest observed numerical-worker resident memory was {rss:.2f}~GiB, and at least {available:.0f}~GiB of host memory remained available in the sampled observations. Observed worker swap usage and changes in host swap-in/out counters were zero. '+old[old.index('In Figure~10'):])
    legacy=lambda s:s.split(r'\label{tab:liquidity-earlier-quality}',1)[1].split(r'\end{table}',1)[0]
    if not single and legacy(text)!=legacy(template):raise ValueError('Earlier quality table was altered')
    return text


def main(root):
    root=root.resolve();config=read(root/'config.json');publication=read(root/'paper_update/config.json')
    if (config['iterations'],config['burn'])!=(65000,5000):raise ValueError('Paper-budget run required')
    if read(root/'reply_support/completion_audit.json')['status']!='passed':raise ValueError('Numerical audit has not passed')
    frequencies=profile_frequencies(root)
    target=Path(publication['tex']);template=(root/'paper_update/before.tex').read_text()
    result=render_tex(template,rows(root/'timing_summary.csv'),rows(root/'sampling_only_summary.csv'),
                      read(root/'reply_support/investigation_results.json'),frequencies)
    if config.get('interference_policy')=='record':
        result=paragraph(result,'The experiments use one shared-memory server',
            'The experiments use one shared-memory server with 52 physical cores across two sockets. The full-data chain and each shard chain are pinned to one physical core, with BLAS, OpenMP and Numba thread counts set to one and actual thread affinities checked. The primary parallel runs use 20 physical cores on one socket through Spark \\texttt{local[20]}. These are monitored shared-host timings without an exclusive reservation or a quiet-host gate. Background-load observations are retained and reported; brief CPU activity does not restart the chains. The earlier interrupted attempts under the stricter screening rule are archived and excluded.')
    result=re.sub(r'^% Timing evidence:.*$',lambda m:f'% Timing evidence: {root}/',result,flags=re.M)
    proposed=root/'paper_update/proposed.tex';proposed.write_text(result)
    (root/'paper_update/update.diff').write_text(''.join(difflib.unified_diff(template.splitlines(True),result.splitlines(True),fromfile=str(target),tofile=str(proposed))))
    if target.is_symlink() or sha(target)!=publication['tex_sha256']:
        if target.read_text()!=result:raise ValueError('The TeX changed during the run; the proposed update is saved for review')
    else:
        marker=root/'paper_update/artifact_operation_started.json'
        if not marker.exists():
            if config.get('artifact_marker'):
                subprocess.run(['node',config['artifact_marker'],'--operation-kind','edit','--expected-output-count','1','--output-format','pdf'],check=True)
            marker.write_text(json.dumps(dict(started_at=time.time()))+'\n')
        mode=target.stat().st_mode
        temporary=target.with_name(target.name+'.timing.partial');temporary.write_text(result)
        temporary.chmod(mode);temporary.replace(target)
    with (root/'paper_update/compile.stdout').open('w') as log:
        subprocess.run(['latexmk','-xelatex','-interaction=nonstopmode','-halt-on-error','-silent',target.name],cwd=target.parent,stdout=log,stderr=subprocess.STDOUT,check=True)
    pdf=target.with_suffix('.pdf');qa=root/'paper_update/qa';qa.mkdir(exist_ok=True)
    subprocess.run(['pdftoppm','-r','110','-png',str(pdf),str(qa/'page')],check=True)
    import pdfplumber
    with pdfplumber.open(pdf) as document:
        contents='\n'.join(page.extract_text() or '' for page in document.pages)
        if '65,000' not in contents or '35,000' in contents:raise ValueError('PDF budget label is inconsistent')
        pages=len(document.pages)
    (root/'paper_update/status.json').write_text(json.dumps(dict(status='compiled_pending_visual_review',
        tex=str(target),pdf=str(pdf),pages=pages,tex_sha256=sha(target),pdf_sha256=sha(pdf),completed_at=time.time()),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--input',type=Path,required=True)
    main(parser.parse_args().input)
