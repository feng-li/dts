#!/usr/bin/env python3
"""Report measured timing components without mixing baseline definitions."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from statistics import median


COMPONENTS = ['startup','fft','optimization','sampler_preparation','compilation',
              'mcmc','covariance','aggregation','other']
LABELS = {'full':'Full-data MCMC','dc_parallel':'Spectral DC, G=20',
          'dc_serial':'Sequential DC, G=20','spark_g1':'Spark, G=1'}


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def timing_row(path):
    result=json.loads(path.read_text())
    monitor=json.loads(path.with_name('resource_monitor.json').read_text())
    if (not monitor.get('finished') or monitor.get('exit_code') != 0 or
        (monitor.get('interference') and monitor.get('policy','exclude')!='record')):
        raise ValueError(f'Timing was not accepted: {path}')
    worker=result['worker_wall_breakdown']
    row={key:result[key] for key in ['date','method','n','repetition','iterations','burn','groups','slots',
                                    'fit_seconds','startup_seconds','all_in_seconds']}
    row.update(startup=result['startup_seconds'],fft=result['fft_seconds'],
               optimization=result.get('global_map_seconds',0.)+worker['native_map_seconds']+worker['coordinate_map_seconds'],
               sampler_preparation=sum(worker[key] for key in
                   ['load_seconds','sampling_prepare_seconds','sampling_extract_seconds','write_seconds']),
               compilation=worker['loop_compile_seconds'],mcmc=worker['compiled_sampling_seconds'],
               covariance=worker['covariance_seconds'],aggregation=result['aggregation_seconds'])
    row['other']=result['all_in_seconds']-sum(row[key] for key in COMPONENTS if key != 'other')
    if row['other'] < -1e-4:
        raise ValueError('Timing components exceed total wall time')
    row['other']=max(0.,row['other'])
    row['interference_policy']=monitor.get('policy','exclude')
    row['competing_samples']=sum(bool(x['competing_compute_processes']) for x in monitor['checks'])
    row['monitor_samples']=len(monitor['checks'])
    row['max_competing_cores']=max((sum(p['cpu_cores'] for p in x['competing_compute_processes'])
                                    for x in monitor['checks']),default=0.)
    row['artifact']=str(path)
    return row


def summarize(rows):
    result=[]
    for date,method in sorted({(x['date'],x['method']) for x in rows}):
        selected=[x for x in rows if x['date']==date and x['method']==method]
        row=dict(date=date,method=method,n=selected[0]['n'],repetitions=len(selected),
                 iterations=selected[0]['iterations'],burn=selected[0]['burn'],
                 groups=selected[0]['groups'],cores=selected[0]['slots'])
        for key in ['fit_seconds','all_in_seconds',*COMPONENTS]:
            values=[x[key] for x in selected]
            row[key]=median(values)
            if key in ('fit_seconds','all_in_seconds'):
                row[key+'_min']=min(values);row[key+'_max']=max(values)
        result.append(row)
    return result


def comparisons(summary):
    result=[]
    for date in sorted({x['date'] for x in summary}):
        methods={x['method']:x for x in summary if x['date']==date}
        if 'full' not in methods or 'dc_parallel' not in methods:
            continue
        full,parallel=methods['full'],methods['dc_parallel']
        row=dict(date=date,n=full['n'],
                 full_to_parallel_fit=full['fit_seconds']/parallel['fit_seconds'],
                 full_to_parallel_all_in=full['all_in_seconds']/parallel['all_in_seconds'])
        for mode in ('dc_serial','spark_g1'):
            row[mode+'_to_parallel_fit']=(methods[mode]['fit_seconds']/parallel['fit_seconds']
                                         if mode in methods else None)
        row['parallel_efficiency_same_dc']=(row['dc_serial_to_parallel_fit']/parallel['cores']
                                             if row['dc_serial_to_parallel_fit'] is not None else None)
        result.append(row)
    return result


def sampling_budget_text(rows):
    """Label the measured budget, rejecting pooled runs with different budgets."""
    budgets={(x['iterations'],x['burn']) for x in rows}
    if len(budgets)!=1:
        raise ValueError('A timing report requires one common fitting iteration/burn-in budget')
    iterations,burn=next(iter(budgets))
    return (f'{iterations:,} iterations, {burn:,} burn-in; '
            'compiled fixed mixture MH; one chain per target.')


def make_plot(root, rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.size':11,'axes.labelsize':12,'xtick.labelsize':11,
                         'ytick.labelsize':11,'pdf.fonttype':42})
    dates=sorted({x['date'] for x in rows if x['method'] in ('full','dc_parallel')})
    fig,axes=plt.subplots(1,len(dates),figsize=(10,5.2),squeeze=False)
    colors=['#b8b8b8','#4477aa','#ee8866','#aaaa44','#ccbb44',
            '#228833','#66ccee','#aa3377','#dddddd']
    labels=['Startup','FFT/preparation','MAP/proposals','Sampler preparation',
            'Compilation','MCMC execution','Covariance','Consensus/diagnostics','Scheduling/other']
    chosen=[]
    for axis,date in zip(axes[0],dates):
        selected=[]
        for method in ('full','dc_parallel'):
            candidates=[x for x in rows if x['date']==date and x['method']==method]
            typical=median(x['all_in_seconds'] for x in candidates)
            selected.append(min(candidates,key=lambda x:abs(x['all_in_seconds']-typical)))
        bottom=np.zeros(2)
        for key,color,label in zip(COMPONENTS,colors,labels):
            values=np.array([x[key]/60 for x in selected])
            axis.bar([0,1],values,bottom=bottom,color=color,label=label,width=.6)
            bottom+=values
        axis.set_xticks([0,1],['Full-data\nMCMC','Spectral DC\nG=20'])
        axis.set_ylabel('Elapsed time (minutes)')
        axis.set_title(f'{selected[0]["n"]/1e6:.2f} million observations')
        axis.spines[['top','right']].set_visible(False)
        axis.set_ylim(0,max(bottom)*1.15)
        for i,value in enumerate(bottom):
            axis.text(i,value+max(bottom)*.02,f'{value:.2f}',ha='center',fontsize=11)
        chosen.extend(dict(date=x['date'],method=x['method'],repetition=x['repetition']) for x in selected)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,fontsize=10)
    fig.tight_layout(rect=(0,.22,1,1))
    fig.savefig(root/'timing_breakdown.pdf',bbox_inches='tight')
    fig.savefig(root/'timing_breakdown.png',dpi=160,bbox_inches='tight')
    plt.close(fig)
    return chosen


def main(root,partial=False):
    paths=list(root.glob('histories/*/cases/*/*/completed.json'))+list(root.glob('controls/cases/*/*/completed.json'))
    rows=[timing_row(path) for path in sorted(paths)]
    summary=summarize(rows)
    gains=comparisons(summary)
    write_csv(root/'timing_cases.csv',rows)
    write_csv(root/'timing_summary.csv',summary)
    write_csv(root/'speedups.csv',gains)
    profile=[]
    for path in sorted(root.glob('sampling_only/cases/*/completed.json')):
        x=json.loads(path.read_text())
        profile.append({key:x[key] for key in ['date','mode','n','iterations','burn','repetition',
                                               'sampling_wall_seconds','max_compile_seconds',
                                               'all_reference_replays_equal']})
    write_csv(root/'sampling_only_cases.csv',profile)
    short=[]
    for length in sorted({x['iterations'] for x in profile}):
        selected=[x for x in profile if x['iterations']==length]
        full=[x['sampling_wall_seconds'] for x in selected if x['mode']=='full']
        parallel=[x['sampling_wall_seconds'] for x in selected if x['mode']=='dc_parallel']
        if full and parallel:
            short.append(dict(iterations=length,full_seconds=median(full),parallel_seconds=median(parallel),
                              full_min=min(full),full_max=max(full),parallel_min=min(parallel),parallel_max=max(parallel),
                              speedup=median(full)/median(parallel)))
    write_csv(root/'sampling_only_summary.csv',short)
    report=['CSI 300 production-length timing comparison',
            sampling_budget_text(rows),
            'Shared host monitored for competing computation; no exclusive reservation.',
            'Primary speed-up compares direct full-data MCMC with parallel G=20 DC.',
            'Component medians are descriptive; plotted bars use one representative repetition and reconcile exactly.',
            'Input liquidity construction, clock calibration and forecasts are outside fitting scope.','']
    for x in summary:
        report.append(f'{x["date"]} {LABELS[x["method"]]}: fit {x["fit_seconds"]:.3f}s '
                      f'[{x["fit_seconds_min"]:.3f}, {x["fit_seconds_max"]:.3f}]; '
                      f'including startup {x["all_in_seconds"]:.3f}s; repetitions={x["repetitions"]}')
    report.extend(['','Speedups:',json.dumps(gains,indent=2),'','Compiled MCMC only:',json.dumps(short,indent=2)])
    (root/'REPORT.txt').write_text('\n'.join(report)+'\n')
    table=[r'\begin{tabular}{llrrrr}',r'\toprule',
           r'$N$ & Method & Cores & Fit (s) & Total (s) & Speed-up\\',r'\midrule']
    gain_by_date={x['date']:x for x in gains}
    for row in summary:
        if row['method'] not in ('full','dc_parallel'):
            continue
        gain=1. if row['method']=='full' else gain_by_date[row['date']]['full_to_parallel_all_in']
        table.append(f'{row["n"]:,} & {LABELS[row["method"]]} & {row["cores"]} & '
                     f'{row["fit_seconds"]:.1f} & {row["all_in_seconds"]:.1f} & {gain:.2f}'+r'\\')
    table.extend([r'\bottomrule',r'\end{tabular}'])
    (root/'timing_table.tex').write_text('\n'.join(table)+'\n')
    if not partial:
        phases=json.loads((root/'commands.json').read_text())
        expected_fit=sum(x['expected_cases'] for x in phases if x['kind']=='fit')
        expected_profile=sum(x['expected_cases'] for x in phases if x['kind']=='profile')
        if len(rows)!=expected_fit or len(profile)!=expected_profile:
            raise ValueError(f'The complete procedure requires {expected_fit} fitting and {expected_profile} profiling cases')
        chosen=make_plot(root,rows)
        (root/'plot_cases.json').write_text(json.dumps(chosen,indent=2)+'\n')
    print('\n'.join(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--allow-partial',action='store_true')
    args=parser.parse_args()
    main(args.input,args.allow_partial)
