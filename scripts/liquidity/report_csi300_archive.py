#!/usr/bin/env python3
"""Audit saved daily index panels and plot available-stock diagnostics; no replay."""
import argparse
import json
import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/csi300-matplotlib")
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--out',type=Path,default=Path('output/pdf/csi300_archive_results.pdf'))
    args=p.parse_args();r=args.root;report=r/'archive_report';report.mkdir(exist_ok=True)
    status=pd.read_csv(r/'batch_status.csv',dtype={'date':str})
    failures=[]
    for row in status[status.status!='built'].itertuples():
        text=Path(row.log).read_text()
        reason='no admissible matching weights' if 'No admissible matching weight snapshot' in text else 'other'
        failures.append(dict(date=row.date,reason=reason))
    pd.DataFrame(failures,columns=['date','reason']).to_csv(report/'failures.csv',index=False)
    checks=Counter();daily=[];clocks=[]
    cols=['date','session','t_ms','size_spec','n_valid','weight_coverage','sh_weight_coverage','sz_weight_coverage',
          'unknown_depth_weight','technical_invalid_weight','shortage_weight','nontrading_weight',
          'liq_available_vw_bps','liq_available_buy_bps','liq_available_sell_bps','sh_available_contribution',
          'sz_available_contribution','liq_raw_vw','liq_stress','is_release_eligible',
          'weight_imputed','weight_imputation_method','weight_lookahead_days','weight_dates_assumed']
    files=sorted((r/'index').glob('*.parquet'))
    checks['status_file_mismatches']=len(set(status.loc[status.status=='built','date']) ^ {f.stem for f in files})
    for f in tqdm(files,desc='Audit saved index days'):
        a=pd.read_parquet(f,columns=cols)
        checks['date_label_errors']+=int((a.date.astype(str)!=f.stem).sum())
        checks['release_rows']+=int(a.is_release_eligible.sum())
        checks['weight_imputed_days']+=int(a.weight_imputed.any())
        checks['weight_lookahead_days_used']+=int((a.weight_lookahead_days>0).any())
        checks['assumed_weight_date_days']+=int(a.weight_dates_assumed.any())
        w=pd.read_parquet(r/'audits'/f'{f.stem}_membership.parquet')
        totals=w.groupby('exchange').weight.sum()
        valid=a.liq_available_vw_bps.notna()
        gate=(a.weight_coverage>=.95-1e-12)&(a.n_valid>=250)
        for ex in ['SH','SZ']:
            if totals.get(ex,0): gate &= a[ex.lower()+'_weight_coverage']/totals[ex]>=.95-1e-12
        checks['gate_mismatches']+=int((valid!=gate).sum())
        checks['negative_spreads']+=int((a.liq_available_vw_bps<0).sum())
        checks['nonfinite_published']+=int((valid & ~np.isfinite(a.liq_available_vw_bps)).sum())
        checks['directional_identity_errors']+=int((abs(a.liq_available_vw_bps-a.liq_available_buy_bps-a.liq_available_sell_bps)>1e-7).sum())
        checks['exchange_identity_errors']+=int((abs(a.liq_available_vw_bps-a.sh_available_contribution-a.sz_available_contribution)>1e-7).sum())
        checks['coverage_decomposition_errors']+=int((abs(a.weight_coverage+a.unknown_depth_weight+a.technical_invalid_weight+a.shortage_weight+a.nontrading_weight-1)>1e-8).sum())
        checks['duplicate_grid_rows']+=int(a.duplicated(['session','size_spec','t_ms']).sum())
        checks['rows']+=len(a);checks['full_verified_rows']+=int(a.liq_raw_vw.notna().sum());checks['stress_rows']+=int(a.liq_stress.notna().sum())
        for spec,g in a.groupby('size_spec'):
            v=g.liq_available_vw_bps.dropna()
            checks['unexpected_daily_size_rows']+=int(len(g)!=14220)
            daily.append(dict(date=f.stem,size_spec=spec,rows=len(g),valid=len(v),mean=v.mean(),median=v.median(),p95=v.quantile(.95),maximum=v.max(),coverage=g.weight_coverage.mean(),unknown=g.unknown_depth_weight.mean(),technical=g.technical_invalid_weight.mean()))
        a['minute']=(a.t_ms//60000).astype(int)
        c=a.groupby(['session','size_spec','minute']).liq_available_vw_bps.agg(['mean','count']).reset_index()
        c['date']=f.stem;clocks.append(c)
    d=pd.DataFrame(daily);d.to_csv(report/'daily.csv',index=False)
    c=pd.concat(clocks,ignore_index=True);c.to_parquet(report/'daily_clock.parquet',index=False)
    summary=d.groupby('size_spec').agg(days=('date','size'),rows=('rows','sum'),valid=('valid','sum'),days_with_values=('valid',lambda x:int((x>0).sum())),days_95pct=('valid',lambda x:int((x>=.95*14220).sum())))
    summary['valid_pct_built']=100*summary.valid/summary.rows
    summary['valid_pct_all_dates']=100*summary.valid/(len(status)*14220)
    summary.to_csv(report/'summary.csv')
    saved=json.loads((r/'diagnostics.json').read_text())
    checks['saved_row_total_mismatch']=int(saved['index_rows']!=int(d.rows.sum()))
    checks['saved_valid_total_mismatch']=int(saved['valid_available_rows']!=int(d.valid.sum()))
    result=dict(status_counts=status.status.value_counts().to_dict(),failure_reasons=dict(Counter(f['reason'] for f in failures)),checks=dict(checks),scope='All saved daily index panels checked. Raw events and individual stock panels not re-audited.')
    (report/'checks.json').write_text(json.dumps(result,indent=2))
    print(summary.to_string());print(json.dumps(result,indent=2))
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    sizes=['notional_1e+06','notional_5e+06','notional_1e+07'];labels=['RMB 1m','RMB 5m','RMB 10m'];colors=['#167c80','#ce7930','#765ca2']
    dates=pd.to_datetime(status.date,format='%Y%m%d');d['dt']=pd.to_datetime(d.date,format='%Y%m%d')
    def finish(pdf,fig,title,page):
        fig.suptitle(title,fontsize=17,x=.08,ha='left')
        fig.text(.08,.025,'CSI300 reconstructed-book diagnostic | Estimated weights allowed | No lunch interpolation',fontsize=9,color='#555555')
        fig.text(.94,.025,str(page),ha='right');fig.tight_layout(rect=[.04,.055,.98,.94]);pdf.savefig(fig);plt.close(fig)
    with PdfPages(args.out) as pdf:
        fig,axs=plt.subplots(2,1,figsize=(11.7,8.3),gridspec_kw={'height_ratios':[1,2]})
        axs[0].axis('off')
        lines=[f'{len(status):,} archive dates | {len(files):,} built | {len(failures):,} failed',
               'Failed dates: '+str(result['failure_reasons']),
               f"Full verified index: {checks['full_verified_rows']:,} rows; calibrated stress: {checks['stress_rows']:,} rows.",
               'A built date can contain no usable index values. Missing values are retained.']
        axs[0].text(.01,.95,'\n\n'.join(lines),va='top',transform=axs[0].transAxes)
        for spec,label,color in zip(sizes,labels,colors):
            g=d[d.size_spec==spec].set_index('dt').reindex(dates)
            axs[1].plot(dates,100*g.valid/g.rows,label=label,color=color,lw=.7)
        axs[1].set(ylabel='Valid seconds within built day (%)',ylim=(-3,103));axs[1].legend(ncol=3)
        finish(pdf,fig,'1  Completion and usable observations',1)
        fig,axs=plt.subplots(3,1,figsize=(11.7,8.3),sharex=True)
        for ax,spec,label,color in zip(axs,sizes,labels,colors):
            g=d[d.size_spec==spec].set_index('dt').reindex(dates)
            ax.plot(dates,g['mean'].where(g.valid>=.95*14220),color=color,lw=.8)
            ax.set_ylabel(label+'\nmean spread (bps)')
        finish(pdf,fig,'2  Daily costs - only days with >=95% valid seconds',2)
        fig,axs=plt.subplots(2,1,figsize=(11.7,8.3),sharex=True)
        for spec,label,color in zip(sizes,labels,colors):
            g=c[c.size_spec==spec].groupby(['session','minute']).agg(mean=('mean','mean'),n=('mean','count')).reset_index()
            for session,h in g.groupby('session'):
                x=h.minute.to_numpy()-(90 if session=='pm' else 0)
                axs[0].plot(x,h['mean'],color=color,label=label if session=='am' else None)
                axs[1].plot(x,h.n,color=color)
        for ax in axs:
            ax.axvline(690,color='grey',ls=':',lw=.8)
            ax.set_xticks([570,630,690,750,807],['09:30','10:30','11:30 / 13:00','14:00','14:57'])
        axs[0].set_ylabel('Mean spread (bps)');axs[0].legend(ncol=3)
        axs[1].set_ylabel('Contributing days');axs[1].set_xlabel('Trading clock; lunch removed from display only')
        finish(pdf,fig,'3  Intraday pattern - equal-weight daily minute means',3)
        fig,axs=plt.subplots(2,1,figsize=(11.7,8.3))
        yearly=[]
        for spec,label,color in zip(sizes,labels,colors):
            g=d[d.size_spec==spec].groupby(d.dt.dt.year).agg(valid=('valid','sum'))
            denom=status.groupby(status.date.str[:4]).size();g['pct']=[100*v/(denom[str(y)]*14220) for y,v in g.valid.items()]
            axs[0].plot(g.index,g.pct,'o-',label=label,color=color)
            yearly.extend(dict(year=int(y),size_spec=spec,valid_pct_all_dates=v) for y,v in g.pct.items())
        axs[0].set(ylabel='Valid / all scheduled seconds (%)',ylim=(-3,103));axs[0].legend(ncol=3)
        for spec,label,color in zip(sizes,labels,colors):
            g=d[d.size_spec==spec].groupby(d.dt.dt.year)[['technical','unknown']].mean()
            axs[1].plot(g.index,100*g.unknown,'o-',color=color,label=label+' unknown depth')
        g=d[d.size_spec==sizes[0]].groupby(d.dt.dt.year).technical.mean()
        axs[1].plot(g.index,100*g,'s--',color='#333333',label='Technical invalid (RMB 1m)')
        axs[1].set_ylabel('Mean unavailable weight on built days (%)');axs[1].legend(ncol=2,fontsize=9)
        finish(pdf,fig,'4  Coverage and missingness by year (2026 is partial)',4)
        pd.DataFrame(yearly).to_csv(report/'yearly.csv',index=False)
        fig,ax=plt.subplots(figsize=(11.7,8.3));ax.axis('off')
        lines=[f"Period: {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}; 1-second sampling.",
               'Target: weighted cash-execution full spread; larger values mean higher liquidity costs.',
               'RMB 1m / 5m / 10m are per-stock trade sizes, not total index portfolio orders.',
               'Each day: 7,200 morning + 7,020 afternoon seconds, separately for each size.',
               'Available index requires >=95% total weight, >=250 stocks, and >=95% within each exchange.',
               'Available-stock weights are renormalized; constituent availability changes over time.',
               f"Imputed weights: {checks['weight_imputed_days']:,} days; future snapshot used: {checks['weight_lookahead_days_used']:,} days.",
               f"Assumed weight effective dates: {checks['assumed_weight_date_days']:,} days; release-eligible rows: {checks['release_rows']:,}.",
               'Future-derived weights are retrospective approximations, not a real-time investable history.',
               'Missing and quarantined costs remain missing. No lunch, overnight or missing-value interpolation.',
               'Audit scope: all saved market index panels; raw events / individual books not re-audited.',
               'Gate / negative / nonfinite / duplicate / decomposition errors: '+
               ' / '.join(str(checks[k]) for k in ['gate_mismatches','negative_spreads','nonfinite_published','duplicate_grid_rows','coverage_decomposition_errors']),
               'Buy-sell / exchange identity / saved row and valid-count mismatches: '+
               ' / '.join(str(checks[k]) for k in ['directional_identity_errors','exchange_identity_errors','saved_row_total_mismatch','saved_valid_total_mismatch']),
               'Differences across sizes and years can reflect changing sample coverage as well as liquidity.']
        ax.text(.01,.98,'\n\n'.join(lines),va='top',fontsize=10,transform=ax.transAxes)
        finish(pdf,fig,'5  Construction, provenance and audit scope',5)
    print(args.out)

if __name__=='__main__':main()
