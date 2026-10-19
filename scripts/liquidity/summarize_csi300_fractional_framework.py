#!/usr/bin/env python3
"""Summarize the controlled fractional-change framework, with paired intervals."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd


def main():
    p=argparse.ArgumentParser(description=__doc__)
    base=Path.home()/'running/dts/artifacts/liquidity'
    p.add_argument('--candidate',type=Path,default=base/'csi300_fractional_changes_clean_probe')
    p.add_argument('--reference',type=Path,default=base/'csi300_forecast_comparison_v1')
    p.add_argument('--out',type=Path,default=base/'csi300_fractional_framework_summary')
    p.add_argument('--dc-check',type=Path,default=base/'csi300_fractional_changes_dc_long/2026Q3')
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    c=pd.read_parquet(args.candidate/'forecasts.parquet');r=pd.read_parquet(args.reference/'forecasts.parquet')
    a=pd.concat([c,r],ignore_index=True);keys=['date','phase','elapsed','horizon']
    if not (a.groupby(keys).model.nunique()==7).all() or a.duplicated(keys+['model']).any():raise ValueError('Unmatched origins')
    if not (a.groupby(keys).actual.nunique()==1).all():raise ValueError('Targets differ')
    a['bias']=a.prediction-a.actual
    scores=a.groupby(['model','horizon']).agg(n=('ae','size'),days=('date','nunique'),mae=('ae','mean'),mse=('se','mean'),crps=('crps','mean'),coverage95=('cover95','mean'),bias=('bias','mean')).reset_index()
    scores['rmse']=np.sqrt(scores.pop('mse'));scores.to_csv(args.out/'scores.csv',index=False)
    paired=[];rng=np.random.default_rng(20261008)
    for h in [60,300]:
        g=a[a.horizon==h]
        for metric in ['ae','se','crps']:
            wide=g.pivot(index=keys,columns='model',values=metric)
            for reference in ['MA1Changes','LastValue','ARIMA011','ARTFIMA10']:
                d=(wide['FractionalChanges']-wide[reference]).groupby(level='date').agg(['sum','count'])
                values=d.to_numpy();n=len(d);boot=[]
                for _ in range(2000):
                    starts=rng.integers(n,size=(n+4)//5);ix=((starts[:,None]+np.arange(5))%n).ravel()[:n]
                    v=values[ix].sum(axis=0);boot.append(v[0]/v[1])
                paired.append(dict(horizon=h,metric=metric,reference=reference,difference=float(d['sum'].sum()/d['count'].sum()),
                                   lower=float(np.quantile(boot,.025)),upper=float(np.quantile(boot,.975))))
    paired=pd.DataFrame(paired);paired.to_csv(args.out/'paired.csv',index=False)
    fits=[]
    for path in sorted(args.candidate.glob('*/fit.json')):
        f=json.loads(path.read_text());theta,ll,ls,d=f['theta']
        if f['n']%2 or f['newton_distance']>.01 or min(f['hessian_eigenvalues'])<=0:raise ValueError(f'Fit unresolved: {path}')
        fits.append(dict(fold=path.parent.name,theta=theta,lambda_=np.exp(ll),d=d,n=f['n'],newton_distance=f['newton_distance'],
                         boundary_squared_share=f['change_audit']['boundary_squared_share']))
    fits=pd.DataFrame(fits);fits.to_csv(args.out/'fit_summary.csv',index=False)
    # The input shape and frozen old calendar provide an independent target audit.
    raw=np.load(args.candidate.parent/'csi300_dc_1m/spread.npy',mmap_mode='r')
    dates=json.loads((args.candidate.parent/'csi300_dc_1m/input_manifest.json').read_text())['dates'];lookup={d:i for i,d in enumerate(dates)}
    for row in c.drop_duplicates(keys).itertuples():
        end=row.elapsed+(0 if row.phase=='am' else 7200)
        actual=raw[lookup[row.date],end:end+row.horizon].mean()
        if not np.isclose(actual,row.actual,rtol=0,atol=1e-12):raise ValueError('Source target mismatch')
    indexed=scores.set_index(['model','horizon']);macros=[]
    for prefix,model in [('Clean','FractionalChanges'),('Null','MA1Changes'),('Naive','LastValue'),('Arima','ARIMA011'),('Old','ARTFIMA10')]:
        for word,h in [('One',60),('Five',300)]:
            row=indexed.loc[(model,h)]
            for tag,column in [('MAE','mae'),('RMSE','rmse'),('CRPS','crps')]:
                macros.append('\\newcommand{\\'+prefix+word+tag+'}{'+f'{row[column]:.4f}'+'}')
    (args.out/'results_macros.tex').write_text('\n'.join(macros)+'\n')
    protocol=json.loads((args.candidate/'protocol.json').read_text())
    report=['Fractional liquidity changes within the manuscript Spectral DC framework','',
      'Development forecast ablation (full Whittle MAP, not DC posterior forecasts):',scores.to_string(index=False),'',
      'Candidate minus reference; negative favors candidate. Five-day circular block bootstrap, 2000 replicates:',paired.to_string(index=False),'',
      'Even-length fits, positive Hessians, Newton decrement <0.01, common origins, identical outcomes, and source targets independently verified.',
      'Model: clock-adjusted within-session log changes -> stationary ARTFIMA(0,d,lambda,1); reconstruct levels from the observed origin.',
      'Identical transform and history for the d=0 MA(1) null. This is ARIMA(0,1,1) on adjusted log levels.',
      'The input concatenates retained within-session changes into declared observed time; its stationarity is a working assumption, not a theorem about wall-clock continuity.',
      'No time-weighted likelihood, session random effect, state equation, powered time shard, or posterior likelihood correction is added.',
      'The period was previously examined. These are exploratory development results, not a new untouched confirmatory test.',
      'Historical approximate weights and availability selection remain. Coverage at one minute remains below nominal.',
      'The separate DC check targets the same stationary change series and native prior; full/combined MCMC diagnostics must be checked before claiming DC accuracy.',
      '',json.dumps(protocol,indent=2)]
    if (args.dc_check/'dc_vs_full.csv').exists():
        dc=pd.read_csv(args.dc_check/'dc_vs_full.csv')
        diag=pd.read_csv(args.dc_check/'shard_diagnostics.csv')
        protocol_dc=json.loads((args.dc_check/'protocol.json').read_text())
        sampling=bool((diag.rhat_max<=1.01).all() and
                      (diag.ess_bulk_min>=400).all() and (diag.ess_tail_min>=400).all())
        accuracy=bool((dc.mean_shift_reference_sd.abs()<.1).all() and
                      dc.sd_ratio.between(.9,1.1).all())
        report.extend(['','Separate manuscript DC check: '+str(args.dc_check),
                       json.dumps(protocol_dc,indent=2),diag.to_string(index=False),dc.to_string(index=False),
                       f'Sampling diagnostic gate passed: {sampling}. Posterior approximation gate passed: {accuracy}.',
                       'Sampling convergence does not establish consensus accuracy. No DC posterior forecast advantage has been established.'])
        (args.out/'dc_check.json').write_text(json.dumps(dict(source=str(args.dc_check),
             sampling_gate_passed=sampling,posterior_accuracy_gate_passed=accuracy,
             diagnostics=diag.to_dict('records'),comparison=dc.to_dict('records')),indent=2)+'\n')
    else:
        report.extend(['','No completed DC comparison found at '+str(args.dc_check)])
    (args.out/'REPORT.txt').write_text('\n'.join(report)+'\n')
    print(scores.to_string(index=False));print(paired[(paired.metric=='ae')&(paired.horizon==300)].to_string(index=False));print(args.out)


if __name__=='__main__':main()
