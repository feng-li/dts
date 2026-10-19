#!/usr/bin/env python3
"""Audit quarterly ARTFIMA modes and report the matched forecast comparison."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[name]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/csi300-forecast-mpl')
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.fft import rfft
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from tqdm import tqdm
from scripts.liquidity.fit_csi300_observed_dc import objective
from scripts.liquidity.compare_csi300_forecasts import MODELS


def audit(job):
    folder,source=job
    fit=json.loads((folder/'fit.json').read_text())
    dates=json.loads((source/'input_manifest.json').read_text())['dates']
    last=dates.index(fit['train_last'])+1
    raw=np.load(source/'spread.npy',mmap_mode='r')[:last]
    y=(np.log(raw)-np.load(folder/'profile.npy'))/fit['scale'];y=y[np.isfinite(y)]
    ft=rfft(y,workers=2)[1:(len(y)-1)//2+1];pg=np.abs(ft)**2/(2*np.pi*len(y))
    q=4*np.sin(np.pi*np.arange(1,len(pg)+1)/len(y))**2
    t=np.array(fit['parameters']['ARTFIMA10']['theta']);_,gradient=objective(t,q,pg,1)
    hessian=np.empty((4,4))
    for j in range(4):
        delta=np.zeros(4);delta[j]=1e-7 if j==0 else 1e-5
        hessian[:,j]=(objective(t+delta,q,pg,1)[1]-objective(t-delta,q,pg,1)[1])/(2*delta[j])
    hessian=(hessian+hessian.T)/2;eigen=np.linalg.eigvalsh(hessian)
    distance=float(np.sqrt(gradient@np.linalg.solve(hessian,gradient))) if eigen.min()>0 else float('inf')
    return dict(fold=folder.name,optimizer_success=fit['parameters']['ARTFIMA10']['success'],
                newton_distance=distance,hessian_min=float(eigen.min()),local_mode_check=distance<.01,
                train_last=fit['train_last'],test_first=fit['test_first'],train_observations=len(y),
                phi=t[0],lambda_=np.exp(t[1]),d=t[3])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path.home()/'running/dts/artifacts/liquidity/csi300_forecast_comparison_v1')
    p.add_argument('--input',type=Path,default=Path.home()/'running/dts/artifacts/liquidity/csi300_dc_1m')
    p.add_argument('--pdf',type=Path,default=Path.home()/'running/dts/output/pdf/csi300_forecast_comparison.pdf')
    args=p.parse_args();root=args.root
    fits=sorted(root.glob('*/fit.json'))
    with ThreadPoolExecutor(max_workers=2) as pool:
        checks=pd.DataFrame(tqdm(pool.map(audit,[(f.parent,args.input) for f in fits]),total=len(fits),desc='Audit training-only MAP fits'))
    checks.to_csv(root/'fit_audit.csv',index=False)
    if not checks.local_mode_check.all():raise ValueError('Unresolved ARTFIMA mode checks; see fit_audit.csv')
    scores=pd.read_csv(root/'scores.csv');paired=pd.read_csv(root/'paired_comparisons.csv')
    forecasts=pd.read_parquet(root/'forecasts.parquet')
    skips=pd.concat([pd.read_csv(f) for f in sorted(root.glob('*/skipped_origins.csv'))],ignore_index=True)
    skips.to_csv(root/'skipped_origins.csv',index=False)
    protocol=json.loads((root/'protocol.json').read_text())
    n=forecasts[['date','phase','elapsed']].drop_duplicates().shape[0]
    data=dict(test_dates=protocol['test_dates'],dates_with_scored_origins=forecasts.date.nunique(),
              candidate_origins=protocol['test_dates']*6,scored_origins=n,skipped_origins=len(skips),
              skip_reasons=skips.reason.value_counts().to_dict(),optimizer_warnings=int((~checks.optimizer_success).sum()),
              all_local_mode_checks_passed=True,models=MODELS,
              test_first=protocol['test_first'],test_last=protocol['test_last'])
    (root/'evaluation.json').write_text(json.dumps(data,indent=2))
    lines=['CSI300 liquidity forecast comparison',json.dumps(data,indent=2),'',
           'Scores on identical origins (errors in bps; horizon in seconds):',scores.to_string(index=False),'',
           'Paired differences: comparator minus ARTFIMA. Negative favors comparator.',
           'Intervals use 1,000 circular moving-block replicates of five trading days.',paired.to_string(index=False),'',
           'Protocol: first 80% of dates initialize training; expanding quarterly refits.',
           'All models use the same earlier observed-time history. Test forecasts condition on 600 contiguous seconds.',
           'Training-only five-minute clock-bin log means are interpolated separately in AM/PM, with a global training scale.',
           'Forecast targets are arithmetic average raw spreads over the NEXT 60 or 300 seconds.',
           'Origins are 30, 60 and 90 minutes into each session; any missing context/target excludes that origin from every model.',
           'ARIMA first-difference models have no drift. AR1 has zero adjusted mean.',
           'AR1/ARIMA110 use conditional Gaussian least squares; ARIMA011 uses exact Gaussian MA likelihood on differences.',
           'ARTFIMA is refitted by full-data positive-frequency Whittle MAP on earlier data, not the old all-history DC draws.',
           'Reported local MAP checks use Newton decrement <0.01 and positive Hessian; they do not prove global optimality.',
           'Gaussian conditional paths are simulated (512 antithetic draws) and exponentiated BEFORE averaging.',
           'Point forecasts use analytic marginal lognormal expectations, except LastValue which uses the last raw spread.',
           'LastValue intervals use a log random walk with training-estimated innovation variance.',
           'Intervals omit parameter uncertainty. CRPS and quantiles are Monte Carlo estimates.',
           'ARTFIMA conditioning uses the latest 600 seconds; covariance integration is checked by doubling the spectral grid.',
           'Data construction includes historical approximate/future-derived weights; this is a retrospective target comparison, not a fully point-in-time investable backtest.',
           'Missing targets select the evaluation sample. Do not extrapolate results to excluded periods.',
           'No Bayesian posterior/DC approximation comparison is made here.', '',
           'Reproduce:', 'python scripts/liquidity/compare_csi300_forecasts.py --workers 2',
           'python scripts/liquidity/report_csi300_forecast_comparison.py']
    (root/'REPORT.txt').write_text('\n'.join(lines)+'\n')
    colors=['#555555','#167c80','#ce7930','#697cc1','#9e4b79']
    labels=['Last value','AR(1)','ARIMA(1,1,0)','ARIMA(0,1,1)','ARTFIMA(1,0)']
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    args.pdf.parent.mkdir(parents=True,exist_ok=True)
    def finish(pdf,fig,title,page):
        fig.suptitle(title,x=.07,ha='left',fontsize=17,y=.97)
        fig.text(.07,.025,'CSI300 RMB 1m | Retrospective approximate-weight target | Training-only transformations',fontsize=9,color='#555555')
        fig.text(.95,.025,str(page),ha='right')
        fig.tight_layout(rect=[.04,.055,.98,.92]);pdf.savefig(fig);plt.close(fig)
    with PdfPages(args.pdf) as pdf:
        fig,axs=plt.subplots(2,3,figsize=(13,8))
        for i,h in enumerate([60,300]):
            s=scores[scores.horizon==h].set_index('model').loc[MODELS]
            for ax,metric,title in zip(axs[i],['mae','rmse','crps'],['MAE','RMSE','CRPS']):
                ax.bar(np.arange(5),s[metric],color=colors)
                ax.set_xticks(np.arange(5),labels,rotation=25,ha='right',fontsize=8)
                ax.set_title(f'{h//60}-minute average: {title}');ax.set_ylabel('bps; lower is better')
                ax.grid(axis='y',alpha=.2)
        finish(pdf,fig,f'1  Forecast errors | {n:,} matched origins on {data["dates_with_scored_origins"]} dates',1)
        fig,axs=plt.subplots(2,2,figsize=(13,8))
        for i,h in enumerate([60,300]):
            s=scores[scores.horizon==h].set_index('model').loc[MODELS]
            for j,level in enumerate([80,95]):
                ax=axs[i,j];ax.bar(np.arange(5),100*s[f'coverage{level}'],color=colors)
                ax.axhline(level,color='black',ls='--',lw=1,label=f'Nominal {level}%')
                ax.set_xticks(np.arange(5),labels,rotation=20,ha='right',fontsize=9)
                ax.set(ylabel='Empirical coverage (%)',ylim=(0,103),title=f'{h//60}-minute average: {level}% intervals');ax.legend()
        finish(pdf,fig,'2  Predictive interval coverage | Parameter uncertainty excluded',2)
        fig,axs=plt.subplots(1,2,figsize=(13,7))
        for ax,h in zip(axs,[60,300]):
            s=paired[(paired.horizon==h)&(paired.metric=='ae')].set_index('model').loc[MODELS[:-1]]
            ax.errorbar(s.difference,np.arange(4),xerr=np.array([s.difference-s.lower,s.upper-s.difference]),fmt='o',capsize=4,color='#167c80')
            ax.axvline(0,color='black',ls='--',lw=1);ax.set_yticks(np.arange(4),labels[:-1]);ax.invert_yaxis()
            ax.set(title=f'{h//60}-minute average',xlabel='MAE difference vs ARTFIMA (bps)\nNegative favors the comparator')
            ax.grid(axis='x',alpha=.2)
        finish(pdf,fig,'3  Paired accuracy differences | 95% intervals, five-day block bootstrap',3)
        fig,ax=plt.subplots(figsize=(13,8));ax.axis('off')
        notes=[f"Test: {data['test_first']} to {data['test_last']} | {data['test_dates']} scheduled dates | 8 quarterly folds.",
               f"Origins: {data['candidate_origins']:,} scheduled; {n:,} scored; {len(skips):,} excluded for missing context or target.",
               'Same models, training histories, forecast origins and targets in every comparison.',
               'Earlier-data log clock profile (five-minute bins) and scale; 600-second contiguous conditioning window.',
               'Targets: future 60/300-second average spread in bps, never crossing missing intervals or session boundaries.',
               'ARTFIMA: full training Whittle MAP, not the previous whole-history DC posterior.',
               f"Optimizer warnings: {data['optimizer_warnings']}; all 8 independent local-mode checks passed.",
               '512 joint Gaussian paths per origin/model; paths exponentiated before averaging; parameter uncertainty excluded.',
               'Last-value point prediction has log random-walk intervals; ARIMA models have no drift.',
               'Training retains the user-selected observed-time concatenation of historical gaps and session boundaries.',
               'Approximate historical weights remain: these results are not a fully point-in-time investable backtest.',
               'Results apply to available windows; skipped periods may have different liquidity behavior.',
               'Prediction error, uncertainty calibration, and DC posterior accuracy are separate questions.']
        ax.text(.01,.98,'\n\n'.join(notes),va='top',fontsize=11,transform=ax.transAxes)
        finish(pdf,fig,'4  Protocol and interpretation',4)
    print(json.dumps(data,indent=2));print(args.pdf)


if __name__=='__main__':main()
