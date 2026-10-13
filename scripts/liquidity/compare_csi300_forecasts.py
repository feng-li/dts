#!/usr/bin/env python3
"""Chronological, matched-origin liquidity forecasts; outputs live outside code.

Expanding quarterly fits on observed-time training data. Forecast conditioning
and targets are strictly contiguous within sessions. ARTFIMA uses full Whittle
MAP, not the previous whole-history DC posterior. Parameter uncertainty is not
included in these plug-in predictive distributions.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ[name] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from numba import njit
from scipy.fft import rfft, irfft
from scipy.linalg import toeplitz, solve_toeplitz
from scipy.optimize import minimize_scalar
from tqdm import tqdm
from scripts.liquidity.fit_csi300_observed_dc import map_fit
from scripts.liquidity._source_compat import register_relocated_protocol

MODELS = ['LastValue', 'AR1', 'ARIMA110', 'ARIMA011', 'ARTFIMA10']
PHASES = [('am', 0, 7200), ('pm', 7200, 14220)]


def transform(train):
    """Five-minute clock-bin means interpolated within phase, training only."""
    log = np.log(train)
    profile = np.empty(train.shape[1])
    for _, start, end in PHASES:
        centers, means = [], []
        for lo in range(start, end, 300):
            hi = min(lo+300, end)
            centers.append((lo+hi-1)/2)
            means.append(np.nanmean(log[:, lo:hi]))
        if not np.isfinite(means).all():
            raise ValueError('No training data for a clock bin')
        profile[start:end] = np.interp(np.arange(start,end),centers,means)
    residual = log-profile
    shift = float(np.nanmean(residual));profile += shift;residual -= shift
    scale = float(np.nanstd(residual))
    if not scale > 0:raise ValueError('Degenerate training scale')
    y = residual[np.isfinite(residual)]/scale
    return profile, scale, y


@njit(cache=True, nogil=True)
def ma_profile(theta, x):
    """Exact zero-mean Gaussian MA(1) innovations likelihood, sigma2 profiled."""
    previous = 0.; variance = 1.+theta*theta
    ss = 0.; logdet = 0.
    for value in x:
        error = value-theta*previous
        ss += error*error/variance
        logdet += np.log(variance)
        previous = error/variance
        variance = 1.+theta*theta-theta*theta/variance
    sigma = ss/len(x)
    return .5*(np.log(sigma)+logdet/len(x)), sigma


def impulse_cov(psi, variance):
    innovations = toeplitz(psi, np.r_[psi[0],np.zeros(len(psi)-1)])
    return variance*(innovations@innovations.T)


def artfima_kernel(theta, context, horizon):
    """Finite-history Gaussian predictor, checked against a doubled spectral grid."""
    phi,ll,ls,d = theta
    def gamma(n):
        w=2*np.pi*np.arange(n//2+1)/n
        q=4*np.sin(w/2)**2;e=np.exp(-np.exp(ll))
        spectrum=np.exp(ls)*((-np.expm1(-np.exp(ll)))**2+e*q)**(-d)/((1-phi)**2+phi*q)
        return irfft(spectrum,n=n)[:context+horizon]
    n=65536;old=gamma(n)
    while True:
        n*=2;new=gamma(n)
        error=np.max(np.abs(new-old))/max(abs(new[0]),1e-12)
        if error<1e-7:break
        if n>=4194304:raise ValueError(f'ARTFIMA covariance quadrature unresolved: {error}')
        old=new
    cross=new[np.arange(1,horizon+1)[:,None]+np.arange(context-1,-1,-1)[None,:]]
    weights=solve_toeplitz(new[:context],cross.T).T
    covariance=toeplitz(new[:horizon])-weights@cross.T
    return weights,(covariance+covariance.T)/2,dict(grid=n,relative_error=float(error))


def kernel(model, parameters, context, horizon):
    if model=='AR1':
        phi=parameters['phi'];weights=np.zeros((horizon,context));weights[:,-1]=phi**np.arange(1,horizon+1)
        covariance=impulse_cov(phi**np.arange(horizon),parameters['sigma2'])
    elif model=='ARIMA110':
        phi=parameters['phi'];a=np.cumsum(phi**np.arange(1,horizon+1))
        weights=np.zeros((horizon,context));weights[:,-1]=1+a;weights[:,-2]=-a
        covariance=impulse_cov(np.cumsum(phi**np.arange(horizon)),parameters['sigma2'])
    elif model=='ARIMA011':
        theta=parameters['theta'];p=1+theta*theta;last=np.zeros(context)
        for j in range(1,context):
            new=-theta*last;new[j]+=1;new[j-1]-=1
            last=new/p
            uncertainty=1-1/p
            p=1+theta*theta-theta*theta/p
        weights=np.tile(theta*last,(horizon,1));weights[:,-1]+=1
        covariance=impulse_cov(np.r_[1.,np.full(horizon-1,1+theta)],parameters['sigma2'])
        covariance+=parameters['sigma2']*theta*theta*uncertainty
    elif model=='ARTFIMA10':
        return artfima_kernel(np.asarray(parameters['theta']),context,horizon)
    else:raise ValueError(model)
    return weights,covariance,{}


def interval_score(actual, lower, upper, alpha):
    return upper-lower+2/alpha*((lower-actual)*(actual<lower)+(actual-upper)*(actual>upper))


def window_reason(values, lo, hi, end, context, horizon):
    if end-context<lo or end+horizon>hi:return 'session_boundary'
    if not np.isfinite(values[end-context:end]).all():return 'missing_history'
    if not np.isfinite(values[end:end+horizon]).all():return 'missing_target'
    return None


def score_samples(actual, point, draws):
    draws=np.sort(draws)
    n=len(draws)
    crps=np.mean(np.abs(draws-actual))-np.dot(2*np.arange(1,n+1)-n-1,draws)/(n*n)
    q=np.quantile(draws,[.025,.1,.9,.975])
    return dict(actual=actual,prediction=point,ae=abs(point-actual),se=(point-actual)**2,
                crps=crps,lower80=q[1],upper80=q[2],lower95=q[0],upper95=q[3],
                cover80=float(q[1]<=actual<=q[2]),cover95=float(q[0]<=actual<=q[3]),
                interval80=interval_score(actual,q[1],q[2],.2),
                interval95=interval_score(actual,q[0],q[3],.05))


def train_models(train):
    profile,scale,y=transform(train)
    parameters={};timings={}
    start=time.monotonic()
    phi=float(np.clip(np.dot(y[:-1],y[1:])/np.dot(y[:-1],y[:-1]),-.999999,.999999))
    variance=float(np.mean((y[1:]-phi*y[:-1])**2))
    parameters['AR1']=dict(phi=phi,sigma2=variance,success=True,fit='Conditional Gaussian least squares, zero residual mean')
    timings['AR1']=time.monotonic()-start
    delta=np.diff(y);start=time.monotonic()
    phi=float(np.clip(np.dot(delta[:-1],delta[1:])/np.dot(delta[:-1],delta[:-1]),-.999999,.999999))
    parameters['ARIMA110']=dict(phi=phi,sigma2=float(np.mean((delta[1:]-phi*delta[:-1])**2)),success=True,fit='Conditional Gaussian least squares on first differences, no drift')
    timings['ARIMA110']=time.monotonic()-start;start=time.monotonic()
    ma=minimize_scalar(lambda t:ma_profile(t,delta)[0],bounds=(-.999999,.999999),method='bounded',options={'xatol':1e-8})
    parameters['ARIMA011']=dict(theta=float(ma.x),sigma2=float(ma_profile(ma.x,delta)[1]),success=bool(ma.success),fit='Exact Gaussian MA(1) likelihood on first differences, no drift')
    timings['ARIMA011']=time.monotonic()-start
    parameters['LastValue']=dict(sigma2=float(np.mean(delta*delta)),success=True,fit='Last observed raw spread; log random-walk prediction intervals')
    timings['LastValue']=0.
    start=time.monotonic();ft=rfft(y,workers=2)[1:(len(y)-1)//2+1]
    pg=np.abs(ft)**2/(2*np.pi*len(y));del ft
    q=4*np.sin(np.pi*np.arange(1,len(pg)+1)/len(y))**2
    starts=[np.array(s) for s in [[.99,-6.,-6.,.1],[.7,-4.,-5.,.3],[.99,-9.,-6.,-.1],[.1,-5.,-4.,.8]]]
    pilot,_,_=map_fit(q[::64],pg[::64],1,starts)
    best,_,eig=map_fit(q,pg,1,[pilot.x])
    parameters['ARTFIMA10']=dict(theta=best.x.tolist(),success=bool(best.success),message=str(best.message),
                                hessian_min=float(eig.min()),fit='Full positive-frequency Whittle MAP; native DTS priors')
    timings['ARTFIMA10']=time.monotonic()-start
    return profile,scale,parameters,timings,len(y)


def run_fold(job):
    root,out,dates,first,indices,context,draws,seed=job
    label=str(pd.Period(pd.Timestamp(dates[indices[0]]),freq='Q'))
    folder=out/label;folder.mkdir(exist_ok=True)
    if (folder/'completed.json').exists():return label
    raw=np.load(root/'spread.npy',mmap_mode='r')
    if (folder/'fit.json').exists():
        fit=json.loads((folder/'fit.json').read_text());profile=np.load(folder/'profile.npy');scale=fit['scale'];parameters=fit['parameters']
    else:
        profile,scale,parameters,timings,n=train_models(raw[:first])
        fit=dict(train_first=dates[0],train_last=dates[first-1],test_first=dates[indices[0]],test_last=dates[indices[-1]],
                 train_observations=n,scale=scale,parameters=parameters,fit_seconds=timings)
        np.save(folder/'profile.npy',profile);(folder/'fit.json').write_text(json.dumps(fit,indent=2))
    horizon=300;predictors={};quadrature={}
    for model in MODELS[1:]:
        weights,cov,info=kernel(model,parameters[model],context,horizon)
        eig=np.linalg.eigvalsh(cov)
        if eig.min() < -1e-7*max(1.,float(eig.max())):raise ValueError(f'Non-PSD covariance: {model}')
        chol=np.linalg.cholesky(cov+np.eye(horizon)*1e-12)
        predictors[model]=(weights,cov,chol)
        quadrature[model]=info
    rows=[];skips=[];origins=0
    for day_index in indices:
        values=raw[day_index]
        for phase,lo,hi in PHASES:
            for elapsed in [1800,3600,5400]:
                end=lo+elapsed
                history=values[end-context:end];future=values[end:end+horizon]
                reason=window_reason(values,lo,hi,end,context,horizon)
                if reason:
                    skips.append(dict(date=dates[day_index],phase=phase,elapsed=elapsed,reason=reason));continue
                origins+=1
                x=(np.log(history)-profile[end-context:end])/scale
                rng=np.random.default_rng(seed+day_index*100000+lo+elapsed)
                normal=rng.normal(size=(draws//2,horizon));normal=np.r_[normal,-normal]
                for model in MODELS:
                    if model=='LastValue':
                        paths=history[-1]*np.exp(scale*np.sqrt(parameters[model]['sigma2'])*np.cumsum(normal,axis=1))
                        points=np.full(horizon,history[-1])
                    else:
                        weights,cov,chol=predictors[model]
                        mean=weights@x
                        logmean=profile[end:end+horizon]+scale*mean
                        paths=np.exp(logmean+scale*(normal@chol.T))
                        points=np.exp(logmean+.5*scale*scale*np.diag(cov))
                    if not np.isfinite(paths).all():raise ValueError(f'Nonfinite forecast: {label} {model}')
                    for h in [60,300]:
                        rows.append(dict(fold=label,date=dates[day_index],phase=phase,elapsed=elapsed,
                            horizon=h,model=model,**score_samples(float(future[:h].mean()),float(points[:h].mean()),paths[:,:h].mean(axis=1))))
    pd.DataFrame(rows).to_parquet(folder/'forecasts.parquet',index=False)
    pd.DataFrame(skips,columns=['date','phase','elapsed','reason']).to_csv(folder/'skipped_origins.csv',index=False)
    (folder/'completed.json').write_text(json.dumps(dict(origins=origins,rows=len(rows),skipped=len(skips),quadrature=quadrature),indent=2))
    return label


def summarize(out, seed):
    paths=sorted(out.glob('*/forecasts.parquet'))
    a=pd.concat([pd.read_parquet(p) for p in paths],ignore_index=True)
    keys=['date','phase','elapsed','horizon']
    counts=a.groupby(keys).model.nunique()
    if not (counts==len(MODELS)).all() or a.duplicated(keys+['model']).any():raise ValueError('Unmatched forecast origins')
    a.to_parquet(out/'forecasts.parquet',index=False)
    summary=a.groupby(['model','horizon']).agg(n=('ae','size'),days=('date','nunique'),mae=('ae','mean'),mse=('se','mean'),
        crps=('crps','mean'),coverage80=('cover80','mean'),coverage95=('cover95','mean'),interval80=('interval80','mean'),interval95=('interval95','mean')).reset_index()
    summary['rmse']=np.sqrt(summary.pop('mse'));summary.to_csv(out/'scores.csv',index=False)
    comparisons=[];rng=np.random.default_rng(seed)
    for h in [60,300]:
        g=a[a.horizon==h]
        for model in MODELS:
            if model=='ARTFIMA10':continue
            for metric in ['ae','se','crps']:
                p=g.pivot(index=keys,columns='model',values=metric)
                differences=(p[model]-p['ARTFIMA10']).groupby(level='date').agg(['sum','count'])
                n=len(differences);values=differences.to_numpy();boot=[]
                for _ in range(1000):
                    # Circular moving blocks of five trading days preserve short day dependence.
                    starts=rng.integers(n,size=(n+4)//5)
                    ix=((starts[:,None]+np.arange(5))%n).ravel()[:n]
                    sampled=values[ix].sum(axis=0);boot.append(sampled[0]/sampled[1])
                comparisons.append(dict(model=model,reference='ARTFIMA10',horizon=h,metric=metric,
                    difference=float(differences['sum'].sum()/differences['count'].sum()),
                    lower=float(np.quantile(boot,.025)),upper=float(np.quantile(boot,.975))))
    pd.DataFrame(comparisons).to_csv(out/'paired_comparisons.csv',index=False)
    for grouping,name in [(['fold','model','horizon'],'fold_scores.csv'),(['phase','elapsed','model','horizon'],'clock_scores.csv')]:
        b=a.groupby(grouping).agg(n=('ae','size'),mae=('ae','mean'),mse=('se','mean'),crps=('crps','mean')).reset_index()
        b['rmse']=np.sqrt(b.pop('mse'));b.to_csv(out/name,index=False)
    print(summary.to_string(index=False))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path.home()/'running/dts/artifacts/liquidity/csi300_dc_1m')
    p.add_argument('--out',type=Path,default=Path.home()/'running/dts/artifacts/liquidity/csi300_forecast_comparison_v1')
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--train-fraction',type=float,default=.8)
    p.add_argument('--context',type=int,default=600)
    p.add_argument('--draws',type=int,default=512)
    p.add_argument('--seed',type=int,default=20261008)
    args=p.parse_args()
    if args.draws<2 or args.draws%2 or not 0<args.train_fraction<1:raise ValueError('Invalid configuration')
    args.out.mkdir(parents=True,exist_ok=True)
    source=args.input/'input_manifest.json';meta=json.loads(source.read_text());dates=meta['dates']
    first=int(len(dates)*args.train_fraction)
    quarters=pd.to_datetime(dates).to_period('Q');jobs=[]
    for q in sorted(set(quarters[first:])):
        indices=[i for i in range(first,len(dates)) if quarters[i]==q]
        jobs.append((args.input,args.out,dates,min(indices),indices,args.context,args.draws,args.seed))
    config=dict(train_fraction=args.train_fraction,context=args.context,draws=args.draws,seed=args.seed,
                test_first=dates[first],test_last=dates[-1],test_dates=len(dates)-first,folds=len(jobs),
                input_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(__file__).with_name('fit_csi300_observed_dc.py')]},
                protocol='Quarterly expanding training, 5-minute-bin training-only clock log location, global training scale; concatenated observed-time training; 600-second contiguous conditioning; matched 30/60/90 minute origins within AM/PM; actual future 60/300-second arithmetic means',
                uncertainty='Plug-in Gaussian future paths (512 by default); averages formed after exponentiation; LastValue uses log random-walk intervals; parameter uncertainty excluded',
                limitations='Retrospective approximate-weight target, not point-in-time investable weights; no old full-history fit reused; ARTFIMA full Whittle MAP not DC posterior prediction')
    path=args.out/'protocol.json'
    register_relocated_protocol(path, config, 'Changed protocol: use a new output directory')
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(run_fold,j) for j in jobs]
        for f in tqdm(as_completed(futures),total=len(jobs),desc='Forecast comparison folds'):
            print(f.result(),flush=True)
    summarize(args.out,args.seed)


if __name__=='__main__':main()
