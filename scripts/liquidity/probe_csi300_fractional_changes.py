#!/usr/bin/env python3
"""Forecast ablation for a manuscript-compatible stationary response transform.

Y is the first difference of clock-adjusted observed-time log spread. The
ARTFIMA(0,d,lambda,1) positive-frequency target and native prior are unchanged.
MAP forecasts are a development probe, not Spectral DC posterior forecasts.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[name]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from numba import njit
from scipy.fft import rfft,irfft
from scipy.linalg import solve_toeplitz,toeplitz
from scipy.optimize import minimize
from tqdm import tqdm
from scripts.liquidity.compare_csi300_forecasts import PHASES,window_reason,score_samples,transform
from scripts.liquidity._source_compat import register_relocated_protocol


@njit(cache=True,nogil=True)
def target(t,q,pg,groups=1):
    """Native DTS negative Whittle log posterior for p=0,q=1."""
    theta,ll,ls,d=t
    if abs(theta)>=1 or abs(ll)>40 or abs(ls)>40 or abs(d)>20:
        return np.inf,np.zeros(4)
    lam=np.exp(ll);e=np.exp(-lam);b0=(-np.expm1(-lam))**2
    value=(np.log(2)+1.5*np.log(2*np.pi)+.5*(ll*ll+ls*ls+d*d))/groups
    gradient=np.array([0.,ll/groups,ls/groups,d/groups])
    for j in range(len(q)):
        a=(1+theta)**2-theta*q[j];b=b0+e*q[j];logb=np.log(b)
        logf=ls-np.log(2*np.pi)+np.log(a)-d*logb
        ratio=pg[j]*np.exp(-logf);factor=1-ratio
        value+=logf+ratio
        gradient[0]+=factor*(2*(1+theta)-q[j])/a
        gradient[1]+=factor*(-d)*lam*e*(2*(-np.expm1(-lam))-q[j])/b
        gradient[2]+=factor;gradient[3]-=factor*logb
    return value,gradient


def null_objective(z,q,pg):
    """Original full-data MA(1) negative Whittle posterior and scaled score."""
    theta,ls=z
    if abs(theta)>=1:return np.inf,np.zeros(2)
    a=(1+theta)**2-theta*q;logf=ls-np.log(2*np.pi)+np.log(a);ratio=pg*np.exp(-logf)
    v=np.sum(logf+ratio)+np.log(2)+.5*(np.log(2*np.pi)+ls*ls)
    g=np.array([np.sum((1-ratio)*(2*(1+theta)-q)/a),np.sum(1-ratio)+ls])
    return v/len(q),g/len(q)


@njit(cache=True,nogil=True)
def null_hessian(z,q,pg):
    """Analytic curvature on the unscaled posterior, without large temporaries."""
    theta,ls=z
    hessian=np.zeros((2,2));hessian[1,1]=1.  # Normal prior on log variance.
    for j in range(len(q)):
        a=(1+theta)**2-theta*q[j]
        derivative=(2*(1+theta)-q[j])/a
        ratio=pg[j]*np.exp(-(ls-np.log(2*np.pi)+np.log(a)))
        hessian[0,0]+=(2*ratio-1)*derivative**2+2*(1-ratio)/a
        hessian[0,1]+=ratio*derivative
        hessian[1,1]+=ratio
    hessian[1,0]=hessian[0,1]
    return hessian


def null_map_quality(z,q,pg):
    """Validate stationarity and positive curvature independently of optimizer status."""
    if not np.isfinite(z).all() or abs(z[0])>=1:
        return dict(passed=False,newton_distance=float('inf'),hessian_eigenvalues=[])
    value,score=null_objective(z,q,pg)
    hessian=null_hessian(z,q,pg)
    eigen=np.linalg.eigvalsh(hessian)
    distance=float('inf')
    if np.isfinite(value) and np.isfinite(score).all() and np.isfinite(eigen).all() and eigen.min()>0:
        gradient=score*len(q)
        squared=float(gradient@np.linalg.solve(hessian,gradient))
        distance=float(np.sqrt(max(0.,squared)))
    return dict(passed=bool(distance<=.01),newton_distance=distance,
                hessian_eigenvalues=eigen.tolist())


def fit_null_spectrum(q,pg,initial):
    """Keep validated original estimates; retry unresolved line searches on the same target."""
    options={'ftol':1e-14,'gtol':1e-10,'maxiter':500}
    bounds=[(-.99999999,.99999999),(-30.,20.)]
    short=minimize(lambda z:null_objective(z,q,pg),initial,jac=True,
                   method='L-BFGS-B',bounds=bounds,options=options)
    quality=null_map_quality(short.x,q,pg)
    attempts=[dict(optimizer_success=bool(short.success),message=str(short.message),
                   **quality)]
    if not quality['passed']:
        retry=minimize(lambda z:null_objective(z,q,pg),short.x,jac=True,
                       method='L-BFGS-B',bounds=bounds,
                       options={**options,'maxiter':1000,'maxls':100})
        retry_quality=null_map_quality(retry.x,q,pg)
        attempts.append(dict(optimizer_success=bool(retry.success),message=str(retry.message),
                             **retry_quality))
        if retry_quality['passed'] or retry.fun<short.fun:
            short,quality=retry,retry_quality
    return dict(null_theta=short.x.tolist(),null_success=quality['passed'],
                null_optimizer_success=bool(short.success),null_message=str(short.message),
                null_newton_distance=quality['newton_distance'],
                null_hessian_eigenvalues=quality['hessian_eigenvalues'],null_attempts=attempts)


def fit_spectrum(q,pg,groups=1):
    bounds=[(-.99999999,.99999999),(-30.,10.),(-30.,20.),(-2.,2.)]
    def optimize(q,pg,starts):
        def objective(t):
            v,g=target(t,q,pg,groups)
            return v/len(q),g/len(q)
        results=[minimize(objective,x,jac=True,bounds=bounds,method='L-BFGS-B',
                           options={'maxiter':1000,'ftol':1e-14,'gtol':1e-10}) for x in starts]
        return min(results,key=lambda r:r.fun)
    pilot=optimize(q[::64],pg[::64],[np.array(s) for s in
        [[-.2,-5.,0.,.1],[-.2,-7.,0.,-.1],[0.,-3.,0.,.3],[-.5,-10.,0.,-.3],[.3,-5.,0.,-.2]]])
    best=optimize(q,pg,[pilot.x])
    hessian=np.empty((4,4))
    for j in range(4):
        delta=np.zeros(4);delta[j]=1e-5
        hessian[:,j]=(target(best.x+delta,q,pg,groups)[1]-target(best.x-delta,q,pg,groups)[1])/(2e-5)
    hessian=(hessian+hessian.T)/2;eigen=np.linalg.eigvalsh(hessian)
    gradient=target(best.x,q,pg,groups)[1]
    decrement=float(np.sqrt(gradient@np.linalg.solve(hessian,gradient))) if eigen.min()>0 else float('inf')
    # The d=0 null removes the unidentified lambda coordinate, as in the manuscript.
    short=fit_null_spectrum(q,pg,[best.x[0],best.x[2]])
    return dict(theta=best.x.tolist(),optimizer_success=bool(best.success),message=str(best.message),
                hessian_eigenvalues=eigen.tolist(),newton_distance=decrement,
                **short,
                covariance=np.linalg.inv(hessian).tolist() if eigen.min()>0 else None)


def increment_predictor(parameters,context,horizon,fractional=True):
    theta,ll,ls,d=parameters
    if not fractional:d=0.
    def gamma(n):
        w=2*np.pi*np.arange(n//2+1)/n;q=4*np.sin(w/2)**2;e=np.exp(-np.exp(ll))
        spectral=np.exp(ls)*((1+theta)**2-theta*q)*((-np.expm1(-np.exp(ll)))**2+e*q)**(-d)
        return irfft(spectral,n=n)[:context+horizon]
    n=65536;old=gamma(n)
    while True:
        n*=2;g=gamma(n);error=np.max(abs(g-old))/max(abs(g[0]),1e-12)
        if error<1e-7:break
        if n>=4194304:raise ValueError('Covariance quadrature did not resolve')
        old=g
    cross=g[np.arange(1,horizon+1)[:,None]+np.arange(context-1,-1,-1)[None,:]]
    weights=solve_toeplitz(g[:context],cross.T).T
    covariance=toeplitz(g[:horizon])-weights@cross.T
    # Integrate changes, preserving their joint covariance.
    weights=np.cumsum(weights,axis=0)
    covariance=np.cumsum(np.cumsum(covariance,axis=0),axis=1)
    return weights,(covariance+covariance.T)/2


def minute_profile(train,base):
    """Training-only phase-specific 60-position clock effect (fixed harmonics)."""
    result=np.array(base,copy=True)
    for _,lo,hi in PHASES:
        residual=np.log(train[:,lo:hi])-base[lo:hi]
        offsets=np.arange(hi-lo)%60
        means=np.array([np.nanmean(residual[:,offsets==s]) for s in range(60)])
        # Fix the zero Fourier coefficient to zero; all 59 other real harmonics
        # of the 60-position profile remain frozen over the test quarter.
        means-=means.mean()
        result[lo:hi]+=means[offsets]
    return result


def training_changes(train,profile,boundary='exclude'):
    values=(np.log(train)-profile).ravel()
    positions=np.flatnonzero(np.isfinite(values))
    changes=np.diff(values[positions])
    left,right=positions[:-1],positions[1:]
    breaks=(right-left!=1)|(right//14220!=left//14220)|((left%14220<7200)&(right%14220>=7200))
    audit=dict(boundary_changes=int(breaks.sum()),boundary_squared_share=float(np.sum(changes[breaks]**2)/np.sum(changes**2)),
               within_session_changes=int((~breaks).sum()),boundary_policy=boundary)
    if boundary=='exclude':changes=changes[~breaks]
    return changes,audit


def fold(job):
    source,old,out,label,seed,clock,train_days,boundary=job
    folder=out/label;folder.mkdir(exist_ok=True)
    if (folder/'forecasts.parquet').exists():return label
    previous=json.loads((old/label/'fit.json').read_text());dates=json.loads((source/'input_manifest.json').read_text())['dates']
    raw=np.load(source/'spread.npy',mmap_mode='r');profile=np.load(old/label/'profile.npy')
    last=dates.index(previous['train_last'])+1
    first=max(0,last-train_days) if train_days else 0
    train=raw[first:last]
    if (folder/'profile.npy').exists():profile=np.load(folder/'profile.npy')
    else:
        if train_days:profile=transform(train)[0]
        if clock=='folded':profile=minute_profile(train,profile)
        np.save(folder/'profile.npy',profile)
    if (folder/'fit.json').exists():fit=json.loads((folder/'fit.json').read_text())
    else:
        start=time.monotonic();y,audit=training_changes(train,profile,boundary);scale=float(y.std());y/=scale
        removed=int(len(y)%2)
        if removed:y=y[1:]
        n=len(y);ft=rfft(y,workers=2)[1:n//2];pg=abs(ft)**2/(2*np.pi*n);q=4*np.sin(np.pi*np.arange(1,len(pg)+1)/n)**2
        fit=fit_spectrum(q,pg);fit.update(scale=scale,n=n,train_last=previous['train_last'],fit_seconds=time.monotonic()-start)
        fit.update(earliest_change_removed_for_even_length=removed,minute_clock=clock)
        fit.update(train_first=dates[first],train_days=last-first,change_audit=audit)
        np.save(folder/'q.npy',q);np.save(folder/'periodogram.npy',pg)
        (folder/'fit.json').write_text(json.dumps(fit,indent=2))
    predictors={}
    for model,fractional in [('FractionalChanges',True),('MA1Changes',False)]:
        theta=fit['theta'] if fractional else [fit['null_theta'][0],0.,fit['null_theta'][1],0.]
        w,cov=increment_predictor(theta,599,300,fractional)
        if np.linalg.eigvalsh(cov).min()<-1e-7:raise ValueError('Nonpositive forecast covariance')
        predictors[model]=(w,cov,np.linalg.cholesky(cov+np.eye(300)*1e-12))
    reference=pd.read_parquet(old/label/'forecasts.parquet');origins=reference.drop_duplicates(['date','phase','elapsed'])
    rows=[]
    for row in origins.itertuples():
        index=dates.index(row.date);lo,hi=(0,7200) if row.phase=='am' else (7200,14220);end=lo+row.elapsed
        values=raw[index]
        if window_reason(values,lo,hi,end,600,300):raise ValueError('Reference origin invalid')
        past=np.log(values[end-600:end])-profile[end-600:end];history=np.diff(past)/fit['scale']
        rng=np.random.default_rng(seed+index*100000+lo+row.elapsed);normal=rng.normal(size=(256,300));normal=np.r_[normal,-normal]
        for model,(w,cov,chol) in predictors.items():
            logmean=np.log(values[end-1])+profile[end:end+300]-profile[end-1]+fit['scale']*(w@history)
            paths=np.exp(logmean+fit['scale']*(normal@chol.T))
            point=np.exp(logmean+.5*fit['scale']**2*np.diag(cov))
            for horizon in [60,300]:
                rows.append(dict(fold=label,date=row.date,phase=row.phase,elapsed=row.elapsed,horizon=horizon,model=model,
                    **score_samples(float(values[end:end+horizon].mean()),float(point[:horizon].mean()),paths[:,:horizon].mean(axis=1))))
    pd.DataFrame(rows).to_parquet(folder/'forecasts.parquet',index=False)
    return label


def main():
    p=argparse.ArgumentParser(description=__doc__)
    base=Path.home()/'running/dts/artifacts/liquidity'
    p.add_argument('--input',type=Path,default=base/'csi300_dc_1m')
    p.add_argument('--reference',type=Path,default=base/'csi300_forecast_comparison_v1')
    p.add_argument('--out',type=Path,default=base/'csi300_fractional_changes_clean_probe')
    p.add_argument('--minute-clock',choices=['none','folded'],default='folded')
    p.add_argument('--train-days',type=int,default=250,help='Earlier archive dates; 0 uses all earlier dates')
    p.add_argument('--boundary-changes',choices=['keep','exclude'],default='exclude')
    p.add_argument('--workers',type=int,default=2);p.add_argument('--seed',type=int,default=20261008)
    args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    protocol=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),reference=str(args.reference),seed=args.seed,minute_clock=args.minute_clock,train_days=args.train_days,boundary_changes=args.boundary_changes,
      transform='First difference of training-clock-adjusted log spread; exclude physical boundary differences when requested, then concatenate the retained changes in declared observed time; training innovation scale; zero conditional change mean',
      candidate='ARTFIMA(0,d,lambda,1) changes; native manuscript standard-normal tail priors and uniform MA prior',
      null='d=0 and lambda removed: MA(1) changes, hence ARIMA(0,1,1) adjusted log levels',
      inference='Development full Whittle MAP ablation; not a DC posterior forecast; even-length transformed series',
      evaluation='Same eight earlier-training quarterly folds and 1676 origins as the existing comparison; already-examined exploratory test period')
    cp=args.out/'protocol.json'
    register_relocated_protocol(cp, protocol, 'Use a new output directory after changes',
                                source_name=Path(__file__).name)
    labels=sorted(f.parent.name for f in args.reference.glob('*/fit.json'))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(fold,(args.input,args.reference,args.out,label,args.seed,args.minute_clock,args.train_days,args.boundary_changes)) for label in labels]
        for f in tqdm(as_completed(futures),total=len(futures),desc='Fractional changes ablation'):print(f.result(),flush=True)
    a=pd.concat([pd.read_parquet(args.out/label/'forecasts.parquet') for label in labels],ignore_index=True)
    a.to_parquet(args.out/'forecasts.parquet',index=False)
    scores=a.groupby(['model','horizon']).agg(n=('ae','size'),mae=('ae','mean'),mse=('se','mean'),crps=('crps','mean'),coverage95=('cover95','mean')).reset_index()
    scores['rmse']=np.sqrt(scores.pop('mse'));scores.to_csv(args.out/'scores.csv',index=False);print(scores.to_string(index=False))


if __name__=='__main__':main()
