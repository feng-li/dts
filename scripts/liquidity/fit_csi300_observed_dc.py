#!/usr/bin/env python3
"""ARTFIMA(1,0) Spectral DC on concatenated observed-time log liquidity.

Specialized compiled objective matches dts.mcmc's positive-frequency Whittle
target and N(0,1) tail priors. This descriptive fit does not remove clock
seasonality, establish stationarity, or implement a forecasting backtest.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time

for name in ['OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_NUM_THREADS']:
    os.environ[name] = '1'
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.liquidity._source_compat import compatible_source_relocation
import numpy as np
import pandas as pd
from numba import njit
from scipy.fft import rfft, next_fast_len
from scipy.optimize import minimize
from scipy.stats import rankdata, norm
from tqdm import tqdm

NAMES = ['phi', 'log_lambda', 'log_sigma2', 'd']
BOUNDS = [(-.99999999, .99999999), (-30., 10.), (-30., 20.), (-5., 5.)]


@njit(cache=True, nogil=True)
def objective(t, q, periodogram, groups):
    phi, ll, ls, d = t
    if abs(phi) >= 1 or ll < -30 or ll > 10 or ls < -30 or ls > 20 or abs(d) > 5:
        return np.inf, np.zeros(4)
    lam = np.exp(ll)
    e = np.exp(-lam)
    one_minus_e = -np.expm1(-lam)
    value = (np.log(2.) + 1.5*np.log(2*np.pi) + .5*(ll*ll+ls*ls+d*d))/groups
    grad = np.array([0., ll/groups, ls/groups, d/groups])
    for i in range(len(q)):
        a = (1-phi)**2 + phi*q[i]
        b = one_minus_e**2 + e*q[i]
        logb = np.log(b)
        logf = ls - np.log(2*np.pi) - np.log(a) - d*logb
        ratio = periodogram[i]*np.exp(-logf)
        value += logf + ratio
        factor = 1-ratio
        grad[0] += factor*(2*(1-phi)-q[i])/a
        grad[1] += factor*(-d)*lam*e*(2*one_minus_e-q[i])/b
        grad[2] += factor
        grad[3] += factor*(-logb)
    return value, grad


def map_fit(q, pg, groups, starts):
    scaling = np.array([.001, 1., 1., 1.])
    center = np.asarray(starts[0])
    def fun(z):
        t = center + scaling*z
        v, g = objective(t, q, pg, groups)
        return v/len(q), g*scaling/len(q)
    bounds = [( (lo-center[j])/scaling[j], (hi-center[j])/scaling[j]) for j,(lo,hi) in enumerate(BOUNDS)]
    results = [minimize(fun, (s-center)/scaling, jac=True, method='L-BFGS-B', bounds=bounds,
                        options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-9}) for s in starts]
    best = min(results, key=lambda r:r.fun)
    best.x = center + scaling*best.x
    h = np.empty((4,4))
    for j in range(4):
        step = 1e-7 if j == 0 else 1e-5
        delta = np.zeros(4);delta[j] = step
        h[:,j] = (objective(best.x+delta,q,pg,groups)[1] - objective(best.x-delta,q,pg,groups)[1])/(2*step)
    vals, vecs = np.linalg.eigh((h+h.T)/2)
    cov = (vecs * (1/np.maximum(vals, 1e-5))) @ vecs.T
    return best, cov, vals


@njit(cache=True, nogil=True)
def chain(q, pg, groups, start, chol, samples, burn, seed):
    np.random.seed(seed)
    theta = start.copy()
    value = objective(theta,q,pg,groups)[0]
    draws = np.empty((samples,4))
    scale = 2.38 / 2
    accepted = 0; window = 0
    for i in range(burn+samples):
        proposal = theta + scale * (chol @ np.random.normal(0.,1.,4))
        candidate = objective(proposal,q,pg,groups)[0]
        accept = np.log(np.random.random()) < value-candidate
        if accept:
            theta = proposal;value = candidate
        if i < burn:
            window += accept
            if (i+1)%100 == 0:
                scale *= np.exp(min(.3, max(-.3, window/100-.234)))
                window = 0
        else:
            draws[i-burn] = theta
            accepted += accept
    return draws, accepted/samples


def diagnostics(draws):
    """Rank/folded split Rhat and multichain initial-positive-pair ESS."""
    m,n,p = draws.shape
    h = n//2
    split = np.concatenate([draws[:,:h],draws[:,-h:]],axis=0)
    rhats=[];esses=[]
    def rhat(x):
        w = np.var(x,axis=1,ddof=1).mean()
        if w <= 0:return float('inf')
        v = (h-1)/h*w+np.var(x.mean(axis=1),ddof=1)
        return np.sqrt(v/w)
    for j in range(p):
        x=split[:,:,j]
        z=norm.ppf((rankdata(x).reshape(x.shape)-.375)/(x.size+.25))
        folded=np.abs(x-np.median(x))
        f=norm.ppf((rankdata(folded).reshape(x.shape)-.375)/(x.size+.25))
        rhats.append(max(rhat(z),rhat(f)))
        centered=z-z.mean(axis=1,keepdims=True)
        ft=np.fft.rfft(centered,n=next_fast_len(2*h),axis=1)
        acov=np.fft.irfft(ft*ft.conj(),axis=1)[:,:h]/h
        w=acov[:,0].mean()*h/(h-1)
        v=w*(h-1)/h+np.var(z.mean(axis=1),ddof=1)
        rho=1-(w-acov.mean(axis=0))/v
        rho[0]=1
        pairs=rho[:h//2*2].reshape(-1,2).sum(axis=1)
        bad=np.flatnonzero(pairs<0)
        if len(bad):pairs=pairs[:bad[0]]
        pairs=np.minimum.accumulate(pairs)
        tau=max(1.,-1+2*pairs.sum())
        esses.append(x.size/tau)
    return np.array(rhats),np.array(esses)


def fit_shard(g, root, groups, samples, burn, seed, start):
    path=root/f'shard_{g:03d}.npz'
    if path.exists():return g
    q=np.ascontiguousarray(np.load(root/'q.npy',mmap_mode='r')[g::groups])
    pg=np.ascontiguousarray(np.load(root/'periodogram.npy',mmap_mode='r')[g::groups])
    best,cov,eigen=map_fit(q,pg,groups,[start])
    chol=np.linalg.cholesky(cov)
    rng=np.random.default_rng(seed+g)
    jobs=[]
    for c in range(4):
        initial=best.x+chol@rng.normal(size=4)*2
        if not np.isfinite(objective(initial,q,pg,groups)[0]):initial=best.x.copy()
        jobs.append((q,pg,groups,initial,chol,samples,burn,seed+10000*g+c))
    with ThreadPoolExecutor(max_workers=4) as pool:
        output=list(pool.map(lambda job:chain(*job), jobs))
    draws=[item[0] for item in output];accepts=[item[1] for item in output]
    draws=np.asarray(draws)
    rh,ess=diagnostics(draws)
    np.savez(path,draws=draws,acceptance=accepts,map=best.x,hessian_eigenvalues=eigen,
             map_success=best.success,map_message=str(best.message),rhat=rh,ess=ess)
    return g


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('artifacts/liquidity/csi300_dc_1m'))
    p.add_argument('--out',type=Path,default=Path('artifacts/liquidity/csi300_dc_1m/observed_log_artfima10'))
    p.add_argument('--groups',type=int,default=80)
    p.add_argument('--workers',type=int,default=40)
    p.add_argument('--samples',type=int,default=3000)
    p.add_argument('--burn',type=int,default=2000)
    p.add_argument('--seed',type=int,default=20261004)
    p.add_argument('--summarize-only',action='store_true',help='Reaggregate saved chains without resampling')
    args=p.parse_args();root=args.out;root.mkdir(parents=True,exist_ok=True)
    config=dict(groups=args.groups,samples=args.samples,burn=args.burn,seed=args.seed,
                input_manifest_sha256=hashlib.sha256((args.input/'input_manifest.json').read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    cp=root/'config.json'
    if cp.exists():
        old=json.loads(cp.read_text())
        keys=[k for k in config if not (args.summarize_only and k=='source_sha256')]
        changed = [k for k in keys if old.get(k) != config[k]]
        if changed:
            if (changed != ['source_sha256'] or not compatible_source_relocation(
                    {Path(__file__).name: old['source_sha256']},
                    {Path(__file__).name: config['source_sha256']})):
                raise ValueError('Configuration differs: use a new output directory')
            history = list(old.get('source_history', []))
            history.append(dict(time=time.time(), previous=old['source_sha256'],
                                current=config['source_sha256'],
                                reason='Moved scripts into scripts/liquidity; numerical calculations unchanged'))
            config['source_history'] = history
        elif old.get('source_history'):
            config['source_history'] = old['source_history']
    elif args.summarize_only:
        raise ValueError('No saved run to summarize')
    if not args.summarize_only:cp.write_text(json.dumps(config,indent=2))
    started=time.monotonic()
    if args.summarize_only and not (root/'full_map.json').exists():raise ValueError('Missing full MAP')
    if not (root/'full_map.json').exists():
        raw=np.load(args.input/'spread.npy',mmap_mode='r')
        y=raw[np.isfinite(raw)]
        if (y<=0).any():raise ValueError('Log spread requires strictly positive data')
        y=np.log(y);center=float(y.mean());scale=float(y.std());y=(y-center)/scale
        n=len(y)
        print(f'Exact FFT of {n:,} observed-time standardized log spreads',flush=True)
        ft=rfft(y,workers=8)[1:(n-1)//2+1]
        pg=np.abs(ft)**2/(2*np.pi*n);del ft
        omega=2*np.pi*np.arange(1,len(pg)+1)/n
        q=4*np.sin(omega/2)**2
        np.save(root/'q.npy',q);np.save(root/'periodogram.npy',pg)
        rows=[];dates=json.loads((args.input/'input_manifest.json').read_text())['dates']
        for year in sorted({d[:4] for d in dates}):
            v=raw[[i for i,d in enumerate(dates) if d.startswith(year)]]
            v=np.log(v[np.isfinite(v)]);z=(v-center)/scale
            rows.append(dict(year=year,n=len(z),residual_mean=z.mean(),residual_sd=z.std()))
        pd.DataFrame(rows).to_csv(root/'yearly_stability.csv',index=False)
        print('Multistart initialization, then full-frequency MAP',flush=True)
        pilot,_,_=map_fit(q[::80],pg[::80],1,[np.array(s) for s in
             [[.8,-5,-3,.3],[.1,-8,-4,.8],[.95,-3,-4,.1],[-.5,-6,-4,1.]]])
        best,cov,eigen=map_fit(q,pg,1,[pilot.x])
        meta=dict(n=n,frequencies=len(pg),log_center=center,log_scale=scale,
                  map=best.x.tolist(),map_success=bool(best.success),map_message=str(best.message),
                  hessian_eigenvalues=eigen.tolist(),covariance=cov.tolist(),
                  likelihood='Single FFT after deleting missing observations and concatenating sessions',
                  transform='log(spread_bps), globally demeaned and scaled using the entire sample',
                  priors='phi uniform(-1,1); log_lambda,log_sigma2,d independent N(0,1); prior power 1/G',
                  computational_bounds=BOUNDS,
                  caveat='Descriptive observed-time approximation; no clock adjustment, stationarity claim or out-of-sample validation')
        (root/'full_map.json').write_text(json.dumps(meta,indent=2))
        print(json.dumps(meta,indent=2),flush=True)
    meta=json.loads((root/'full_map.json').read_text())
    # Compile before parallel work. Numba releases the GIL during each chain.
    q=np.array([.1,.2]);pg=np.array([.1,.2]);s=np.array(meta['map'])
    if not args.summarize_only:
        chain(q,pg,args.groups,s,np.eye(4)*.0001,2,2,args.seed)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            jobs=[pool.submit(fit_shard,g,root,args.groups,args.samples,args.burn,args.seed,s) for g in range(args.groups)]
            for job in tqdm(as_completed(jobs),total=len(jobs),desc='Spectral DC shards'):job.result()
    # Estimate one precision matrix per shard using all chains. Applying different
    # estimated weights to each chain can create artificial between-chain shifts.
    from dts.aggregation import consensus
    shards=[np.load(root/f'shard_{g:03d}.npz') for g in range(args.groups)]
    combined=consensus([a['draws'].reshape(-1,4) for a in shards],ridge=1e-12).reshape(4,args.samples,4)
    np.save(root/'consensus_draws.npy',combined)
    rh,ess=diagnostics(combined)
    table=[]
    for g,a in enumerate(shards):
        table.append(dict(shard=g,accept_min=a['acceptance'].min(),accept_max=a['acceptance'].max(),
                          rhat_max=a['rhat'].max(),ess_min=a['ess'].min(),map_success=bool(a['map_success']),
                          hessian_min=a['hessian_eigenvalues'].min()))
    diagnostic=pd.DataFrame(table);diagnostic.to_csv(root/'shard_diagnostics.csv',index=False)
    transformed=combined.reshape(-1,4).copy();transformed[:,1:3]=np.exp(transformed[:,1:3])
    quantiles=np.quantile(transformed,[.025,.5,.975],axis=0)
    summary=pd.DataFrame(dict(parameter=['phi','lambda','sigma2','d'],lower=quantiles[0],median=quantiles[1],upper=quantiles[2],rhat=rh,ess=ess))
    summary.to_csv(root/'posterior_summary.csv',index=False)
    result=dict(n=meta['n'],groups=args.groups,chains_per_shard=4,retained_draws_per_chain=args.samples,
                worst_shard_rhat=float(diagnostic.rhat_max.max()),minimum_shard_ess=float(diagnostic.ess_min.min()),
                shards_failing_rhat=int((diagnostic.rhat_max>1.01).sum()),
                shards_failing_ess=int((diagnostic.ess_min<400).sum()),
                map_failures=int((~diagnostic.map_success).sum()),
                nonpositive_hessians=int((diagnostic.hessian_min<=0).sum()),
                consensus_rhat_max=float(rh.max()),consensus_ess_min=float(ess.min()),
                wall_seconds=time.monotonic()-started,
                aggregation='One per-shard precision from all chains; common weights applied across chains; ridge 1e-12',
                aggregation_source_sha256=config['source_sha256'],
                caveat=meta['caveat'])
    if args.summarize_only and (root/'results.json').exists():
        result['wall_seconds']=json.loads((root/'results.json').read_text())['wall_seconds']
    (root/'results.json').write_text(json.dumps(result,indent=2))
    print(summary.to_string(index=False));print(json.dumps(result,indent=2))


if __name__=='__main__':main()
