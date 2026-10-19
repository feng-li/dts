#!/usr/bin/env python3
"""Run the manuscript's native sampler/consensus on cached liquidity changes.

Consumes the exact full-series periodogram produced by the framework probe.
This entry point adds no likelihood factors or time-domain partitioning.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
for name in ['OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS']:
    os.environ[name]='1'
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from numba import njit
from tqdm import tqdm
from dts.aggregation import consensus,wasserstein_quantile_distance
from dts.mcmc import sampler
from scripts.liquidity.fit_csi300_observed_dc import diagnostics
from scripts.liquidity.probe_csi300_fractional_changes import fit_spectrum,target
from scripts.liquidity._source_compat import register_relocated_protocol


@njit(cache=True,nogil=True)
def target_value(theta,q,pg,groups):
    """Same native target, avoiding unused gradient work in MH evaluations."""
    ma,ll,ls,d=theta
    if not np.isfinite(theta).all() or abs(ma)>=1:
        return np.inf
    lam=np.exp(ll);e=np.exp(-lam);b0=(-np.expm1(-lam))**2
    value=(np.log(2)+1.5*np.log(2*np.pi)+.5*(ll*ll+ls*ls+d*d))/groups
    for j in range(len(q)):
        logf=ls-np.log(2*np.pi)+np.log((1+ma)**2-ma*q[j])-d*np.log(b0+e*q[j])
        value+=logf+pg[j]*np.exp(-logf)
    return value


@njit(cache=True,nogil=True)
def regular_target_value(theta,q,pg,groups,n,group):
    """Exact full-frequency evaluation for systematic G=1/2 on even T.

    Roots-of-unity products evaluate the sum of log spectra analytically;
    the periodogram/spectrum term still visits every assigned ordinate.
    This is algebraic evaluation of the same likelihood, not subsampling.
    """
    if n<=0 or (groups!=1 and groups!=2):
        return target_value(theta,q,pg,groups)
    ma,ll,ls,d=theta
    if not np.isfinite(theta).all() or abs(ma)>=1:
        return np.inf
    lam=np.exp(ll);e=np.exp(-lam);b0=(-np.expm1(-lam))**2
    m=n//groups
    phase=1 if groups==1 or group==1 else -1
    loga=np.log1p(-phase*((-ma)**m))
    logb=np.log(-np.expm1(-lam*m)) if phase==1 else np.log1p(np.exp(-lam*m))
    if groups==1 or group==1:
        loga-=np.log1p(ma);logb-=np.log(-np.expm1(-lam))
    if (n//2-1)%groups==group:
        loga-=np.log1p(-ma);logb-=np.log1p(e)
    value=(np.log(2)+1.5*np.log(2*np.pi)+.5*(ll*ll+ls*ls+d*d))/groups
    value+=len(q)*(ls-np.log(2*np.pi))+loga-d*logb
    ratio=0.
    for j in range(len(q)):
        ratio+=pg[j]*((b0+e*q[j])**d)/((1+ma)**2-ma*q[j])
    return value+2*np.pi*np.exp(-ls)*ratio


@njit(cache=True,nogil=True)
def compiled_rwm(q,pg,groups,initial,chol,total,burn,seed,n=0,group=0):
    """The manuscript's fixed Hessian Gaussian random-walk Metropolis rule."""
    np.random.seed(seed);theta=initial.copy();value=regular_target_value(theta,q,pg,groups,n,group)
    draws=np.empty((total-burn,4));accepted=0
    for iteration in range(total):
        proposal=theta+chol@np.random.normal(0.,1.,4)
        candidate=regular_target_value(proposal,q,pg,groups,n,group)
        if np.log(np.random.random())<value-candidate:
            theta=proposal;value=candidate
            if iteration>=burn:accepted+=1
        if iteration>=burn:draws[iteration-burn]=theta
    return draws,accepted/(total-burn)


def shard_job(job):
    folder,out,groups,group,chains,total,burn,seed,backend=job
    path=out/f'G{groups}_shard{group}.npz'
    if path.exists():return groups,group
    metadata=json.loads((folder/'fit.json').read_text());n=metadata['n']
    q=np.ascontiguousarray(np.load(folder/'q.npy',mmap_mode='r')[group::groups])
    pg=np.ascontiguousarray(np.load(folder/'periodogram.npy',mmap_mode='r')[group::groups])
    omega=2*np.pi*np.arange(1,n//2)[group::groups]/n
    fit=fit_spectrum(q,pg,groups=groups)
    if fit['covariance'] is None or fit['newton_distance']>.01:
        raise ValueError(f'MAP unresolved for G={groups}, shard={group}: {fit}')
    center=np.array(fit['theta']);cov=np.array(fit['covariance'])*(2.38**2/4)
    rng=np.random.default_rng(seed+10000*groups+group)
    initial_states=[]
    for chain in range(chains):
        initial=center+np.linalg.cholesky(cov)@rng.normal(size=4)
        if not np.isfinite(target(initial,q,pg,groups)[0]):initial=center.copy()
        initial_states.append(initial)
    def run(chain):
        chain_seed=seed+100000*groups+1000*group+chain
        if backend=='compiled':
            a,acceptance=compiled_rwm(q,pg,groups,initial_states[chain],np.linalg.cholesky(cov),total,burn,chain_seed)
        else:
            a,logp,accepted=sampler(ar_order=0,ma_order=1,data=None,I_pg=pg,TFI_term=True,
                 omega_shard=omega,n_samples=total,paramsStar=initial_states[chain],proposal_width=cov,
                 Burn_in=burn,exact=False,n_groups=groups,mean_term=False,
                 random_state=chain_seed,progress=False)
            if not np.isfinite(logp).all():raise ValueError('Nonfinite log posterior')
            acceptance=float(accepted.mean())
        if not np.isfinite(a).all():raise ValueError('Nonfinite draws')
        return a,acceptance
    with ThreadPoolExecutor(max_workers=chains) as pool:output=list(pool.map(run,range(chains)))
    draws=[a for a,_ in output];acceptance=[acceptance for _,acceptance in output]
    draws=np.asarray(draws);rh,ess=diagnostics(draws)
    quantiles=np.quantile(draws.reshape(-1,4),[.05,.95],axis=0)
    indicators=np.concatenate([(draws<=quantiles[0]).astype(float),(draws>=quantiles[1]).astype(float)],axis=2)
    _,tail=diagnostics(indicators)
    np.savez_compressed(path,draws=draws,acceptance=acceptance,rhat=rh,ess_bulk=ess,ess_tail=tail,
                        metadata=json.dumps(fit))
    return groups,group


def main():
    p=argparse.ArgumentParser(description=__doc__)
    base=Path.home()/'running/dts/artifacts/liquidity'
    p.add_argument('--input',type=Path,default=base/'csi300_fractional_changes_clean_probe')
    p.add_argument('--out',type=Path,default=base/'csi300_fractional_changes_dc')
    p.add_argument('--fold',default='2024Q4')
    p.add_argument('--groups',default='1,2,4')
    p.add_argument('--chains',type=int,default=4)
    p.add_argument('--iterations',type=int,default=15000)
    p.add_argument('--burn',type=int,default=5000)
    p.add_argument('--workers',type=int,default=4)
    p.add_argument('--seed',type=int,default=20261008)
    p.add_argument('--backend',choices=['native','compiled'],default='native')
    args=p.parse_args();groups=[int(g) for g in args.groups.split(',')]
    if min(groups)<1 or len(set(groups))!=len(groups) or args.iterations-args.burn<4 or args.chains<2:raise ValueError('Invalid configuration')
    folder=args.input/args.fold;out=args.out/args.fold;out.mkdir(parents=True,exist_ok=True)
    config=dict(fold=args.fold,groups=groups,chains=args.chains,iterations=args.iterations,burn=args.burn,seed=args.seed,backend=args.backend,
       input_sha256=hashlib.sha256((folder/'fit.json').read_bytes()).hexdigest(),
       source_sha256={str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [Path(__file__),Path(__file__).with_name('probe_csi300_fractional_changes.py')]},
       model='ARTFIMA(0,d,lambda,1) on fixed transformed changes',
       method='Manuscript fixed Gaussian random-walk MH; systematic frequency partition; native prior 1/G; fixed MAP Hessian proposal; package joint consensus with common per-shard weights',
       scope='Observed-time working series; no added likelihood or mean factor; stationarity and boundary convention remain modelling assumptions')
    cp=out/'protocol.json'
    register_relocated_protocol(cp, config, 'Use a new output after configuration/source changes')
    jobs=[(folder,out,g,j,args.chains,args.iterations,args.burn,args.seed,args.backend) for g in groups for j in range(g)]
    if args.backend=='compiled':
        t=np.array([.1,-2.,0.,.2]);compiled_rwm(np.array([.1,.2]),np.array([.1,.2]),1,t,np.eye(4)*.001,5,1,args.seed)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(shard_job,j) for j in jobs]
        for f in tqdm(as_completed(futures),total=len(futures),desc='Manuscript Spectral DC shards'):print(f.result(),flush=True)
    rows=[];merged={}
    for g in groups:
        shards=[np.load(out/f'G{g}_shard{j}.npz') for j in range(g)]
        for j,a in enumerate(shards):
            rows.append(dict(groups=g,shard=j,rhat_max=a['rhat'].max(),ess_bulk_min=a['ess_bulk'].min(),ess_tail_min=a['ess_tail'].min(),acceptance_min=a['acceptance'].min()))
        shape=shards[0]['draws'].shape
        combined=consensus([a['draws'].reshape(-1,4) for a in shards],ridge=1e-12).reshape(shape)
        merged[g]=combined;np.save(out/f'G{g}_joint_draws.npy',combined)
        rh,ess=diagnostics(combined);transformed=combined.reshape(-1,4).copy();transformed[:,1:3]=np.exp(transformed[:,1:3])
        quantiles=np.quantile(transformed,[.025,.5,.975],axis=0)
        pd.DataFrame(dict(parameter=['theta','lambda','sigma2','d'],lower=quantiles[0],median=quantiles[1],upper=quantiles[2],rhat=rh,ess_bulk=ess)).to_csv(out/f'G{g}_summary.csv',index=False)
    pd.DataFrame(rows).to_csv(out/'shard_diagnostics.csv',index=False)
    if 1 in merged:
        reference=merged[1].reshape(-1,4);sd=reference.std(axis=0,ddof=1);comparison=[]
        for g in groups:
            if g==1:continue
            candidate=merged[g].reshape(-1,4)
            w=wasserstein_quantile_distance(reference,candidate)
            for j,name in enumerate(['theta','log_lambda','log_sigma2','d']):
                comparison.append(dict(groups=g,parameter=name,normalized_quantile_w1=w[j],
                    mean_shift_reference_sd=(candidate[:,j].mean()-reference[:,j].mean())/sd[j],
                    sd_ratio=candidate[:,j].std(ddof=1)/sd[j]))
        pd.DataFrame(comparison).to_csv(out/'dc_vs_full.csv',index=False)
    print(out)


if __name__=='__main__':main()
