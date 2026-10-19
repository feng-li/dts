"""Exact-target sampling blocks and interruption-safe numerical checkpoints."""
import json
import os
from pathlib import Path

import numpy as np
from numba import njit,prange


@njit(cache=True,nogil=True,parallel=True)
def full_reference_target(point,q,pg,n):
    """Exact full-grid log-spectrum sums; parallelize the complete ratio sum."""
    ma,ll,ls,d=point
    if not np.isfinite(point).all() or abs(ma)>=1:return np.inf
    lam=np.exp(ll);e=np.exp(-lam);b0=(-np.expm1(-lam))**2
    loga=np.log1p(-(-ma)**n)-np.log1p(ma)-np.log1p(-ma)
    logb=np.log(-np.expm1(-lam*n))-np.log(-np.expm1(-lam))-np.log1p(e)
    ratio=0.
    for j in prange(len(q)):
        ratio+=pg[j]*((b0+e*q[j])**d)/((1+ma)**2-ma*q[j])
    value=np.log(2)+1.5*np.log(2*np.pi)+.5*(ll*ll+ls*ls+d*d)
    return value+len(q)*(ls-np.log(2*np.pi))+loga-d*logb+2*np.pi*np.exp(-ls)*ratio


@njit(cache=True, nogil=True)
def mixture_log_density(point, centers, precisions, log_determinants):
    values=np.empty(len(centers))
    for k in range(len(centers)):
        delta=point-centers[k]
        values[k]=-.5*(delta@precisions[k]@delta+log_determinants[k]+4*np.log(2*np.pi))
    maximum=values.max()
    return maximum+np.log(np.exp(values-maximum).sum())-np.log(len(centers))


@njit(cache=True, nogil=True)
def reference_steps(q,pg,n,initial,chol,centers,component_chols,precisions,
                    log_determinants,iterations,burn,seed):
    """Full-data MH reference: fixed RWMH plus exact mode-jump MH moves.

    The fixed 10% independence component uses the complete Gaussian-mixture
    proposal density in its acceptance ratio. Every likelihood evaluation
    includes all assigned frequencies, with the original G=1 prior.
    """
    np.random.seed(seed)
    point=initial.copy();value=full_reference_target(point,q,pg,n)
    draws=np.empty((iterations-burn,4));accepted=np.zeros(iterations-burn,dtype=np.bool_)
    for iteration in range(iterations):
        jump=np.random.random()<.1
        component=np.random.randint(len(centers)) if jump else 0
        proposal=(centers[component]+component_chols[component]@np.random.normal(0.,1.,4)
                  if jump else point+chol@np.random.normal(0.,1.,4))
        candidate=full_reference_target(proposal,q,pg,n)
        correction=(mixture_log_density(point,centers,precisions,log_determinants)
                    -mixture_log_density(proposal,centers,precisions,log_determinants)) if jump else 0.
        accept=np.log(np.random.random())<value-candidate+correction
        if accept:point=proposal;value=candidate
        if iteration>=burn:
            draws[iteration-burn]=point;accepted[iteration-burn]=accept
    return draws,accepted


def atomic_npz(path,**values):
    path=Path(path)
    temporary=path.with_name(path.name+f'.{os.getpid()}.partial')
    with temporary.open('wb') as handle:np.savez_compressed(handle,**values)
    temporary.replace(path)


def sampling_blocks(path,initial,covariance,pg,omega,groups,chains,iterations,burn,
                    maximum,extension,ess_target,seed,quality_function,
                    backend='compiled',reference=None,parameterization='native',
                    kernel='random-walk',proposal_center=None):
    """Continue retained chains from their last state with a fixed proposal.

    Each completed block atomically stores draws, current states, and NumPy
    RNG state. An interrupted block repeats from that checkpoint. Burn-in is
    applied once. ESS diagnostics can extend the retained sample, up to the
    configured maximum; they never change the target or loosen the gate.
    """
    from dts.mcmc import sampler
    if parameterization not in {'native','ma-sqrtlambda'}:
        raise ValueError('Unknown sampling parameterization')
    if kernel!='random-walk' and parameterization=='native':
        raise ValueError('The mixture kernel requires the boundary-aware coordinates')
    path=Path(path)
    generators=[np.random.default_rng(seed+c) for c in range(chains)]
    points=np.array(initial,dtype=float)
    draws=np.empty((chains,0,4));rates=[];total=0;history=[]
    if path.exists():
        with np.load(path) as saved:
            saved_parameterization=str(saved['parameterization'].item()) if 'parameterization' in saved else 'native'
            saved_kernel=str(saved['kernel'].item()) if 'kernel' in saved else 'random-walk'
            if saved_parameterization!=parameterization or saved_kernel!=kernel:
                raise ValueError('Checkpoint parameterization or MH kernel changed')
            draws=saved['draws'];points=saved['points'];total=int(saved['iterations'])
            rates=json.loads(str(saved['rates_json'].item()))
            history=json.loads(str(saved['history_json'].item()))
            states=json.loads(str(saved['rng_states_json'].item()))
            if 'proposal_covariance' in saved:covariance=saved['proposal_covariance']
            if 'proposal_center' in saved and saved['proposal_center'].size:
                proposal_center=saved['proposal_center']
        for generator,state in zip(generators,states):generator.bit_generator.state=state
    while True:
        if total:
            quality=quality_function(draws)
            if (quality['passed'] and quality['ess_bulk_min']>=ess_target
                    and quality['ess_tail_min']>=ess_target) or total>=maximum:break
        count=iterations if not total else min(extension,maximum-total)
        discarded=burn if not total else 0
        block=[];accepted=[]
        for chain in range(chains):
            if reference is None and parameterization=='ma-sqrtlambda':
                from scripts.liquidity._spectral_coordinates import coordinate_sampler
                result,logp,accept=coordinate_sampler(points[chain],covariance,pg,omega,
                    groups,count,discarded,generators[chain],backend=backend,
                    kernel=kernel,center=proposal_center)
                if not np.isfinite(logp).all():raise ValueError('Nonfinite coordinate posterior density')
            elif reference is None:
                result,logp,accept=sampler(ar_order=0,ma_order=1,data=None,I_pg=pg,
                    TFI_term=True,omega_shard=omega,n_samples=count,Burn_in=discarded,
                    paramsStar=points[chain],proposal_width=covariance,n_groups=groups,
                    random_state=generators[chain],progress=False,backend=backend)
                if not np.isfinite(logp).all():raise ValueError('Nonfinite posterior density')
            else:
                block_seed=int(generators[chain].integers(0,2**31-1))
                result,accept=reference_steps(reference['q'],pg,reference['n'],points[chain],
                    np.linalg.cholesky(covariance),reference['centers'],reference['chols'],
                    reference['precisions'],reference['log_determinants'],count,discarded,block_seed)
            if not np.isfinite(result).all():raise ValueError('Nonfinite posterior samples')
            block.append(result);accepted.append(int(accept.sum()))
        added=np.asarray(block)
        draws=np.concatenate([draws,added],axis=1);points=draws[:,-1].copy()
        total+=count;rates.append(dict(draws=count-discarded,accepted=accepted))
        quality=quality_function(draws)
        history.append(dict(iterations=total,ess_bulk_min=quality['ess_bulk_min'],
                            ess_tail_min=quality['ess_tail_min'],passed=quality['passed']))
        atomic_npz(path,draws=draws,points=points,iterations=total,
            rates_json=json.dumps(rates),history_json=json.dumps(history),
            rng_states_json=json.dumps([g.bit_generator.state for g in generators]),
            parameterization=parameterization,kernel=kernel,
            proposal_covariance=covariance,
            proposal_center=np.asarray(proposal_center) if proposal_center is not None else np.empty(0))
    denominator=sum(r['draws'] for r in rates)
    acceptance=np.sum([r['accepted'] for r in rates],axis=0)/denominator
    return draws,acceptance.tolist(),total,history
