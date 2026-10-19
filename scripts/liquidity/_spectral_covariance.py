"""Reduce Monte Carlo error in the covariance estimates used by consensus.

The MCMC covariance is shrunk towards a separately converged Gauss-Hermite
estimate of the same four-dimensional subposterior covariance. The shrinkage
strength uses batch estimates of Monte Carlo variance, without full-reference
samples or forecast outcomes. It vanishes as sampling uncertainty vanishes
when the numerical target differs from the true covariance.
"""
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np
from numba import njit
from scipy.special import logsumexp

from scripts.liquidity.run_csi300_fractional_changes_dc import target_value


@njit(cache=True, nogil=True)
def coordinate_densities(points, q, periodogram, groups):
    """Visit every assigned ordinate at every integration point."""
    result = np.empty(len(points))
    for i in range(len(points)):
        point = points[i]
        if point[1] <= 0 or not np.isfinite(point).all():
            result[i] = -np.inf
            continue
        native = point.copy()
        native[0] = np.tanh(point[0])
        native[1] = 2*np.log(point[1])
        absolute = abs(point[0])
        logj = 2*(np.log(2.)-absolute-np.log1p(np.exp(-2*absolute))) + np.log(2.)-np.log(point[1])
        result[i] = -target_value(native, q, periodogram, groups) + logj/groups
    return result


def quadrature_moments(q, periodogram, groups, center, covariance,
                       orders=(5, 7, 9), tolerance=1e-4):
    """Check integration convergence in standardized, dimensionless units."""
    center, covariance = np.asarray(center), np.asarray(covariance)
    chol = np.linalg.cholesky(covariance)
    previous = None
    history = []
    for order in orders:
        nodes, weights = np.polynomial.hermite.hermgauss(order)
        indices = np.array(list(itertools.product(range(order), repeat=4)))
        standard = np.sqrt(2.)*nodes[indices]
        points = center + standard@chol.T
        proposal = -.5*np.sum(standard**2, axis=1) - .5*np.linalg.slogdet(covariance)[1] - 2*np.log(2*np.pi)
        log_weights = (coordinate_densities(points, q, periodogram, groups) - proposal
                       + np.log(weights[indices]/np.sqrt(np.pi)).sum(axis=1))
        normalizer = float(logsumexp(log_weights))
        if not np.isfinite(normalizer):
            raise ValueError('No finite quadrature mass')
        normalized = np.exp(log_weights-normalizer)
        mean = normalized@points
        centered = points-mean
        cov = (centered.T*normalized)@centered
        np.linalg.cholesky(cov)
        current = dict(mean=mean, covariance=cov, log_normalizer=normalizer)
        record = dict(order=order, nodes=len(points))
        if previous is not None:
            mean_error = float(abs(np.linalg.solve(chol, mean-previous['mean'])).max())
            difference = np.linalg.solve(chol, np.linalg.solve(chol, cov-previous['covariance']).T).T
            covariance_error = float(np.linalg.norm(difference)/2)
            normalizer_error = abs(normalizer-previous['log_normalizer'])
            record.update(mean_error=mean_error, covariance_error=covariance_error,
                          normalizer_error=normalizer_error)
            history.append(record)
            if max(mean_error, covariance_error, normalizer_error) <= tolerance:
                return {**current, 'order':order, 'history':history, 'passed':True}
        else:
            history.append(record)
        previous = current
    raise ValueError(f'Quadrature covariance did not converge: {history[-1]}')


def regularize_covariance(draws, anchor, batches=30, agreement_limit=6.):
    """Shrink in standardized units using estimated sampling error only."""
    flat = np.asarray(draws).reshape(-1, 4)
    sample = np.cov(flat, rowvar=False)
    chol = np.linalg.cholesky(anchor)
    white = np.linalg.solve(chol, (flat-flat.mean(axis=0)).T).T
    count = min(batches, len(flat)//128)
    if count < 4:
        raise ValueError('Insufficient draws for covariance error estimation')
    width = len(flat)//count
    blocks = white[:count*width].reshape(count, width, 4)
    outer = np.einsum('bni,bnj->bij', blocks, blocks)/width
    variance = outer.var(axis=0, ddof=1)/count
    standardized = np.linalg.solve(chol, np.linalg.solve(chol, sample).T).T
    difference = standardized-np.eye(4)
    agreement = float(np.max(abs(difference)/np.maximum(np.sqrt(variance), 1e-8)))
    if agreement > agreement_limit:
        raise ValueError(f'Quadrature/MCMC covariance disagreement: {agreement:.3f} standard errors')
    squared = float(np.sum(difference**2))
    strength = min(1., float(variance.sum())/squared) if squared else 1.
    covariance = (1-strength)*sample + strength*anchor
    np.linalg.cholesky(covariance)
    return covariance, dict(shrinkage_strength=strength,
                            covariance_agreement_max_z=agreement,
                            covariance_mc_variance=float(variance.sum()),
                            covariance_batches=count)


def ensure_covariance(group, folder, config):
    from scripts.liquidity._spectral_sampling import atomic_npz
    folder = Path(folder)
    frequency = folder/'frequencies'/f'shard{group:03d}.npz'
    sample = folder/'shards'/f'G{config["groups"]}_shard{group:03d}.npz'
    destination = folder/'covariances'
    destination.mkdir(exist_ok=True)
    output = destination/f'shard{group:03d}.npz'
    identity = dict(version=1, frequency_sha256=hashlib.sha256(frequency.read_bytes()).hexdigest(),
                    sample_sha256=hashlib.sha256(sample.read_bytes()).hexdigest(),
                    source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                    groups=config['groups'])
    def write_quality(quality):
        sidecar=output.with_suffix('.json')
        temporary=sidecar.with_name(sidecar.name+f'.{os.getpid()}.partial')
        temporary.write_text(json.dumps(quality,indent=2));temporary.replace(sidecar)
    if output.exists():
        with np.load(output) as saved:
            if json.loads(str(saved['identity_json'].item())) == identity:
                quality=json.loads(str(saved['quality_json'].item()))
                write_quality(quality)
                return quality
        raise ValueError('Covariance cache inputs changed')
    with np.load(sample) as values:
        draws = values['coordinate_draws']
        metadata = json.loads(str(values['quality_json'].item()))
    fit = metadata['coordinate_map']
    with np.load(frequency) as values:
        moments = quadrature_moments(values['q'], values['periodogram'], config['groups'],
                                     fit['theta'], fit['covariance'])
    covariance, regularization = regularize_covariance(draws, moments['covariance'])
    quality = dict(covariance_passed=True, covariance_method='regularized',
                   quadrature_order=moments['order'], quadrature_history=moments['history'],
                   **regularization)
    atomic_npz(output, covariance=covariance, quadrature_covariance=moments['covariance'],
               quadrature_mean=moments['mean'], identity_json=json.dumps(identity),
               quality_json=json.dumps(quality))
    write_quality(quality)
    return quality
