"""DC timing must not require optimizing the full-data posterior."""
from types import SimpleNamespace

import pytest

from scripts.liquidity.benchmark_csi300_computation import (
    fit_with_budget, full_data_initialization, is_compute_process)


@pytest.mark.parametrize('method', ['dc_serial', 'dc_parallel'])
def test_dc_initialization_does_not_evaluate_full_target(method):
    def forbidden(*args, **kwargs):
        raise AssertionError('DC attempted full-data optimization')

    assert full_data_initialization(method, SimpleNamespace(fit_spectrum=forbidden)) is None


def test_full_chain_fits_its_own_target():
    calls = []
    fit = dict(theta=[0., -4., 0., .1], newton_distance=.001)

    def optimize(q, pg, groups, starts=None):
        assert starts is None
        calls.append((q, pg, groups))
        return fit

    assert full_data_initialization('full', SimpleNamespace(fit_spectrum=optimize), 'q', 'pg') is fit
    assert calls == [('q', 'pg', 1)]


def test_unresolved_full_proposal_fit_is_rejected():
    probe = SimpleNamespace(fit_spectrum=lambda *args, **kwargs: dict(newton_distance=.1))
    with pytest.raises(ValueError, match='Unresolved full-data MAP'):
        full_data_initialization('full', probe, 'q', 'pg')


@pytest.mark.parametrize('policy, count', [('two-start', 2), ('single-start', 1)])
def test_budget_passes_complete_frequency_arrays_without_pilot_or_null_fit(policy, count):
    calls = []
    q, pg = object(), object()

    def optimize(point, actual_q, actual_pg, groups):
        calls.append((actual_q, actual_pg, groups))
        return dict(negative_log_posterior=-len(calls), theta=point.tolist())

    result = fit_with_budget(SimpleNamespace(optimize_spectrum_start=optimize), q, pg, 20, policy)
    assert calls == [(q, pg, 20)]*count
    assert result['full_frequency_starts'] == count


def test_loose_budget_restores_optimizer_when_an_attempt_fails():
    original = lambda *args, **kwargs: None
    probe = SimpleNamespace(minimize=original)

    def fail(*args):
        assert probe.minimize is not original
        raise RuntimeError('attempt failed')

    probe.optimize_spectrum_start = fail
    with pytest.raises(RuntimeError, match='attempt failed'):
        fit_with_budget(probe, 'q', 'pg', 1, 'single-loose')
    assert probe.minimize is original


def test_loose_budget_applies_options_and_restores_optimizer():
    calls = []

    def minimize(*args, **kwargs):
        calls.append(kwargs['options'])
        return None

    probe = SimpleNamespace(minimize=minimize)

    def optimize(point, q, pg, groups):
        probe.minimize(None, options=dict(maxiter=1000, ftol=1e-14, gtol=1e-10, maxls=100))
        return dict(theta=point.tolist(), negative_log_posterior=0.)

    probe.optimize_spectrum_start = optimize
    fit_with_budget(probe, 'q', 'pg', 1, 'single-loose')
    assert calls == [dict(maxiter=100, ftol=1e-8, gtol=1e-6, maxls=30)]
    assert probe.minimize is minimize


@pytest.mark.parametrize('command, expected', [
    ('R', True), ('Rscript', True), ('R.bin', True), ('RDD Process', False),
    ('Renderer', False), ('python3.12', True), ('ray::Actor', True), ('java', True)])
def test_monitor_distinguishes_r_computation_from_browser_processes(command, expected):
    assert is_compute_process(command) is expected
