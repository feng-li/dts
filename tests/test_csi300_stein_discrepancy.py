"""Analytic and distributional checks for the graph Stein diagnostic."""
import numpy as np
from scripts.liquidity._stein_discrepancy import graph_stein_discrepancy, verified_spanner


def test_single_atom_has_analytic_discrepancy():
    point = np.array([[.5, -.2, 0., 1.]])
    result = graph_stein_discrepancy(point, -point)
    assert np.isclose(result['discrepancy'], 4+abs(point).sum())


def test_spanner_dilation_is_verified():
    points = np.random.default_rng(10).normal(size=(80, 4))
    edges, metadata = verified_spanner(points, neighbours=2)
    assert edges.ndim == 2 and len(edges) < len(points)*(len(points)-1)//2
    assert metadata['maximum_stretch'] <= 2+1e-10


def test_biased_independent_points_score_worse_than_target_points():
    points = np.random.default_rng(81).normal(size=(100, 2))
    reference = graph_stein_discrepancy(points, -points)['discrepancy']
    shifted = points+3
    biased = graph_stein_discrepancy(shifted, -shifted)['discrepancy']
    collapsed = points*.1
    underdispersed = graph_stein_discrepancy(collapsed, -collapsed)['discrepancy']
    assert biased > reference and underdispersed > reference


def test_atom_weights_preserve_empirical_measure():
    points = np.array([[-1.], [0.], [1.]])
    result = graph_stein_discrepancy(points, -points, weights=[.25, .5, .25])
    assert np.isfinite(result['discrepancy']) and result['discrepancy'] >= 0
