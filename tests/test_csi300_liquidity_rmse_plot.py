"""Check paired score ratios, uncertainty and missing-origin display."""
import numpy as np
import pandas as pd
import pytest
import matplotlib.pyplot as plt

from scripts.liquidity.plot_csi300_liquidity_forecast_errors import (
    MODELS, ORIGINS, draw, paired_rmse_gains,
)


def forecasts():
    rows = []
    for day in range(80):
        for horizon in [60, 300]:
            for origin in ORIGINS:
                if horizon == 300 and origin in ['11:29', '14:56']:
                    continue
                # Unequal cluster counts make pooled averaging substantive.
                if day < 15 and origin == '09:31':
                    continue
                for model, ratio in [(MODELS[0][0], .8), (MODELS[1][0], 1.)]:
                    rows.append(dict(day_index=day+915, date=f'{day:08d}',
                                     origin=origin, horizon=horizon, model=model,
                                     error=ratio*(day+1)))
    return pd.DataFrame(rows)


def test_paired_scale_gain_and_pooled_rmse():
    result = paired_rmse_gains(forecasts(), replicates=200)
    valid = result[result.n > 0]
    # A fixed 0.8 error ratio gives exactly 20%, for every paired resample.
    np.testing.assert_allclose(valid[['rmse_reduction_pct', 'lower_pct', 'upper_pct']], 20)
    pool = valid[(valid.horizon == 60) & (valid.origin == 'Pooled')].iloc[0]
    f = forecasts()
    e = f[(f.horizon == 60) & (f.model == MODELS[1][0])].error
    assert pool.rmse_benchmark == pytest.approx(np.sqrt(np.mean(e**2)))
    assert pool.clusters == 5  # original archive offsets, not retained-row ranks


def test_uncertainty_crosses_zero_when_daily_winner_changes():
    f = forecasts()
    sel = f.model == MODELS[0][0]
    f.loc[sel, 'error'] = np.where(f.loc[sel, 'day_index'] < 955, .8, 1.2)*10
    f.loc[~sel, 'error'] = 10
    a = paired_rmse_gains(f, replicates=1000, seed=7)
    b = paired_rmse_gains(f, replicates=1000, seed=7)
    pd.testing.assert_frame_equal(a, b)
    pooled = a[a.origin == 'Pooled']
    assert (pooled.lower_pct < 0).all() and (pooled.upper_pct > 0).all()


def test_unavailable_five_minute_slots_remain_empty():
    result = paired_rmse_gains(forecasts(), replicates=100)
    missing = result[(result.horizon == 300) & result.origin.isin(['11:29', '14:56'])]
    assert missing.n.eq(0).all()
    fig = draw(result)
    assert list(fig.axes[0].get_xticks()) == list(range(8))
    assert list(fig.axes[1].get_xticks()) == [0, 1, 2, 4, 5, 6]
    plt.close(fig)


def test_actor_count_must_be_positive():
    with pytest.raises(ValueError, match='ray_actors must be at least 1'):
        paired_rmse_gains(forecasts(), ray_actors=0)


def test_parallel_completion_order_preserves_display_order(monkeypatch):
    from scripts.liquidity import plot_csi300_liquidity_forecast_errors as plot
    expected = paired_rmse_gains(forecasts(), replicates=200)

    def reversed_completion(jobs, weights, ray_actors):
        assert ray_actors == 4
        return {index: plot._summarize_rmse_bootstrap(row, sums, weights)
                for index, row, sums in reversed(jobs)}

    monkeypatch.setattr(plot, '_bootstrap_with_ray', reversed_completion)
    actual = paired_rmse_gains(forecasts(), replicates=200, ray_actors=4)
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
