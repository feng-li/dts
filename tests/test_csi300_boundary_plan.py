import numpy as np
import pytest

from scripts.liquidity.plan_csi300_boundary_forecasts import audit_tasks, select_dates


def test_training_count_is_past_only_even_and_at_least_ten_million():
    raw = np.ones((3, 14220))
    tasks = audit_tasks(raw, ['20180102', '20180103', '20180104'],
                        np.array([10_000_001, 19, 100]))
    first = tasks[tasks.day_index == 0]
    assert not first.minimum_training_met.any()
    second = tasks[tasks.day_index == 1]
    assert (second.train_changes == 10_000_001).all()
    assert (second.train_fft_observations == 10_000_000).all()
    assert (second.train_last == '20180102').all()
    third = tasks[tasks.day_index == 2]
    assert (third.train_fft_observations == 10_000_020).all()


def test_horizons_do_not_cross_lunch_or_end_of_grid():
    tasks = audit_tasks(np.ones((2, 14220)), ['20180102', '20180103'],
                        np.array([10_000_000, 0]))
    selected = tasks[tasks.day_index == 1]
    assert len(selected) == 14
    assert selected[selected.origin.isin(['11:29', '14:56'])].horizon.eq(60).all()
    assert selected[selected.origin == '09:31'].context_changes.eq(59).all()
    assert selected[selected.origin == '13:05'].context_changes.eq(299).all()
    assert selected[selected.origin == '11:25'].context_changes.eq(599).all()


def test_missing_future_never_prevents_forecast_issuance_or_date_selection():
    raw = np.ones((3, 14220))
    raw[2, 60:120] = np.nan  # Missing next minute, complete 09:31 history.
    tasks = audit_tasks(raw, ['20180102', '20180103', '20180104'],
                        np.array([10_000_000, 0, 0]))
    row = tasks[(tasks.date == '20180104') & (tasks.origin == '09:31') & (tasks.horizon == 60)].iloc[0]
    assert row.issued and not row.scorable and row.status == 'missing_target'
    selected = select_dates(tasks, count=1)
    assert set(selected.date) == {'20180104'}


def test_missing_history_is_not_imputed_and_date_shortfall_is_explicit():
    raw = np.ones((2, 14220))
    raw[1] = np.nan
    tasks = audit_tasks(raw, ['20180102', '20180103'], np.array([10_000_000, 0]))
    assert not tasks.issued.any()
    with pytest.raises(ValueError, match='Only 0 usable dates'):
        select_dates(tasks, 1)
