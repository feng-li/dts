"""Check external source execution and the unchanged forecast-input contract."""
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from scripts.liquidity._construction import COMMANDS, DTS_ROOT, construction_checkout
from scripts.liquidity.plan_csi300_boundary_forecasts import audit_tasks

SCRIPTS = DTS_ROOT / 'scripts/liquidity'
HFL = DTS_ROOT / 'HighFrequencyLiquidity'


@pytest.fixture
def isolated_checkout(tmp_path):
    root = tmp_path / 'construction checkout'
    scripts = root / 'scripts/liquidity'
    scripts.mkdir(parents=True)
    (root / 'scripts/__init__.py').write_text('')
    (scripts / '__init__.py').write_text('')
    (scripts / '_paths.py').write_text(
        'from pathlib import Path\nROOT = Path(__file__).resolve().parents[2]\n')
    projects = root / 'projects/liquidity'
    projects.mkdir(parents=True)
    (projects / 'replay.py').write_text('# Checkout identity for this process test.\n')
    program = '''import json, os, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.liquidity._paths import ROOT
print(json.dumps(dict(root=str(ROOT), cwd=os.getcwd(), arguments=sys.argv[1:],
                     data=os.environ['HFL_DATA_ROOT'], output=os.environ['HFL_OUTPUT_ROOT'])))
raise SystemExit(7)
'''
    for name in ['build_all_csi300_liquidity', 'build_csi300_hf_liquidity',
                 'prepare_csi300_dc', 'report_csi300_archive']:
        (scripts / f'{name}.py').write_text(program)
    return root


def child_env(root):
    env = os.environ.copy()
    for key in ['HFL_DATA_ROOT', 'HFL_OUTPUT_ROOT']:
        env.pop(key, None)
    # Having DTS first on PYTHONPATH must not load its scripts in the child.
    env.update(HFL_REPO=str(root), PYTHONPATH=str(DTS_ROOT))
    return env


@pytest.mark.parametrize('name', ['build_all_csi300_liquidity',
                                  'build_csi300_hf_liquidity', 'prepare_csi300_dc'])
def test_legacy_commands_run_external_sources_and_preserve_exit_status(
        tmp_path, isolated_checkout, name):
    arguments = ['--out', 'an output with spaces', '--workers', '2']
    result = subprocess.run([sys.executable, str(SCRIPTS / f'{name}.py'), *arguments],
                            cwd=tmp_path, env=child_env(isolated_checkout),
                            capture_output=True, text=True)
    assert result.returncode == 7, result.stderr
    saved = json.loads(result.stdout)
    assert saved['root'] == str(isolated_checkout)
    assert saved['cwd'] == str(tmp_path)
    assert saved['data'] == str(isolated_checkout / 'data/csi300')
    assert saved['output'] == str(Path.home() / 'running/dts/artifacts/liquidity')
    assert saved['arguments'][-len(arguments):] == arguments
    if name == 'prepare_csi300_dc':
        assert saved['arguments'][:4] == [
            '--root', saved['output'] + '/csi300_monthly_imputed_1000ms',
            '--out', saved['output'] + '/csi300_dc_1m']


def test_launcher_override_and_input_output_environment(tmp_path, isolated_checkout):
    env = child_env(tmp_path / 'wrong checkout')
    env.update(HFL_DATA_ROOT=str(tmp_path / 'custom data'),
               HFL_OUTPUT_ROOT=str(tmp_path / 'custom output'))
    result = subprocess.run([
        sys.executable, str(SCRIPTS / 'run_construction.py'),
        '--hfl-repo', str(isolated_checkout), 'report_csi300_archive', 'saved archive'],
        cwd=tmp_path, env=env, capture_output=True, text=True)
    assert result.returncode == 7, result.stderr
    saved = json.loads(result.stdout)
    assert saved['root'] == str(isolated_checkout)
    assert saved['data'] == env['HFL_DATA_ROOT']
    assert saved['output'] == env['HFL_OUTPUT_ROOT']
    assert saved['arguments'] == ['saved archive']


def test_missing_external_checkout_does_not_use_local_snapshot(tmp_path, monkeypatch):
    monkeypatch.setenv('HFL_REPO', str(tmp_path / 'absent'))
    with pytest.raises(FileNotFoundError, match='HFL_REPO'):
        construction_checkout()


@pytest.mark.skipif(not (HFL / 'scripts/liquidity/_paths.py').is_file(),
                    reason='Initialise the HighFrequencyLiquidity submodule')
def test_default_checkout_is_root_submodule(monkeypatch):
    monkeypatch.delenv('HFL_REPO', raising=False)
    assert construction_checkout() == HFL.resolve()


@pytest.mark.skipif(not (HFL / 'scripts/liquidity/_paths.py').is_file(),
                    reason='Requires the separate HighFrequencyLiquidity source checkout')
def test_external_export_feeds_existing_forecast_planner(tmp_path):
    assert all((HFL / 'scripts/liquidity' / f'{name}.py').is_file() for name in COMMANDS)
    index = tmp_path / 'source/index'
    index.mkdir(parents=True)
    times = np.r_[np.arange(34201000, 41400001, 1000),
                  np.arange(46801000, 53820001, 1000)]
    dates = ['20230103', '20230104']
    expected = np.vstack([10 + np.sin(np.arange(14220) / 300),
                          11 + np.cos(np.arange(14220) / 300)])
    expected[0, 200:230] = np.nan
    expected[0, 7300:7320] = np.nan
    for date, values in zip(dates, expected):
        frame = pd.DataFrame(dict(size_spec='notional_1e+06', t_ms=times,
                                  session=np.where(times < 42000000, 'am', 'pm'),
                                  liq_available_vw_bps=values))
        other_size = frame.iloc[:10].assign(size_spec='notional_5e+06',
                                            liq_available_vw_bps=999.)
        pd.concat([other_size, frame]).to_parquet(index / f'{date}.parquet', index=False)
    out = tmp_path / 'prepared'
    result = subprocess.run([
        sys.executable, str(SCRIPTS / 'prepare_csi300_dc.py'),
        '--root=' + str(index.parent), '--out=' + str(out), '--workers', '2'],
        cwd=tmp_path, env=child_env(HFL), capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    raw = np.load(out / 'spread.npy')
    np.testing.assert_array_equal(raw, expected)
    meta = json.loads((out / 'input_manifest.json').read_text())
    assert meta['dates'] == dates and meta['shape'] == [2, 14220]
    assert meta['target'] == 'liq_available_vw_bps'
    assert meta['size_spec'] == 'notional_1e+06'
    assert meta['valid_observations'] == int(np.isfinite(expected).sum())
    runs = pd.read_csv(out / 'contiguous_runs.csv')
    assert set(runs.session) == {'am', 'pm'}
    assert ((runs.start < 7200) == (runs.stop <= 7200)).all()
    daily_changes = np.bincount(runs.day_index, weights=runs.n - 1, minlength=2).astype(int)
    tasks = audit_tasks(raw, dates, daily_changes)
    assert tasks.loc[tasks.day_index == 0, 'train_changes'].eq(0).all()
    assert tasks.loc[tasks.day_index == 1, 'train_changes'].eq(daily_changes[0]).all()
    assert tasks.loc[tasks.day_index == 1, 'train_last'].eq(dates[0]).all()
    assert not tasks.minimum_training_met.any()  # Two toy days are below 10 million.
