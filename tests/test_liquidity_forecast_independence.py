"""Forecasting uses supplied data without any construction checkout or bridge."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def test_forecasting_runs_with_only_dts_sources_and_supplied_data(tmp_path):
    checkout = tmp_path / 'dts'
    ignore = shutil.ignore_patterns('__pycache__', '.ipynb_checkpoints')
    shutil.copytree(ROOT / 'dts', checkout / 'dts', ignore=ignore)
    scripts = checkout / 'scripts/liquidity'
    scripts.mkdir(parents=True)
    shutil.copyfile(ROOT / 'scripts/__init__.py', scripts.parent / '__init__.py')
    construction = {'_construction.py', 'run_construction.py',
                    'build_all_csi300_liquidity.py', 'build_csi300_hf_liquidity.py',
                    'prepare_csi300_dc.py'}
    for source in (ROOT / 'scripts/liquidity').iterdir():
        if source.suffix in {'.py', '.json'} and source.name not in construction:
            shutil.copyfile(source, scripts / source.name)
    data = checkout / 'given_data'
    data.mkdir()
    values = np.vstack([np.linspace(10., 12., 14220), np.linspace(11., 13., 14220)])
    np.save(data / 'spread.npy', values)
    dates = ['20230103', '20230104']
    (data / 'input_manifest.json').write_text(json.dumps(dict(
        dates=dates, shape=[2, 14220], target='liq_available_vw_bps',
        size_spec='notional_1e+06')))
    pd.DataFrame([
        dict(day_index=i, date=day, session=session, start=start, stop=stop, n=stop-start)
        for i, day in enumerate(dates)
        for session, start, stop in [('am', 0, 7200), ('pm', 7200, 14220)]
    ]).to_csv(data / 'contiguous_runs.csv', index=False)
    env = os.environ.copy()
    env.update(PYTHONPATH=str(checkout), HFL_REPO=str(tmp_path / 'absent'),
               NUMBA_CACHE_DIR=str(tmp_path / 'numba'),
               PYTHONPYCACHEPREFIX=str(tmp_path / 'pycache'),
               MPLCONFIGDIR=str(tmp_path / 'matplotlib'),
               OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    program = '''
import importlib, importlib.abc, json, os, sys
from pathlib import Path
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, {min(os.sched_getaffinity(0))})
root = Path.cwd()
assert not (root / 'HighFrequencyLiquidity').exists()
assert not (root / 'scripts/liquidity/_construction.py').exists()
class NoConstruction(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {'scripts.liquidity._construction', 'scripts.liquidity._paths'} or fullname.startswith('HighFrequencyLiquidity'):
            raise AssertionError('Forecast imported construction: ' + fullname)
sys.meta_path.insert(0, NoConstruction())
names = ['plan_csi300_boundary_forecasts', 'compare_csi300_boundary_random_days',
         'compare_csi300_boundary_spark', 'compare_csi300_boundary_arima110',
         'compare_csi300_boundary_arima101']
modules = {name: importlib.import_module('scripts.liquidity.' + name) for name in names}
assert all(Path(module.__file__).is_relative_to(root) for module in modules.values())
assert Path(importlib.import_module('dts').__file__).is_relative_to(root)
spark = modules['compare_csi300_boundary_spark'].parse_arguments(['--groups', '20', '--chains', '1'])
assert spark.groups == 20 and spark.chains == 1
import numpy as np
import pandas as pd
data = root / 'given_data'
meta = json.loads((data / 'input_manifest.json').read_text())
raw = np.load(data / 'spread.npy')
runs = pd.read_csv(data / 'contiguous_runs.csv')
changes = np.bincount(runs.day_index, weights=runs.n-1, minlength=len(raw)).astype(int)
tasks = modules['plan_csi300_boundary_forecasts'].audit_tasks(raw, meta['dates'], changes)
assert len(tasks) == 28
assert tasks.loc[tasks.day_index == 1, 'train_changes'].eq(14218).all()
assert not tasks.minimum_training_met.any()
daily = modules['compare_csi300_boundary_random_days']
weights, covariance, chol = daily.increment_kernel(np.array([-.2, -5., np.log(.3), 0.]), 59, 60)
history = np.diff(np.log(raw[1, :60]))
prediction = np.exp(np.log(raw[1, 59]) + np.cumsum(weights @ history))
assert prediction.shape == (60,) and np.isfinite(prediction).all() and (prediction > 0).all()
assert covariance.shape == (60, 60) and chol.shape == (60, 60)
print(json.dumps(dict(submodule_present=False, forecast_modules=len(modules),
                      planned_tasks=len(tasks), forecast_seconds=len(prediction))))
'''
    result = subprocess.run([sys.executable, '-c', program], cwd=checkout, env=env,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert json.loads(result.stdout) == dict(submodule_present=False, forecast_modules=5,
                                            planned_tasks=28, forecast_seconds=60)
