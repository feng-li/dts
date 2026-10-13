#!/usr/bin/env python3
"""Summarize saved CSI liquidity panels without replaying or changing inputs."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    args = p.parse_args(); root = args.directory
    rows = []; shapes = []; states = Counter(); violations = pairs = 0
    for path in tqdm(sorted((root/'index').glob('*.parquet')), desc='Summarize index'):
        a = pd.read_parquet(path)
        for (session, spec), g in a.groupby(['session', 'size_spec']):
            valid = g.liq_raw_vw.notna()
            raw = g.loc[valid, 'liq_raw_vw']
            share = g.loc[valid, 'top_contribution'].div(raw.replace(0, np.nan))
            rows.append(dict(date=path.stem, session=session, size_spec=spec, rows=len(g),
                available_count=int(g.liq_available_vw_bps.notna().sum()) if 'liq_available_vw_bps' in g else 0,
                mean_available_bps=g.liq_available_vw_bps.mean() if 'liq_available_vw_bps' in g else np.nan,
                capped_count=int(g.liq_capped_vw_bps.notna().sum()) if 'liq_capped_vw_bps' in g else 0,
                raw_count=int(valid.sum()), stress_count=int(g.liq_stress.notna().sum()),
                release_count=int(g.is_release_eligible.sum()), mean_coverage=g.weight_coverage.mean(),
                min_coverage=g.weight_coverage.min(), mean_raw_bps=raw.mean(),
                mean_ew_bps=g.liq_raw_ew.mean(), mean_sell_bps=g.liq_sell.mean(), mean_buy_bps=g.liq_buy.mean(),
                mean_sh_contribution=g.sh_contribution.mean(), mean_sz_contribution=g.sz_contribution.mean(),
                mean_weight_hhi=g.loc[valid,'weight_hhi'].mean(), max_single_stock_cost_share=share.max(),
                mean_insuff_weight=g.insuff_depth_weight.mean()))
        a['clock_minute'] = a.t_ms//60000
        shapes.append(a.groupby(['session','size_spec','clock_minute'])[
            ['liq_raw_vw','liq_raw_ew','liq_stress','weight_coverage']].mean().reset_index().assign(date=path.stem))
    if not rows:
        raise ValueError('No index files to summarize')
    # Each published index has a complete daily audit snapshot. Besides avoiding
    # hundreds of thousands of small reads, this excludes obsolete/orphan stock
    # sidecars that did not participate in the saved index's construction.
    for day in sorted({row['date'] for row in rows}):
        for audit in json.loads((root/'audits'/f'{day}.json').read_text()):
            states.update(audit.get('state_counts',{}))
            violations += audit.get('size_monotonicity_violations',0)
            pairs += audit.get('size_monotonicity_pairs',0)
    coverage_files = sorted((root/'coverage').glob('*.csv'))
    if coverage_files:
        pd.concat([pd.read_csv(f) for f in coverage_files], ignore_index=True).to_csv(
            root/'coverage_summary.csv', index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(root/'diagnostics.csv',index=False)
    pd.concat(shapes,ignore_index=True).to_parquet(root/'intraday_diagnostics.parquet',index=False)
    qa = dict(days=summary.date.nunique(), index_rows=int(summary.rows.sum()),
              valid_available_rows=int(summary.available_count.sum()),
              valid_capped_rows=int(summary.capped_count.sum()),
              valid_raw_rows=int(summary.raw_count.sum()), valid_stress_rows=int(summary.stress_count.sum()),
              release_rows=int(summary.release_count.sum()), stock_size_comparisons=pairs,
              stock_size_violations=violations, stock_state_counts=dict(states),
              note='Counts include separate size scales. Provisional rows are not a released CSI 300 index.')
    (root/'diagnostics.json').write_text(json.dumps(qa,indent=2))
    print(json.dumps(qa,indent=2))


if __name__ == '__main__':
    main()
