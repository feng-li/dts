#!/usr/bin/env python3
"""Review a completed incremental run against the preserved archive report."""
import argparse,json
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd
from tqdm import tqdm

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);a=p.parse_args();r=a.root
    out=r/'incremental_review';out.mkdir(exist_ok=True)
    s=pd.read_csv(r/'batch_status.csv',dtype={'date':str});fail=[]
    for row in s[s.status!='built'].itertuples():
        text=Path(row.log).read_text()
        reason=('no_matching_weights' if 'No admissible matching weight snapshot' in text else
                'nullable_reference_conversion' if "cannot convert to 'int64'" in text else 'other')
        fail.append(dict(date=row.date,reason=reason))
    pd.DataFrame(fail).to_csv(out/'failures.csv',index=False)
    checks=Counter();daily=[];reuse=Counter();stocks=Counter();failed_checks=Counter()
    cols=['session','size_spec','t_ms','weight_coverage','n_valid','sh_weight_coverage','sz_weight_coverage',
          'liq_available_vw_bps','liq_available_buy_bps','liq_available_sell_bps','sh_available_contribution',
          'sz_available_contribution','unknown_depth_weight','technical_invalid_weight','shortage_weight','nontrading_weight']
    files=sorted((r/'index').glob('*.parquet'))
    checks['status_index_mismatch']=len(set(s.loc[s.status=='built','date'])^{f.stem for f in files})
    for f in tqdm(files,desc='Review corrected index days'):
        day=f.stem;x=pd.read_parquet(f,columns=cols);w=pd.read_parquet(r/'audits'/f'{day}_membership.parquet')
        totals=w.groupby('exchange').weight.sum();valid=x.liq_available_vw_bps.notna()
        gate=(x.weight_coverage>=.95-1e-12)&(x.n_valid>=250)
        for ex,total in totals.items():gate &= x[ex.lower()+'_weight_coverage']/total>=.95-1e-12
        checks['gate_mismatch']+=int((gate!=valid).sum())
        checks['nonfinite_published']+=int((valid&~np.isfinite(x.liq_available_vw_bps)).sum())
        checks['negative_spreads']+=int((x.liq_available_vw_bps<0).sum())
        checks['duplicate_grid']+=int(x.duplicated(['session','size_spec','t_ms']).sum())
        checks['directional_identity']+=int((abs(x.liq_available_vw_bps-x.liq_available_buy_bps-x.liq_available_sell_bps)>1e-7).sum())
        checks['exchange_identity']+=int((abs(x.liq_available_vw_bps-x.sh_available_contribution-x.sz_available_contribution)>1e-7).sum())
        checks['coverage_identity']+=int((abs(x.weight_coverage+x.unknown_depth_weight+x.technical_invalid_weight+x.shortage_weight+x.nontrading_weight-1)>1e-8).sum())
        for size,g in x.groupby('size_spec'):
            v=g.liq_available_vw_bps
            daily.append(dict(date=day,size_spec=size,rows=len(g),valid=int(v.notna().sum()),mean=v.mean()))
            checks['bad_day_size_count']+=len(g)!=14220
        manifest=r/'daily_versions'/f'{day}.json'
        if manifest.exists():
            m=json.loads(manifest.read_text());reuse['daily_manifests']+=1
            reuse['legacy_adopted_days']+=bool(m.get('legacy_adopted'))
            reuse['recorded_stock_reused']+=m.get('stock_reused',0);reuse['recorded_stock_rebuilt']+=m.get('stock_rebuilt',0)
        if day[:4] in ['2020','2021','2022']:
            for stock in json.loads((r/'audits'/f'{day}.json').read_text()):
                board='SH' if stock['code'].startswith('6') else ('ChiNext' if stock['code'].startswith('30') else 'SZ_main')
                group=day[:4]+'_'+board
                stocks[group+'_stockdays']+=1;stocks[group+'_failed']+=stock.get('status')!='constructed'
                stocks[group+'_crossed_rows']+=stock.get('state_counts',{}).get('crossed',0)
                if stock.get('status')!='constructed':failed_checks[str(stock.get('error') or stock.get('ledger',{}).get('failed_checks') or stock.get('status'))]+=1
    d=pd.DataFrame(daily);old=pd.read_csv(r/'archive_report/daily.csv',dtype={'date':str})
    comparison=d.merge(old[['date','size_spec','valid','mean']],on=['date','size_spec'],suffixes=('_new','_old'),validate='one_to_one')
    comparison.to_csv(out/'daily_comparison.csv',index=False)
    sm=comparison.groupby('size_spec')[['valid_old','valid_new']].sum()
    sm['pct_all_dates']=sm.valid_new/(len(s)*14220)*100
    sm['days_95pct']=d.groupby('size_spec').valid.apply(lambda v:int((v>=14220*.95).sum()))
    sm.to_csv(out/'summary.csv')
    yearly=d.assign(year=d.date.str[:4]).groupby(['year','size_spec']).valid.sum().unstack();yearly.to_csv(out/'yearly_valid.csv')
    unaffected=comparison[~((comparison.date.str[:4]=='2020')|comparison.date.between('20200824','20230409'))]
    result=dict(status=s.status.value_counts().to_dict(),failures=dict(Counter(v['reason'] for v in fail)),checks=dict(checks),reuse=dict(reuse),stocks=dict(stocks),failed_stock_reasons=dict(failed_checks),
                unaffected_days=int(unaffected.date.nunique()),unaffected_valid_count_changes=int((unaffected.valid_new!=unaffected.valid_old).sum()),
                unaffected_daily_mean_changes=int((~np.isclose(unaffected.mean_new,unaffected.mean_old,rtol=1e-12,atol=1e-10,equal_nan=True)).sum()),
                scope='All saved index rows; prior daily summaries used for unaffected comparison, not bytewise equality.')
    (out/'checks.json').write_text(json.dumps(result,indent=2));print(sm.to_string());print(yearly.to_string());print(json.dumps(result,indent=2))

if __name__=='__main__':main()
