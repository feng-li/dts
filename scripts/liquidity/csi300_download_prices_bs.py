"""Download daily raw (unadjusted) closes for all CSI300 members ever, via baostock.

Universe: every code appearing in data/csi300/csi300_membership.csv.
Prices are needed by scripts/liquidity/csi300_backprop_weights.py (price-back-propagated
historical weights). Raw close (adjustflag=3) matches index cap weighting:
splits/bonus shares move price and share count oppositely; cash dividends drop
out of the price-return index.

Usage
-----
  python scripts/liquidity/csi300_download_prices_bs.py [--out /tmp/opencode/prices]

Output: data/csi300/raw/prices/{code}.csv (date,close), dates dashed YYYY-MM-DD.
Requires: pip install baostock  (free, no registration)
"""
import argparse
import csv
import time
from pathlib import Path

import baostock as bs

ROOT = Path(__file__).resolve().parents[2]
MEMBERSHIP = ROOT / "data" / "csi300" / "csi300_membership.csv"


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    codes = sorted({r["code"] for r in csv.DictReader(open(MEMBERSHIP, encoding="utf-8-sig"))})
    print(f"{len(codes)} codes", flush=True)

    assert bs.login().error_code == "0"
    t0, done, fails = time.time(), 0, []
    for i, code in enumerate(codes):
        p = out / f"{code}.csv"
        if p.exists() and p.stat().st_size > 100:
            done += 1
            continue
        sym = ("sh." if code.startswith("6") else "sz.") + code
        for attempt in range(3):
            rs = bs.query_history_k_data_plus(sym, "date,close", start_date="2017-12-01",
                                              end_date="2026-09-29", frequency="d",
                                              adjustflag="3")  # 3 = 不复权
            if rs.error_code != "0":
                print(f"  {code} err {rs.error_code}, retry", flush=True)
                time.sleep(3)
                continue
            rows = []
            while rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                fails.append(code)
                print(f"[{i+1}] {code} EMPTY", flush=True)
                break
            with open(p, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["date", "close"])
                w.writerows(rows)
            done += 1
            if done % 50 == 0:
                print(f"[{done}/{len(codes)}] {time.time()-t0:.0f}s", flush=True)
            break
        else:
            fails.append(code)

    bs.logout()
    print(f"DONE {done}/{len(codes)} in {time.time()-t0:.0f}s; empty/failed: {fails}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "csi300" / "raw" / "prices")
    main(ap.parse_args().out)
