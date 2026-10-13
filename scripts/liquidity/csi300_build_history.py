"""Build CSI300 (000300) historical constituents, 2018-present, from official CSI announcements.

Method
------
1. Baseline: current constituents snapshot (EastMoney BK0500), corrected by the
   2026-09-09 special adjustment (信达证券 delisting swap), whose effective date
   is derived from the delisted stock's last trading day.
2. Every index-adjustment announcement affecting 沪深300 was fetched from the
   official CSI website (csindex-home announcement API) and its in/out lists
   parsed from attachments (xlsx/pdf) or content-embedded tables. Raw materials
   are cached in data/csi300/raw/.
3. Membership at any past date is reconstructed backwards:
   before = (after - IN) | OUT, verified |state| == 300 at every step.
4. Special adjustments effective "自退市日起" use the delisted stock's last
   trading day (from EastMoney kline) as the effective close date.

Verification
------------
- every announcement yields exactly the "更换N只" count stated in its text
- all 2215 trading days show consNumber == 300 (CSI perf API)
- weekly sampling 2018-01..today: exactly one covering state of size 300
- 宁德时代 2021-12-13 first inclusion confirmed by news (创业板 3-year rule)

Outputs (data/csi300/)
----------------------
- csi300_adjustments.csv        one row per adjustment event
- csi300_membership.csv         (code, name, valid_from, valid_until) spells
- csi300_current.csv            current 300 members
- csi300_monthly_snapshots.csv  month-end (trading day) membership, 2018-01..now
"""
import csv
import json
import pickle
import re
import sys
import warnings
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message="Workbook contains no default style")

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "csi300" / "raw"
OUT = ROOT / "data" / "csi300"

IDS = [12590, 11518, 11859, 11379, 11529, 11429, 11514, 12470, 13888, 14223, 14497,
       14796, 15044, 15267, 15471, 15546, 15690, 1006022, 3006000, 3006137, 3006227]
# lists lost in the old CMS for 12590 (2017-12 review); not needed post-2018

CODE_RE = re.compile(r"^\d{6}$")
# special adjustments effective on the delisting date of the absorbed stock:
# id -> (delisted code, its last trading day, YYYY-MM-DD)
SPECIAL = {
    15546: ("600837", "2025-02-05"),    # 海通证券, absorbed by 国泰君安
    1006022: ("601989", "2025-08-12"),  # 中国重工, absorbed by 中国船舶
    3006227: ("601059", "2026-09-14"),  # 信达证券, absorbed by 中金公司
}


# ------------------------------------------------------------------ parsers
def strip_html(html):
    return re.sub(r"\s+", "", re.sub(r"<[^>]+>", "", html or ""))


def _norm_cell(v):
    s = str(v).strip()
    if re.fullmatch(r"\d+(\.0+)?", s):
        n = int(float(s))
        if 0 < n <= 999999:
            return str(n).zfill(6)
    return s


def parse_effective_date(text):
    for pat in (
        r"于(\d{4})年(\d{1,2})月(\d{1,2})日收[市盘]后生效",
        r"自(\d{4})年(\d{1,2})月(\d{1,2})日收[市盘]后生效",
        r"决定于(\d{4})年(\d{1,2})月(\d{1,2})日(?:收[市盘]后)?(?:调整|生效)",
        r"于(\d{4})年(\d{1,2})月(\d{1,2})日(?:正式)?生效",
        r"自(\d{4})年(\d{1,2})月(\d{1,2})日起",
    ):
        m = re.search(pat, text)
        if m:
            y, mo, d = map(int, m.groups())
            return f"{y:04d}-{mo:02d}-{d:02d}"
    return None


def parse_xlsx_wide(path):
    """Wide layout: one row per index, 调出 pairs then 调入 pairs (2026 era)."""
    raw = pd.read_excel(path, header=None)
    idx = None
    for i in range(min(len(raw), 500)):
        vals = [_norm_cell(v) for v in raw.iloc[i].tolist()]
        if "000300" in vals:
            idx = i
            break
    if idx is None:
        return None, None
    out_cols, in_cols = [], []
    for h in range(idx):
        row = [str(v).strip() for v in raw.iloc[h].tolist()]
        groups = {j: v for j, v in enumerate(row) if v in ("调出", "调入")}
        if groups:
            cols = sorted(groups)
            bounds = cols + [raw.shape[1]]
            for gi, start in enumerate(cols):
                rng = list(range(start, bounds[gi + 1]))
                (out_cols if groups[start] == "调出" else in_cols).extend(rng)
            break
    if not out_cols and not in_cols:
        return None, None
    outs, ins = [], []
    for cols, acc in ((out_cols, outs), (in_cols, ins)):
        for j in cols:
            code = _norm_cell(raw.iloc[idx, j])
            if CODE_RE.match(code):
                name = str(raw.iloc[idx, j + 1]).strip() if j + 1 < raw.shape[1] else ""
                acc.append((code, name))
    return outs, ins


HEAD_RE = re.compile(r"([\u4e00-\u9fffA-Za-z0-9·]+)\s*指数样本\s*(?:股)?\s*调整名单")


def parse_pdf(path):
    """PDF with per-index sections; tables assigned to nearest heading above."""
    import pdfplumber
    outs, ins = [], []
    section = None
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            marks = []
            try:
                for hit in page.search("备选名单"):
                    marks.append((hit["top"], "备选"))
            except Exception:
                pass
            try:
                for hit in page.search(HEAD_RE, regex=True):
                    marks.append((hit["top"], hit["groups"][0]))
            except Exception:
                pass
            marks.sort(key=lambda x: x[0])
            items = [(t.bbox[1], "table", t) for t in page.find_tables()]
            items += [(top, "mark", lab) for top, lab in marks]
            items.sort(key=lambda x: x[0])
            for _, kind, obj in items:
                if kind == "mark":
                    section = obj
                    continue
                if section is None or "备选" in section:
                    continue
                if not ("沪深300" in section or "000300" in section):
                    continue
                for row in obj.extract():
                    cells = [_norm_cell(c) if c else "" for c in row]
                    while len(cells) < 4:
                        cells.append("")
                    if CODE_RE.match(cells[0]):
                        outs.append((cells[0], cells[1]))
                    if CODE_RE.match(cells[2]):
                        ins.append((cells[2], cells[3]))
    return outs, ins


def parse_content_text(html):
    """In/out lists embedded as text in the announcement body (2017-2021 era)."""
    text = strip_html(html)
    m = re.search(r"沪深300指数样本(?:股)?调整名单[:：]", text)
    if not m:
        return None, None
    seg = text[m.end():]
    stop = re.search(
        r"备选名单|(?:中证|上证|沪深|深证|国证|创业板|科创|北证)[\u4e00-\u9fffA-Za-z0-9]{0,16}指数样本(?:股)?(?:调整名单|定期调整名单)[:：]",
        seg)
    if stop:
        seg = seg[:stop.start()]
    outs, ins = [], []
    pat = re.compile(r"(\d{6})([^\d]+?)(\d{6})([^\d]+?)(?=\d{6}[^\d]+?\d{6}|$)")
    for mo in pat.finditer(seg):
        outs.append((mo.group(1), mo.group(2)))
        ins.append((mo.group(3), mo.group(4)))
    return (outs, ins) if outs else (None, None)


# ------------------------------------------------------------------ collect
def collect_adjustments():
    rows = []
    for ann_id in IDS:
        f = RAW / "details" / f"{ann_id}.json"
        det = json.load(open(f))
        title = det.get("title") or ""
        text = strip_html(det.get("content"))
        eff = parse_effective_date(text)
        outs, ins, src = None, None, None
        for e in det.get("enclosureList") or []:
            fn = e["fileName"].replace("附件：", "")
            local = RAW / "attachments" / f"{det['publishDate']}_{ann_id}_{fn}"
            if not local.exists():
                continue
            got = (parse_xlsx_wide(local) if local.suffix.lower() in (".xlsx", ".xls")
                   else parse_pdf(local) if local.suffix.lower() == ".pdf" else (None, None))
            if got and (got[0] or got[1]):
                outs, ins = got
                src = local.name
                break
        ct_out, ct_in = parse_content_text(det.get("content"))
        if ct_out is not None and (outs is None or not ins or not outs):
            outs, ins = ct_out, ct_in
            src = (src + "+" if src else "") + "content-text"
        if outs is None:
            print(f"!! no lists parsed for {ann_id} {title[:40]}")
            continue
        mswap = re.search(r"沪深300指数更换(\d+)只", text)
        n_expected = int(mswap.group(1)) if mswap else len(outs)
        assert len(outs) == n_expected and len(ins) == n_expected, \
            f"{ann_id}: out={len(outs)} in={len(ins)} expected={n_expected}"
        rows.append({"id": ann_id, "publish_date": det["publishDate"],
                     "effective_close": eff, "title": title, "source": src,
                     "out": outs, "in": ins})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ rebuild
def main():
    cal = pd.read_csv(RAW / "csi300_trading_days.csv", dtype=str)
    trading_days = sorted(cal["tradeDate"])
    assert set(cal["consNumber"].astype(float)) == {300.0}

    def next_trading_day(d, after_close=True):
        k = d.replace("-", "")
        for t in trading_days:
            if t > k or (t == k and not after_close):
                return t
        raise ValueError(d)

    adj = collect_adjustments()
    adj = adj[adj["id"] != 12590].copy()
    for rid, (code, last_trade) in SPECIAL.items():
        row = adj[adj["id"] == rid].iloc[0]
        assert code in [c for c, _ in row["out"]]
        adj.loc[adj["id"] == rid, "effective_close"] = last_trade
    def after_close(row):
        if row['id'] in SPECIAL:
            return True  # Explicit last-trading-day convention for delisting swaps.
        detail = json.loads((RAW / 'details' / f"{row['id']}.json").read_text())
        text = strip_html(detail.get('content'))
        y, month, day = map(int, row['effective_close'].split('-'))
        return bool(re.search(rf'{y}年0?{month}月0?{day}日收[市盘]后', text))

    adj["effective_from"] = [next_trading_day(r['effective_close'], after_close(r))
                             for _, r in adj.iterrows()]
    adj = adj.sort_values("effective_from").reset_index(drop=True)
    eff = adj["effective_from"].tolist()
    assert all(a < b for a, b in zip(eff, eff[1:]))

    base = pd.read_csv(RAW / "csi300_current_em.csv", dtype={"code": str})
    current = {r.code: r.name for r in base.itertuples()}
    last = adj.iloc[-1]
    assert last["id"] == 3006227
    for c, _ in last["out"]:
        current.pop(c, None)
    for c, n in last["in"]:
        current[c] = n
    assert len(current) == 300

    # backwards walk
    FIRST_FROM = "2017-12-12"
    s, until, walked = dict(current), None, []
    for i in range(len(adj) - 1, -1, -1):
        r = adj.iloc[i]
        walked.append((r["effective_from"], until, dict(s)))
        ins, outs = dict(r["in"]), dict(r["out"])
        s = {k: v for k, v in s.items() if k not in ins}
        s.update(outs)
        assert len(s) == 300, f"step {r['id']}: {len(s)}"
        until = r["effective_from"]
    walked.append((FIRST_FROM, until, dict(s)))
    states = walked[::-1]

    for d in pd.date_range("2018-01-01", "2026-09-29", freq="W"):
        k = d.strftime("%Y%m%d")
        covering = [st for vf, vu, st in states if vf.replace("-", "") <= k and (vu is None or k < vu)]
        assert len(covering) == 1 and len(covering[0]) == 300, d

    # membership spells
    open_iv, intervals = {}, []
    for vf, vu, st in states:
        for c, n in st.items():
            open_iv.setdefault(c, [n, vf])
        for c in [c for c in open_iv if c not in st]:
            n, f_ = open_iv.pop(c)
            intervals.append([c, n, f_, vf])
    for c, (n, f_) in open_iv.items():
        intervals.append([c, n, f_, ""])
    cur_spells = [iv for iv in intervals if iv[3] == ""]
    assert len(cur_spells) == 300 and {iv[0] for iv in cur_spells} == set(current)

    by_code = defaultdict(list)
    for c, n, f_, u in intervals:
        by_code[c].append((f_, u))
    for c, sp in by_code.items():
        sp.sort()
        for (f1, u1), (f2, _) in zip(sp, sp[1:]):
            assert u1 and u1.replace("-", "") <= f2.replace("-", ""), c

    # outputs
    with open(OUT / "csi300_adjustments.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "publish_date", "effective_close", "effective_from",
                    "n_out", "n_in", "out_codes", "in_codes", "title"])
        for _, r in adj.iterrows():
            w.writerow([r["id"], r["publish_date"], r["effective_close"], r["effective_from"],
                        len(r["out"]), len(r["in"]),
                        " ".join(c for c, _ in sorted(r["out"])),
                        " ".join(c for c, _ in sorted(r["in"])), r["title"]])
    with open(OUT / "csi300_membership.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["code", "name", "valid_from", "valid_until"])
        for c, n, f_, u in sorted(intervals):
            w.writerow([c, n, f_, u or ""])
    with open(OUT / "csi300_current.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["code", "name"])
        for c, n in sorted(current.items()):
            w.writerow([c, n])
    trade_set = set(trading_days)
    d, snaps = date(2018, 1, 31), []
    while d <= date(2026, 9, 29):
        k = d.strftime("%Y%m%d")
        while k not in trade_set:
            d -= timedelta(days=1)
            k = d.strftime("%Y%m%d")
        covering = [st for vf, vu, st in states if vf.replace("-", "") <= k and (vu is None or k < vu)]
        snaps += [(k, c) for c in sorted(covering[0])]
        d = (d.replace(day=28) + timedelta(days=8)).replace(day=1)
    with open(OUT / "csi300_monthly_snapshots.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["date", "code"])
        w.writerows(snaps)
    print(f"states={len(states)} intervals={len(intervals)} snapshots={len(snaps)//300} -> {OUT}")


if __name__ == "__main__":
    main()
