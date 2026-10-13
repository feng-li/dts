"""Download CSI (中证指数) index-adjustment announcements and attachments.

Data flow
---------
1. POST /announcement/queryAnnouncementByVo  -> list index announcements (paged)
2. GET  /announcement/queryAnnouncementById  -> detail per announcement
   (title/content/enclosureList; content mentions of 沪深300 mark relevance)
3. enclosureList[].fileUrl (oss-ch.csindex.com.cn) -> attachment PDF/XLSX

Raw cache layout (consumed by scripts/liquidity/csi300_build_history.py):
  data/csi300/raw/details/{id}.json
  data/csi300/raw/attachments/{publishDate}_{id}_{fileName}

WAF notes (Aliyun, attack-page 403)
-----------------------------------
- the GET detail endpoint is the sensitive one: keep >=0.6s pacing, warm the
  session cookie first, and back off 60-300s on 403
- the POST list endpoint and the oss-ch attachment host are not throttled
- burst behaviour bans the IP for ~1h; the poll/backoff below survives it

Usage
-----
  python scripts/liquidity/csi300_fetch_announcements.py            # fetch known CSI300 ids
  python scripts/liquidity/csi300_fetch_announcements.py --scan     # full scan since 2017-10
  python scripts/liquidity/csi300_fetch_announcements.py --scan --since 2023-01-01
"""
import argparse
import json
import re
import time
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "csi300" / "raw"
DETAILS = RAW / "details"
ATTACH = RAW / "attachments"

BASE = "https://www.csindex.com.cn/csindex-home"
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36")
HEADERS = {"User-Agent": UA, "Referer": "https://www.csindex.com.cn/",
           "Content-Type": "application/json;charset=UTF-8"}

# announcements known/likely to affect CSI300 membership (see build_history)
CSI300_IDS = [12590, 11518, 11859, 11379, 11529, 11429, 11514, 12470, 13888,
              14223, 14497, 14796, 15044, 15267, 15471, 15546, 15690, 1006022,
              3006000, 3006137, 3006227]

# conservative title pre-filter for scan mode: clearly other index families
EXCLUDE = re.compile(
    r"三板|北证|新三板|创新成指|中华交易服务|港股通|香港|港股|跨境|全球|海外|美元|"
    r"债券|国债|信用|商品|期权|基金|REITs|离岸|东盟|欧洲|日经|纳斯达克|道琼斯|标普")

session = requests.Session()
session.headers.update(HEADERS)


def warmup():
    session.get("https://www.csindex.com.cn/", timeout=30)
    time.sleep(1)


def get_json(url, **kw):
    for attempt in range(8):
        try:
            r = session.get(url, timeout=30, **kw)
            if r.status_code == 403:
                wait = min(300, 60 * (attempt + 1))
                print(f"  403 (WAF) -> sleep {wait}s", flush=True)
                time.sleep(wait)
                warmup()
                continue
            return r.json()
        except (ValueError, requests.RequestException) as e:
            wait = 15 * (attempt + 1)
            print(f"  {type(e).__name__} -> sleep {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"failed: {url}")


def list_announcements(since="2017-10-01"):
    rows, page = [], 1
    while page < 100:
        payload = {"lang": "cn", "page": {"key": "", "page": str(page), "rows": "100"},
                   "classList": ["index"], "indexList": [], "relatedTopics": [],
                   "typeList": ["announcement"]}
        d = session.post(f"{BASE}/announcement/queryAnnouncementByVo",
                         json=payload, timeout=30).json()
        batch = d.get("data") or []
        if isinstance(batch, dict):
            batch = batch.get("dataList") or []
        if not batch:
            break
        rows += batch
        oldest = batch[-1]["publishDate"]
        print(f"page {page}: +{len(batch)} oldest={oldest}", flush=True)
        if oldest < since:
            break
        page += 1
        time.sleep(0.5)
    return rows


def fetch_detail(ann_id):
    f = DETAILS / f"{ann_id}.json"
    if f.exists():
        return json.load(open(f))
    d = get_json(f"{BASE}/announcement/queryAnnouncementById", params={"id": ann_id})
    data = d.get("data") or {}
    f.write_text(json.dumps(data, ensure_ascii=False))
    time.sleep(0.6)
    return data


def download_attachments(det):
    saved = []
    for e in det.get("enclosureList") or []:
        fn = e["fileName"].replace("附件：", "")
        local = ATTACH / f"{det['publishDate']}_{det['id']}_{fn}"
        if local.exists() and local.stat().st_size > 0:
            saved.append(local.name)
            continue
        for attempt in range(4):
            try:
                r = requests.get(e["fileUrl"], headers={"User-Agent": UA}, timeout=60)
                local.write_bytes(r.content)
                saved.append(local.name)
                print(f"   attachment: {local.name} ({len(r.content)}B)", flush=True)
                break
            except Exception as ex:
                print(f"   retry {local.name}: {type(ex).__name__}", flush=True)
                time.sleep(5)
        time.sleep(0.5)
    return saved


def scan_mode(since):
    warmup()
    rows = list_announcements(since)
    adj = [x for x in rows if x["theme"] == "指数调样" and x["publishDate"] >= since
           and not EXCLUDE.search(x["title"])]
    print(f"调整公告 {len(adj)} 条需查详情", flush=True)
    hits = []
    for i, x in enumerate(adj):
        det = fetch_detail(x["id"])
        text = re.sub(r"<[^>]+>", "", det.get("content") or "")
        if re.search(r"沪深300|000300", det.get("title") or "") or "沪深300" in text:
            hits.append(det)
            enc = download_attachments(det)
            print(f"HIT {det['publishDate']} id={det['id']} {det['title'][:48]} 附件={enc}", flush=True)
        if (i + 1) % 50 == 0:
            print(f"... {i+1}/{len(adj)} fetched, hits={len(hits)}", flush=True)
    print(f"涉及沪深300: {len(hits)} 条", flush=True)
    return hits


def ids_mode(ids):
    warmup()
    for ann_id in ids:
        det = fetch_detail(ann_id)
        print(f"{det.get('publishDate')} id={ann_id} {det.get('title','?')[:48]}", flush=True)
        download_attachments(det)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan", action="store_true", help="full scan for CSI300-affecting announcements")
    ap.add_argument("--since", default="2017-10-01", help="scan start date (scan mode)")
    ap.add_argument("--ids", nargs="*", type=int, default=None, help="explicit announcement ids")
    args = ap.parse_args()
    DETAILS.mkdir(parents=True, exist_ok=True)
    ATTACH.mkdir(parents=True, exist_ok=True)
    if args.scan:
        scan_mode(args.since)
    else:
        ids_mode(args.ids if args.ids else CSI300_IDS)


if __name__ == "__main__":
    main()
