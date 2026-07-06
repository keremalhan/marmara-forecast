#!/usr/bin/env python3
"""Bulk zeqdb fetcher: half-year chunks 2003-2025 with recursive auto-split.
Uses curl via subprocess (plain HTTP). Polite: >=1.5s between requests.
Writes raw responses to raw/ and a chunk log JSON.
"""
import subprocess, time, re, json, os, sys
from datetime import date, timedelta

OUT = os.environ.get("KOERI_RAW_DIR", os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(OUT, "raw")   # gitignored; override with KOERI_RAW_DIR
BASE = "http://www.koeri.boun.edu.tr/sismo/zeqdb/submitRecSearchT.asp"
BOX = dict(EnMin="39.0", EnMax="42.5", BoyMin="25.0", BoyMax="31.5",
           MAGMin="1.0", MAGMax="9.0", DerMin="0", DerMax="500", Tip="Hepsi")
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) research-catalog-fetch"
CAP_SUSPECT = 4000
chunk_log = []

def qurl(d1, d2, ofname):
    p = (f"bYear={d1.year}&bMont={d1.month:02d}&bDay={d1.day:02d}"
         f"&eYear={d2.year}&eMont={d2.month:02d}&eDay={d2.day:02d}")
    for k, v in BOX.items():
        p += f"&{k}={v}"
    p += f"&ofName={ofname}"
    return f"{BASE}?{p}"

def parse_response(path):
    """Return (found, rows, err). rows = list of 15-field lists."""
    try:
        txt = open(path, "rb").read().decode("cp1254", errors="replace")
    except Exception as e:
        return None, [], f"read-fail {e}"
    m = re.search(r"Bulunan:\s*(\d+)", txt)
    if not m:
        if "Liste sonu" in txt or "sonu" in txt:
            return 0, [], None if "Bulunan" not in txt else "no-count"
        return None, [], "no Bulunan marker"
    found = int(m.group(1))
    i1, i2 = txt.find("Deprem Kodu"), txt.find("Liste sonu")
    if i1 < 0 or i2 < 0:
        return found, [], "missing table markers"
    seg = re.sub(r"</?b>|<hr>|</?font[^>]*>", "", txt[i1:i2])
    rows = [p.split("\t") for p in seg.split("<br>") if re.match(r"^\d{6}\t", p)]
    rows = [r for r in rows if len(r) == 15]
    return found, rows, None

def fetch(d1, d2, depth=0):
    tag = f"{d1.isoformat()}_{d2.isoformat()}"
    fn = os.path.join(RAW, f"zeqdb_{tag}.html")
    url = qurl(d1, d2, f"{tag}.txt")
    ok = False
    for attempt in range(4):
        time.sleep(1.6 if attempt == 0 else 5 * (3 ** attempt))
        r = subprocess.run(["curl", "-s", "-m", "300", "-A", UA, url, "-o", fn,
                            "-w", "%{http_code}"], capture_output=True, text=True)
        code = r.stdout.strip()
        if r.returncode == 0 and code == "200" and os.path.getsize(fn) > 500:
            found, rows, err = parse_response(fn)
            if err is None and found is not None:
                ok = True
                break
            print(f"  [{tag}] attempt {attempt}: parse issue found={found} err={err}", flush=True)
        else:
            print(f"  [{tag}] attempt {attempt}: curl rc={r.returncode} http={code}", flush=True)
    if not ok:
        chunk_log.append(dict(chunk=tag, url=url, status="FAILED"))
        print(f"  [{tag}] FAILED after retries", flush=True)
        return
    span_days = (d2 - d1).days
    mismatch = (found != len(rows))
    dates = sorted(r[2] for r in rows) if rows else []
    early_gap = None
    if rows:
        e = dates[0].replace(".", "-")
        early_gap = (date.fromisoformat(e) - d1).days
    suspicious = mismatch or found >= CAP_SUSPECT or (early_gap is not None and early_gap > 7 and found >= CAP_SUSPECT // 2)
    entry = dict(chunk=tag, url=url, http=200, found=found, parsed=len(rows),
                 date_min=dates[0] if dates else None, date_max=dates[-1] if dates else None,
                 raw_file=os.path.basename(fn), status="ok")
    if suspicious and span_days > 20:
        entry["status"] = "split (cap-suspect)" if not mismatch else "split (count-mismatch)"
        chunk_log.append(entry)
        print(f"  [{tag}] found={found} parsed={len(rows)} -> SPLIT", flush=True)
        mid = d1 + timedelta(days=span_days // 2)
        fetch(d1, mid, depth + 1)
        fetch(mid + timedelta(days=1), d2, depth + 1)
    else:
        if mismatch:
            entry["status"] = "ok-with-mismatch"
        chunk_log.append(entry)
        print(f"  [{tag}] found={found} parsed={len(rows)} span={dates[0] if dates else '-'}..{dates[-1] if dates else '-'}", flush=True)

def main():
    os.makedirs(RAW, exist_ok=True)
    for year in range(2003, 2026):
        fetch(date(year, 1, 1), date(year, 6, 30))
        fetch(date(year, 7, 1), date(year, 12, 31))
    # 2026 full-year already probed once; refetch officially into raw/ for provenance
    fetch(date(2026, 1, 1), date(2026, 7, 5))
    with open(os.path.join(OUT, "zeqdb_chunk_log.json"), "w") as f:
        json.dump(chunk_log, f, indent=1)
    total = sum(c.get("parsed", 0) for c in chunk_log if c["status"].startswith("ok"))
    print(f"DONE. chunks={len(chunk_log)} total_rows_in_ok_chunks={total}", flush=True)

if __name__ == "__main__":
    main()
