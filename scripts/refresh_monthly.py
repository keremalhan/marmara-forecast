"""Incremental monthly catalog refresh (network-dependent — runs unsandboxed under
launchd). Fetches KOERI monthly preliminary XML lists, appends genuinely-new events
to koeri_events.csv, and rebuilds the catalog. Best-effort: on any failure it exits
non-zero and leaves the existing catalog untouched, so the prospective run still
scores closed forecasts.

DATA SOURCE (fixed 2026-07-06): the zeqdb search endpoint became a JS/AJAX app shell
and is no longer scrapable, but the monthly preliminary lists remain plain XML at
  http://udim.koeri.boun.edu.tr/zeqmap/xmlt/YYYYMM.xml
(entries: <earhquake name="YYYY.MM.DD HH:MM:SS" lokasyon=".." lat=".." lng=".."
mag=".." Depth=".." />). Times are LOCAL Turkey time (UTC+3, fixed since 2016 —
empirically proven against UTC anchors in fetch_manifest.json); magnitudes are
single untyped rapid solutions (treated as xM-unknown, like all preliminary rows).
Monthly rows carry no KOERI event code, so novelty is decided by time+space
proximity (±10 s AND ±0.1°) against the existing catalog, plus a synthetic
event_code for idempotence across runs.

NOTE for a future operator: rows fetched this way are PRELIMINARY. Every ~2-3
months, reviewed zeqdb data should ideally replace the preliminary tail — if the
zeqdb app regains a scrapable endpoint, or via AFAD/ISC. Accumulation of the
prospective track record only needs the catalog to ADVANCE, which this provides.

Run:  "<venv>/bin/python3" Raw_Data/koeri_update_20260705/refresh_monthly.py
"""
import os
import re
import subprocess
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)  # repo root
RAW = os.path.join(ROOT, "data", "raw")
CSV = os.path.join(ROOT, "data", "koeri_events.csv")
XML_BASE = "http://udim.koeri.boun.edu.tr/zeqmap/xmlt/{ym}.xml"
BOX = dict(lat_min=39.0, lat_max=42.5, lon_min=25.0, lon_max=31.5)
UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) research-catalog-fetch"

ENTRY_RE = re.compile(
    r'<earhquake\s+name="([\d.]+ [\d:]+)"\s+lokasyon="([^"]*)"\s+'
    r'lat="([\d.\-]+)"\s+lng="([\d.\-]+)"\s+mag="([\d.\-]*)"\s+Depth="([\d.\-]*)"',
    re.IGNORECASE)


def month_list(d1, d2):
    """Inclusive list of YYYYMM strings covering [d1, d2]."""
    out, y, m = [], d1.year, d1.month
    while (y, m) <= (d2.year, d2.month):
        out.append(f"{y}{m:02d}")
        m += 1
        if m == 13:
            y, m = y + 1, 1
    return out


def fetch_month(ym):
    dst = os.path.join(RAW, f"refresh_{ym}.xml")
    rc = subprocess.run(
        ["curl", "-s", "--max-time", "120", "-A", UA, XML_BASE.format(ym=ym), "-o", dst],
        capture_output=True).returncode
    if rc != 0 or not os.path.exists(dst) or os.path.getsize(dst) < 100:
        return None
    return open(dst, "rb").read().decode("iso-8859-9", errors="replace")


def parse_xml(txt, ym):
    out = []
    for m in ENTRY_RE.finditer(txt):
        try:
            dt_local = pd.Timestamp(m.group(1).replace(".", "-"))
            lat, lon = float(m.group(3)), float(m.group(4))
            mag = float(m.group(5)) if m.group(5) else None
            dep = float(m.group(6)) if m.group(6) else None
        except Exception:
            continue
        if mag is None or not (BOX["lat_min"] <= lat <= BOX["lat_max"]
                               and BOX["lon_min"] <= lon <= BOX["lon_max"]):
            continue
        dt_utc = dt_local - pd.Timedelta(hours=3)  # TRT (UTC+3, fixed post-2016) -> UTC
        out.append(dict(
            datetime_utc=dt_utc, latitude=lat, longitude=lon, depth_km=dep,
            mag_pref=mag, mag_type="xM-unknown",
            md=None, ml=None, mw=None, ms=None, mb=None,
            location_name=m.group(2).strip(),
            catalog_source="monthly_preliminary", source_file=f"raw/refresh_{ym}.xml",
            event_code=f"mxml_{dt_utc.strftime('%Y%m%d%H%M%S')}_{lat:.2f}_{lon:.2f}"))
    return out


def drop_already_known(nd, cur, overlap_start):
    """Novelty test for code-less preliminary rows: an incoming row is a duplicate
    if any existing event lies within +/-10 s AND +/-0.1 deg of it."""
    tail = cur[cur["datetime_utc"] >= pd.Timestamp(overlap_start) - pd.Timedelta(days=2)]
    if len(tail) == 0:
        return nd
    t_ex = tail["datetime_utc"].values.astype("datetime64[s]").astype(np.int64)
    la_ex, lo_ex = tail["latitude"].values, tail["longitude"].values
    keep = []
    for _, r in nd.iterrows():
        t = np.int64(pd.Timestamp(r["datetime_utc"]).timestamp())
        near = (np.abs(t_ex - t) <= 10) & (np.abs(la_ex - r["latitude"]) <= 0.1) \
               & (np.abs(lo_ex - r["longitude"]) <= 0.1)
        keep.append(not bool(near.any()))
    return nd[np.array(keep)]


def main():
    cur = pd.read_csv(CSV)
    cur["datetime_utc"] = pd.to_datetime(cur["datetime_utc"])
    start = (cur["datetime_utc"].max() - timedelta(days=5)).date()
    end = date.today()
    os.makedirs(RAW, exist_ok=True)

    rows = []
    for ym in month_list(start, end):
        txt = fetch_month(ym)
        if txt is None:
            print(f"refresh: month {ym} fetch failed (skipped)")
            continue
        got = parse_xml(txt, ym)
        print(f"refresh: month {ym}: {len(got)} in-box rows parsed")
        rows.extend(got)
    if not rows:
        print("refresh: no parseable rows from any month (endpoint down?)")
        sys.exit(1)

    nd = pd.DataFrame(rows)
    nd["datetime_utc"] = pd.to_datetime(nd["datetime_utc"])
    nd = nd[nd["datetime_utc"] > cur["datetime_utc"].max() - pd.Timedelta(days=5)]
    nd = nd.drop_duplicates("event_code", keep="first")
    known_codes = set(cur["event_code"].dropna().astype(str))
    nd = nd[~nd["event_code"].astype(str).isin(known_codes)]
    nd = drop_already_known(nd, cur, start)
    if len(nd) == 0:
        print("refresh: no new events (catalog already current)")
        sys.exit(0)

    nd = nd.dropna(axis=1, how="all")  # silence pandas concat FutureWarning (all-NA md/ml/... cols)
    out = pd.concat([cur, nd], ignore_index=True).sort_values("datetime_utc")
    out.to_csv(CSV, index=False)
    r = subprocess.run([sys.executable, "-m", "marmara.catalog"], cwd=ROOT,
                       env={**os.environ, "PYTHONPATH": os.path.join(ROOT,"src"), "MARMARA_ROOT": ROOT},
                       capture_output=True, text=True)
    print(f"refresh: +{len(nd)} new events, new end {out['datetime_utc'].max()}; "
          f"catalog rebuild rc={r.returncode}")
    if r.returncode != 0:
        print(r.stdout[-1500:])
        print(r.stderr[-1500:])
    sys.exit(0 if r.returncode == 0 else 1)


if __name__ == "__main__":
    main()
