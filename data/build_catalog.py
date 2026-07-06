#!/usr/bin/env python3
"""Build koeri_events.csv from fetched zeqdb chunks + monthly preliminary XMLs.
Run with project venv python3 (needs pandas).
Outputs: koeri_events.csv, build_stats.json (feeds fetch_manifest.json).
"""
import json, re, os
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import pandas as pd

OUT = os.environ.get("KOERI_RAW_DIR", os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(OUT, "raw")   # gitignored; override with KOERI_RAW_DIR
TR = ZoneInfo("Europe/Istanbul")
BOX = dict(lat1=39.0, lat2=42.5, lon1=25.0, lon2=31.5)
stats = {}

def fnum(s):
    s = s.strip()
    if s in ("", "-.-", "0.0", "-"):
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    return v if v != 0.0 else None

# ---------------- 1) zeqdb reviewed ----------------
log = json.load(open(os.path.join(OUT, "zeqdb_chunk_log.json")))
ok_chunks = [c for c in log if c["status"].startswith("ok")]
zrows, tip_counter = [], {}
for c in ok_chunks:
    fn = c["raw_file"]
    txt = open(os.path.join(RAW, fn), "rb").read().decode("cp1254", errors="replace")
    seg = re.sub(r"</?b>|<hr>|</?font[^>]*>", "", txt[txt.find("Deprem Kodu"):txt.find("Liste sonu")])
    rows = [p.split("\t") for p in seg.split("<br>") if re.match(r"^\d{6}\t", p)]
    assert len(rows) == c["found"], f"{fn}: parse {len(rows)} != found {c['found']}"
    for r in rows:
        tip = r[13].strip()
        tip_counter[tip] = tip_counter.get(tip, 0) + 1
        if tip != "Ke":            # drop Sm (suspected explosion) & anything non-earthquake
            continue
        dt = datetime.strptime(r[2] + " " + r[3][:8], "%Y.%m.%d %H:%M:%S")  # zeqdb == UTC
        md, ml, mw, ms, mb = (fnum(r[i]) for i in (8, 9, 10, 11, 12))
        xm = fnum(r[7])
        mag_type = "xM-unknown"
        for name, val in (("Mw", mw), ("ML", ml), ("Md", md), ("Ms", ms), ("Mb", mb)):
            if val is not None and xm is not None and abs(val - xm) < 0.05:
                mag_type = name
                break
        zrows.append(dict(datetime_utc=dt, latitude=float(r[4]), longitude=float(r[5]),
                          depth_km=float(r[6]), mag_pref=xm, mag_type=mag_type,
                          md=md, ml=ml, mw=mw, ms=ms, mb=mb,
                          location_name=r[14].strip(), catalog_source="zeqdb_reviewed",
                          source_file="raw/" + fn, event_code=r[1].strip()))
z = pd.DataFrame(zrows)
stats["zeqdb_tip_distribution"] = tip_counter
stats["zeqdb_rows_ke"] = len(z)
dup_codes = z[z.duplicated("event_code", keep=False)]
stats["zeqdb_duplicate_event_codes"] = int(dup_codes["event_code"].nunique())
z = z.drop_duplicates("event_code", keep="first")
stats["zeqdb_rows_after_code_dedupe"] = len(z)
z_end = z["datetime_utc"].max()
stats["zeqdb_coverage"] = [str(z["datetime_utc"].min()), str(z_end)]

# ---------------- 2) monthly preliminary XMLs ----------------
def parse_month(fn):
    raw = open(os.path.join(RAW, fn), "rb").read().decode("iso-8859-9", errors="replace")
    evs = re.findall(r'<earhquake\s+name="([^"]+)"\s+lokasyon="([^"]*)"\s+lat="([^"]*)"'
                     r'\s+lng="([^"]*)"\s+mag="([^"]*)"\s*Depth="([^"]*)"', raw)
    out = []
    for name, lok, lat, lng, mag, dep in evs:
        try:
            dt_local = datetime.strptime(name, "%Y.%m.%d %H:%M:%S").replace(tzinfo=TR)
            dt_utc = dt_local.astimezone(timezone.utc).replace(tzinfo=None)
            out.append(dict(datetime_utc=dt_utc, latitude=float(lat), longitude=float(lng),
                            depth_km=float(dep), mag_pref=float(mag), mag_type="xM-unknown",
                            md=None, ml=None, mw=None, ms=None, mb=None,
                            location_name=lok.strip(), catalog_source="monthly_preliminary",
                            source_file="raw/" + fn, event_code=None))
        except Exception:
            pass
    return pd.DataFrame(out), len(evs)

# 2a. empirical tz cross-check: match 202603 (claimed local) vs zeqdb March-2026 (UTC)
m03, _ = parse_month("monthly_202603.xml")   # parsed AS local->utc already
zmar = z[(z.datetime_utc >= "2026-03-01") & (z.datetime_utc < "2026-04-01")]
merged = 0; offsets = []
for _, mr in m03[(m03.latitude.between(39, 42.5)) & (m03.longitude.between(25, 31.5))].head(400).iterrows():
    cand = zmar[(abs(zmar.latitude - mr.latitude) < 0.03) & (abs(zmar.longitude - mr.longitude) < 0.03)]
    if len(cand):
        d = (cand.datetime_utc - mr.datetime_utc).dt.total_seconds().abs()
        if d.min() < 120:
            offsets.append(d.min()); merged += 1
stats["tz_crosscheck_202603_vs_zeqdb"] = dict(
    matched_events=merged,
    median_abs_residual_seconds_after_localToUTC=(float(pd.Series(offsets).median()) if offsets else None),
    note="monthly XML parsed as Europe/Istanbul local; residual ~0 confirms local-time hypothesis")

mparts, m_raw_counts = [], {}
for mf in ["monthly_202603.xml", "monthly_202604.xml", "monthly_202605.xml",
           "monthly_202606.xml", "monthly_202607.xml"]:
    df, nall = parse_month(mf)
    m_raw_counts[mf] = dict(all_turkey=nall)
    df = df[(df.latitude >= BOX["lat1"]) & (df.latitude <= BOX["lat2"]) &
            (df.longitude >= BOX["lon1"]) & (df.longitude <= BOX["lon2"]) &
            (df.mag_pref >= 1.0)]
    m_raw_counts[mf]["in_box_mag1plus"] = len(df)
    mparts.append(df)
m = pd.concat(mparts, ignore_index=True)
seam_events = m[(m.datetime_utc > z_end) & (m.datetime_utc < datetime(2026, 4, 1))]
stats["seam_events_between_zeqdb_end_and_apr1"] = len(seam_events)
m = m[m.datetime_utc > z_end].copy()
# dedupe within monthly (±10s, ±0.1 deg)
m = m.sort_values("datetime_utc").reset_index(drop=True)
drop_idx = set()
for i in range(1, len(m)):
    j = i - 1
    while j >= 0 and (m.datetime_utc[i] - m.datetime_utc[j]).total_seconds() <= 10:
        if (abs(m.latitude[i] - m.latitude[j]) <= 0.1 and abs(m.longitude[i] - m.longitude[j]) <= 0.1
                and j not in drop_idx):
            drop_idx.add(i)
        j -= 1
stats["monthly_internal_dupes_dropped"] = len(drop_idx)
m = m.drop(index=list(drop_idx)).reset_index(drop=True)
stats["monthly_raw_counts"] = m_raw_counts
stats["monthly_rows_used"] = len(m)
stats["monthly_coverage_used"] = [str(m.datetime_utc.min()), str(m.datetime_utc.max())]

# cross-source dedupe (zeqdb preferred) — matters only at the seam
zt = z[z.datetime_utc > z_end - timedelta(days=2)]
cross = 0; drop_m = []
for i, mr in m.iterrows():
    cand = zt[(abs(zt.latitude - mr.latitude) <= 0.1) & (abs(zt.longitude - mr.longitude) <= 0.1)]
    if len(cand) and (cand.datetime_utc - mr.datetime_utc).dt.total_seconds().abs().min() <= 10:
        drop_m.append(i); cross += 1
m = m.drop(index=drop_m)
stats["cross_source_dupes_dropped_monthly"] = cross

# ---------------- 3) final catalog ----------------
cat = pd.concat([z, m], ignore_index=True).sort_values("datetime_utc").reset_index(drop=True)
stats["final_rows"] = len(cat)
stats["final_by_source"] = cat.catalog_source.value_counts().to_dict()

# residual near-dup scan (report only)
tt = cat.datetime_utc.astype("int64") // 10**9
near = 0
for i in range(1, len(cat)):
    j = i - 1
    while j >= 0 and tt[i] - tt[j] <= 10:
        if abs(cat.latitude[i] - cat.latitude[j]) <= 0.1 and abs(cat.longitude[i] - cat.longitude[j]) <= 0.1 \
           and (pd.isna(cat.mag_pref[i]) or pd.isna(cat.mag_pref[j]) or abs(cat.mag_pref[i] - cat.mag_pref[j]) <= 0.5):
            near += 1
        j -= 1
stats["residual_near_dup_pairs_reported_not_dropped"] = near

# ---------------- 4) verification ----------------
def anchor(dt_utc, lat, lon, tol_min=2):
    t0 = datetime.fromisoformat(dt_utc)
    w = cat[(cat.datetime_utc >= t0 - timedelta(minutes=tol_min)) &
            (cat.datetime_utc <= t0 + timedelta(minutes=tol_min)) &
            (abs(cat.latitude - lat) < 0.25) & (abs(cat.longitude - lon) < 0.35)]
    if len(w) == 0:
        return dict(found=False)
    r = w.loc[w.mag_pref.idxmax()]
    return dict(found=True, datetime_utc=str(r.datetime_utc), mag_pref=float(r.mag_pref),
                mag_type=r.mag_type, lat=float(r.latitude), lon=float(r.longitude),
                source=r.catalog_source)

stats["anchor_2025_04_23_M6.2"] = anchor("2025-04-23T09:49:00", 40.84, 28.23)
stats["anchor_2019_09_26_M5.8"] = anchor("2019-09-26T10:59:00", 40.88, 28.21)
stats["anchor_2025_10_02_M5.0"] = anchor("2025-10-02T11:55:00", 40.81, 27.94)

old = cat[(cat.longitude >= 25.6) & (cat.longitude <= 30.9) &
          (cat.latitude >= 39.6) & (cat.latitude <= 41.9) &
          (cat.mag_pref >= 2.0) & (cat.datetime_utc <= "2025-05-08 23:59:59")]
stats["oldbox_M2_through_20250508"] = dict(count=len(old), reference=15932,
                                           ratio=round(len(old) / 15932, 3))

upd = cat[cat.datetime_utc >= "2025-05-08 00:00:00"]
stats["update_window_counts_20250508_to_today"] = {
    f"M>={t}": int((upd.mag_pref >= t).sum()) for t in (3, 4, 4.5, 5, 6)}
stats["update_window_rows"] = len(upd)

# ---------------- 5) write CSV ----------------
outcsv = cat.copy()
outcsv["datetime_utc"] = outcsv.datetime_utc.dt.strftime("%Y-%m-%dT%H:%M:%S")
for c in ("md", "ml", "mw", "ms", "mb", "mag_pref"):
    outcsv[c] = outcsv[c].map(lambda v: "" if pd.isna(v) else f"{v:.1f}")
outcsv["depth_km"] = outcsv.depth_km.map(lambda v: f"{v:.1f}")
cols = ["datetime_utc", "latitude", "longitude", "depth_km", "mag_pref", "mag_type",
        "md", "ml", "mw", "ms", "mb", "location_name", "catalog_source", "source_file", "event_code"]
outcsv[cols].to_csv(os.path.join(OUT, "koeri_events.csv"), index=False, encoding="utf-8")
with open(os.path.join(OUT, "build_stats.json"), "w") as f:
    json.dump(stats, f, indent=1, ensure_ascii=False, default=str)
print(json.dumps(stats, indent=1, ensure_ascii=False, default=str))
