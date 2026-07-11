"""build the dense (sub-Mc3) Marmara catalogue from the AFAD bulletin
(data/external/dense/afad_raw.csv) into data/external/marmara_dense_catalog.csv
[datetime_utc, latitude, longitude, mag], for the DenseCatalogSource.

Steps (all logged to results/dense_build_report.json):
  1. homogenize AFAD magnitudes to proxy-Mw (marmara.catalog.mag_to_mw), consistent
     with the KOERI catalogue.
  2. internal dedup (±10 s, ±0.1°, ±0.5 mag), same thresholds as catalog.dedupe.
  3. quarry-blast filter (catalog.py has NO blast filter), so we apply a documented,
     conservative heuristic: drop events that are simultaneously local-daytime
     (UTC+3 hour in [6,18)), shallow (< 5 km) and small (raw M < 2.5). Reported.
  4. per-year completeness Mc (marmara.catalog.compute_mc, max-curvature +0.2); keep
     only events with M >= Mc(year), the design.s "M >= Mc(year)" requirement.
  5. cross-check overlap vs KOERI (±10 s, ±15 km) for transparency (NOT removed; the
     dense catalogue is used standalone by DenseCatalogSource).

NEVER fabricates events. Run: "<venv>/bin/python" -m marmara.build_dense
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from marmara.paths import RESULTS, DATA, DATA_EXTERNAL
from marmara.catalog import mag_to_mw, compute_mc

RAW = DATA_EXTERNAL / "dense" / "afad_raw.csv"
OUT_CSV = DATA_EXTERNAL / "marmara_dense_catalog.csv"
REPORT = RESULTS / "dense_build_report.json"
KM_PER_DEG = 111.0


def main():
    if not RAW.exists():
        print(f"missing {RAW}; run: python -m marmara.sources.fetch_data dense_afad"); return
    d = pd.read_csv(RAW)
    d["datetime_utc"] = pd.to_datetime(d["datetime_utc"], format="mixed", errors="coerce")
    d = d.dropna(subset=["datetime_utc", "latitude", "longitude", "mag"]).reset_index(drop=True)
    n_raw = len(d)
    d["mag_w"] = mag_to_mw(d["mag"].to_numpy(), d["mag_type"].astype("string").to_numpy())

    # 2. internal dedup (time-sorted)
    d = d.sort_values("datetime_utc").reset_index(drop=True)
    t_s = (d["datetime_utc"].values.astype("datetime64[ns]") - d["datetime_utc"].values.astype("datetime64[ns]")[0]) / np.timedelta64(1, "s")
    lat, lon, mw = d.latitude.to_numpy(), d.longitude.to_numpy(), d.mag_w.to_numpy()
    keep = np.ones(len(d), bool); win = []
    for i in range(len(d)):
        while win and (t_s[i] - t_s[win[0]]) > 10.0:
            win.pop(0)
        if any(abs(lat[i] - lat[j]) <= 0.1 and abs(lon[i] - lon[j]) <= 0.1 and abs(mw[i] - mw[j]) <= 0.5 for j in win):
            keep[i] = False
        else:
            win.append(i)
    d = d[keep].reset_index(drop=True); n_dedup = len(d)

    # 3. quarry-blast heuristic
    local_hr = (d["datetime_utc"].dt.hour + 3) % 24            # Turkey UTC+3
    depth = pd.to_numeric(d.get("depth_km"), errors="coerce").fillna(99.0)
    blast = (local_hr >= 6) & (local_hr < 18) & (depth < 5.0) & (d["mag"] < 2.5)
    d = d[~blast].reset_index(drop=True); n_blast = int(blast.sum())

    # 4. per-year Mc
    d["year"] = d["datetime_utc"].dt.year
    mc_by_year = {}
    kept = []
    for yr, g in d.groupby("year"):
        if len(g) < 100:
            mc = 2.5
        else:
            mc, _ = compute_mc(g["mag_w"].to_numpy())
        mc_by_year[int(yr)] = float(mc)
        kept.append(g[g["mag_w"] >= mc])
    dc = pd.concat(kept).sort_values("datetime_utc").reset_index(drop=True)

    # 5. overlap vs KOERI (transparency only)
    koeri = pd.read_csv(RESULTS / "catalog.csv"); koeri["datetime_utc"] = pd.to_datetime(koeri["datetime_utc"])
    kt = koeri["datetime_utc"].values.astype("datetime64[s]").astype(np.int64)
    klat, klon = koeri.latitude.to_numpy(), koeri.longitude.to_numpy()
    dt = dc["datetime_utc"].values.astype("datetime64[s]").astype(np.int64)
    order = np.argsort(kt)
    kt_s = kt[order]; matched = 0
    dlat, dlon = dc.latitude.to_numpy(), dc.longitude.to_numpy()
    coslat = np.cos(np.radians(40.7))
    for i in range(len(dc)):
        lo = np.searchsorted(kt_s, dt[i] - 10); hi = np.searchsorted(kt_s, dt[i] + 10)
        for j in order[lo:hi]:
            if abs(dlat[i] - klat[j]) * KM_PER_DEG <= 15 and abs(dlon[i] - klon[j]) * KM_PER_DEG * coslat <= 15:
                matched += 1; break

    out = dc[["datetime_utc", "latitude", "longitude", "mag_w"]].rename(columns={"mag_w": "mag"})
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    rep = {"n_raw_afad": n_raw, "n_after_internal_dedup": n_dedup,
           "n_blasts_removed": n_blast, "n_final_dense": int(len(out)),
           "mc_by_year": mc_by_year, "date_range": [str(out.datetime_utc.min()), str(out.datetime_utc.max())],
           "koeri_overlap_pm10s_15km": int(matched),
           "afad_unique_below_koeri": int(len(out) - matched),
           "note": "dense = AFAD (M>=1.0) homogenized to Mw, internally deduped, "
                   "daytime-shallow-small blast heuristic applied (catalog.py has no "
                   "blast filter), per-year Mc completeness. Used standalone by "
                   "DenseCatalogSource; KOERI overlap reported, not removed."}
    json.dump(rep, open(REPORT, "w"), indent=2)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
