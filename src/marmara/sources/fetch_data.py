"""Reproducible fetchers for the three external sources (reliable/safe/easy tier).

Usage:  "<venv>/bin/python3" -m marmara.sources.fetch_data gnss|repeaters|dense
Network calls hit reputable academic sources only. Run in an environment that permits
outbound https (the fetch writes into Raw_Data/external/).
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401

from .base import DATA

# Eurasia-fixed frame (isolates Anatolian block motion relative to Eurasia, the
# tectonically meaningful reference for the North Anatolian Fault / Marmara).
NGL = "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/EU/{sta}.EU.tenv3"
# Marmara GNSS stations (NGL DataHoldings, in/near the model box, good time span).
GNSS_STATIONS = {
    "ISTA": (29.019, 41.104), "TUBI": (29.451, 40.787), "ISTN": (28.832, 40.991),
    "SARY": (27.916, 41.443), "TEKR": (27.497, 40.958), "SLEE": (29.601, 41.169),
    "IZMT": (29.951, 40.802), "BILE": (29.977, 40.142), "HEND": (30.741, 40.795),
    "HARC": (29.153, 39.678), "BURS": (29.015, 40.214), "KARB": (28.683, 41.347),
    "ESKS": (30.464, 39.746), "BAND": (27.997, 40.331),
}


def fetch_gnss():
    gdir = DATA / "gnss"; gdir.mkdir(parents=True, exist_ok=True)
    got = []
    for sta, (lon, lat) in GNSS_STATIONS.items():
        url = NGL.format(sta=sta)
        dst = gdir / f"{sta}.tenv3"
        try:
            with urllib.request.urlopen(url, timeout=90) as r:
                data = r.read()
            if len(data) < 200:
                print(f"  {sta}: empty, skip"); continue
            dst.write_bytes(data)
            n = data.count(b"\n")
            got.append({"station": sta, "lon": lon, "lat": lat})
            print(f"  {sta}: {n} epochs -> {dst.name}")
        except Exception as e:
            print(f"  {sta}: FAILED ({e})")
    pd.DataFrame(got).to_csv(gdir / "gnss_stations.csv", index=False)
    print(f"wrote {gdir/'gnss_stations.csv'} with {len(got)} stations")


# --- GNSS trajectory: auto-discover ALL Marmara-bbox stations from NGL's
# DataHoldings, fetch full daily .tenv3 (global IGS20 frame; NGL retired IGS14) +
# the master steps file. bbox and >=4y-before-2022 rule by design.
NGL_DATAHOLDINGS = "https://geodesy.unr.edu/NGLStationPages/DataHoldings.txt"
NGL_STEPS = "https://geodesy.unr.edu/NGLStationPages/steps.txt"
NGL_TENV3 = "https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/{sta}.tenv3"
V2_BBOX = dict(min_lon=26.0, max_lon=31.0, min_lat=39.5, max_lat=41.5)
V2_MIN_YEARS_BEFORE_2022 = 4.0


def _get(url, timeout=120):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def fetch_gnss_traj():
    """Build data/external/gnss_traj/: gnss_stations.csv (auto from DataHoldings),
    every station's IGS20 .tenv3, and steps.txt. Global IGS20 frame: the per-station
    trajectory fit removes the secular plate motion, so the frame is absorbed."""
    gdir = DATA / "gnss_traj"; gdir.mkdir(parents=True, exist_ok=True)
    cutoff = pd.Timestamp("2022-01-01")
    print("fetching NGL DataHoldings ...")
    (gdir / "DataHoldings.txt").write_bytes(_get(NGL_DATAHOLDINGS))
    print("fetching NGL steps.txt ...")
    (gdir / "steps.txt").write_bytes(_get(NGL_STEPS))

    rows = []
    for line in (gdir / "DataHoldings.txt").read_text().splitlines()[1:]:
        f = line.split()
        if len(f) < 11:
            continue
        try:
            sta, lat, lon = f[0], float(f[1]), float(f[2])
            dtbeg, dtend, numsol = pd.Timestamp(f[7]), pd.Timestamp(f[8]), int(f[10])
        except Exception:
            continue
        if not (V2_BBOX["min_lat"] <= lat <= V2_BBOX["max_lat"]
                and V2_BBOX["min_lon"] <= lon <= V2_BBOX["max_lon"]):
            continue
        # >=4 years of coverage BEFORE 2022-01-01
        yrs_before = (min(dtend, cutoff) - dtbeg).days / 365.25
        if dtbeg > cutoff or yrs_before < V2_MIN_YEARS_BEFORE_2022:
            continue
        rows.append({"station": sta, "lon": lon, "lat": lat,
                     "dtbeg": dtbeg.date(), "dtend": dtend.date(), "numsol": numsol})
    print(f"{len(rows)} stations in bbox with >= {V2_MIN_YEARS_BEFORE_2022} yr before 2022")

    got = []
    for r in rows:
        dst = gdir / f"{r['station']}.tenv3"
        try:
            data = _get(NGL_TENV3.format(sta=r["station"]))
            if len(data) < 200 or data[:15].lstrip().startswith(b"<"):
                print(f"  {r['station']}: no IGS20 tenv3, skip"); continue
            dst.write_bytes(data)
            got.append(r)
            print(f"  {r['station']}: {data.count(chr(10).encode())} epochs")
        except Exception as e:
            print(f"  {r['station']}: FAILED ({e})")
    pd.DataFrame(got).to_csv(gdir / "gnss_stations.csv", index=False)
    print(f"wrote {gdir/'gnss_stations.csv'} with {len(got)} stations + steps.txt")


def info_repeaters():
    print("Repeating-earthquake catalog has no single-command open download; safest sources:")
    print("  * Schmittbuhl, Karabulut, Lengline, Bouchon 2016 GRL 'Long-lasting seismic")
    print("    repeaters in the Central Basin of the Main Marmara Fault' (supp. table).")
    print("  * Bohnhoff et al. 2017 (Marmara repeaters / creep).")
    print("  Save as Raw_Data/external/marmara_repeaters.csv [datetime,lat,lon,family_id].")


def info_dense():
    print("Dense re-picked catalog (Mc~1.5), safest sources:")
    print("  * ESM/AFAD or KOERI enhanced bulletin; or a published EQTransformer/PhaseNet")
    print("    Marmara catalog on Zenodo (search 'Marmara deep learning catalog').")
    print("  Save as Raw_Data/external/marmara_dense_catalog.csv")
    print("    [datetime_utc,latitude,longitude,mag].")


# --- AFAD extended bulletin (Mmin 1.0), the dense sub-Mc3 catalogue.
AFAD_API = "https://deprem.afad.gov.tr/apiv2/event/filter"
DENSE_BBOX = dict(min_lon=25.6, max_lon=30.9, min_lat=39.6, max_lat=41.9)  # model box
DENSE_MMIN = 1.0


def fetch_dense_afad(start="2008-01-01", end="2026-07-01"):
    """Fetch the AFAD event bulletin (M>=1.0) for the model box, 2008->present, in
    monthly chunks (safe against per-query caps), into data/external/dense/afad_raw.csv.
    Deterministic given the same server state; the human can re-run to refresh."""
    import urllib.parse
    gdir = DATA / "dense"; gdir.mkdir(parents=True, exist_ok=True)
    b = DENSE_BBOX
    cur, stop = pd.Timestamp(start), pd.Timestamp(end)
    rows, capped = [], []
    while cur < stop:
        nxt = min(cur + pd.DateOffset(months=1), stop)
        qs = urllib.parse.urlencode({
            "start": f"{cur:%Y-%m-%d} 00:00:00", "end": f"{nxt:%Y-%m-%d} 00:00:00",
            "minlat": b["min_lat"], "maxlat": b["max_lat"],
            "minlon": b["min_lon"], "maxlon": b["max_lon"],
            "minmag": DENSE_MMIN, "format": "json"})
        try:
            data = json.loads(_get(f"{AFAD_API}?{qs}", timeout=90))
        except Exception as e:
            print(f"  {cur:%Y-%m}: FAILED ({e})"); cur = nxt; continue
        n = len(data) if isinstance(data, list) else 0
        if n >= 10000:                     # likely a server cap -> flag for sub-chunking
            capped.append(str(cur.date()))
        for e in data:
            rows.append({"datetime_utc": e.get("date"), "latitude": e.get("latitude"),
                         "longitude": e.get("longitude"), "mag": e.get("magnitude"),
                         "mag_type": e.get("type"), "depth_km": e.get("depth"),
                         "eventID": e.get("eventID")})
        if (cur.month == 1):
            print(f"  {cur:%Y}: running total {len(rows)}", flush=True)
        cur = nxt
    df = pd.DataFrame(rows).drop_duplicates("eventID")
    for c in ("latitude", "longitude", "mag", "depth_km"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df.to_csv(gdir / "afad_raw.csv", index=False)
    print(f"wrote {gdir/'afad_raw.csv'}: {len(df)} unique AFAD events (M>={DENSE_MMIN}); "
          f"capped months: {capped or 'none'}")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "gnss"
    {"gnss": fetch_gnss, "gnss_traj": fetch_gnss_traj, "dense_afad": fetch_dense_afad,
     "repeaters": info_repeaters, "dense": info_dense}[arg]()


if __name__ == "__main__":
    main()
