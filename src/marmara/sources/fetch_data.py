"""Reproducible fetchers for the three external sources (reliable/safe/easy tier).

Usage:  "<venv>/bin/python3" -m marmara.sources.fetch_data gnss|repeaters|dense
Network calls hit reputable academic sources only. Run in an environment that permits
outbound https (the fetch writes into Raw_Data/external/).
"""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401

from .base import DATA

# Eurasia-fixed frame (isolates Anatolian block motion relative to Eurasia — the
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


def info_repeaters():
    print("Repeating-earthquake catalog — no single-command open download; safest sources:")
    print("  * Schmittbuhl, Karabulut, Lengline, Bouchon 2016 GRL 'Long-lasting seismic")
    print("    repeaters in the Central Basin of the Main Marmara Fault' (supp. table).")
    print("  * Bohnhoff et al. 2017 (Marmara repeaters / creep).")
    print("  Save as Raw_Data/external/marmara_repeaters.csv [datetime,lat,lon,family_id].")


def info_dense():
    print("Dense re-picked catalog (Mc~1.5) — safest sources:")
    print("  * ESM/AFAD or KOERI enhanced bulletin; or a published EQTransformer/PhaseNet")
    print("    Marmara catalog on Zenodo (search 'Marmara deep learning catalog').")
    print("  Save as Raw_Data/external/marmara_dense_catalog.csv")
    print("    [datetime_utc,latitude,longitude,mag].")


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "gnss"
    {"gnss": fetch_gnss, "repeaters": info_repeaters, "dense": info_dense}[arg]()


if __name__ == "__main__":
    main()
