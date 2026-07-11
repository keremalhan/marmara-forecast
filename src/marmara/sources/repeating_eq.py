"""Source 2: repeating earthquakes as creepmeters on the Main Marmara Fault.

Repeating earthquakes rupture the same small patch; their recurrence interval is
inversely proportional to the local aseismic CREEP rate. A published repeater catalog
is therefore a cheap, direct creep-state observable, a real physical feature no
seismicity-rate statistic captures.

Reliable/safe/easy data (external fetch script, sources/fetch_data.py repeaters): a published
Marmara repeater catalog. Best-documented sources: Schmittbuhl et al. 2016 GRL and
Bohnhoff et al. 2017 (Main Marmara Fault repeaters / creep). Drop a CSV
`marmara_repeaters.csv` [datetime, lat, lon, family_id] into Raw_Data/external/.

Features (per cell, repeaters within R km, strictly datetime < t0):
  repeater_count_730d   -- repeater events in the trailing 2 yr (activity/creep pulse)
  creep_rate_proxy      -- repeaters/yr within R km over the trailing window (creep-rate
                           proxy; higher recurrence -> faster creep)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401

from .base import DATA, FeatureSource

PATH = DATA / "marmara_repeaters.csv"
R_KM = 25.0
WIN_D = 730.0


class RepeatingEqSource(FeatureSource):
    name = "repeating_eq"
    columns = ["repeater_count_730d", "creep_rate_proxy"]
    data_spec = {
        "requires": str(PATH),
        "fetch": "sources/fetch_data.py repeaters  (from published supplementary tables)",
        "format": "CSV [datetime, lat, lon, family_id]",
        "provenance": "Schmittbuhl et al. 2016 GRL; Bohnhoff et al. 2017 "
                      "(Main Marmara Fault repeaters / creep).",
        "why": "Repeater recurrence is a direct aseismic-creep observable; a locked patch "
               "that stops creeping (repeaters slow) is a physical loading signal.",
    }

    def available(self):
        return (PATH.exists(), "" if PATH.exists() else f"missing {PATH}")

    def add_columns(self, cells: pd.DataFrame) -> pd.DataFrame:
        r = pd.read_csv(PATH)
        r["datetime"] = pd.to_datetime(r["datetime"])
        rt = r["datetime"].to_numpy()
        rlon = r["lon"].to_numpy(); rlat = r["lat"].to_numpy()
        order = np.argsort(rt)
        rt, rlon, rlat = rt[order], rlon[order], rlat[order]

        cnt = np.zeros(len(cells)); rate = np.zeros(len(cells))
        clon = cells["cell_lon"].to_numpy(); clat = cells["cell_lat"].to_numpy()
        t0 = pd.to_datetime(cells["t0"]).to_numpy()
        # group rows by window for efficiency
        for w, idx in cells.groupby("window").groups.items():
            t0w = t0[cells.index.get_indexer(idx)][0]
            lo = np.searchsorted(rt, t0w - np.timedelta64(int(WIN_D), "D"), "left")
            hi = np.searchsorted(rt, t0w, "left")               # causal: < t0
            wl, wla = rlon[lo:hi], rlat[lo:hi]
            for r_ in idx:
                p = cells.index.get_loc(r_)
                dkm = np.sqrt(((clon[p] - wl) * 85.0) ** 2 + ((clat[p] - wla) * 111.0) ** 2)
                n = int((dkm <= R_KM).sum())
                cnt[p] = n
                rate[p] = n / (WIN_D / 365.25)
        return pd.DataFrame({"repeater_count_730d": cnt, "creep_rate_proxy": rate},
                            index=cells.index)
