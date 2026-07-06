"""Source 1 — time-variable GNSS signal (vs the static strain grid).

The static strain field is interseismic and time-invariant by design. The
genuinely new signal is how the crust deforms THROUGH TIME — a slow-slip or locking
transient shows up as a departure of a nearby GNSS station from its long-term secular
trend. That transient is the physically meaningful precursor a static field cannot carry.

Reliable/safe/easy primary data (external fetch script, sources/fetch_data.py gnss): the
Nevada Geodetic Lab (NGL) per-station daily time series (.tenv3, IGS14), plus a small
`gnss_stations.csv` [station,lon,lat] of the fetched Marmara stations. Drop into
Raw_Data/external/gnss/.

Feature (per cell -> nearest station within R km, strictly epochs < t0):
  gnss_rate_change   -- recent horizontal rate (last 365 d) minus the long-term secular
                        rate (all epochs < t0-365d): a transient / locking-change proxy.
                        NaN (no long-record station yet) -> 0 (no detected transient), so
                        the column is NOT a station-availability-vs-time proxy.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401

from .base import DATA, FeatureSource

GDIR = DATA / "gnss"
STATIONS = GDIR / "gnss_stations.csv"
R_KM = 60.0


def _decimal_year(t0):
    t = pd.to_datetime(t0)
    year = t.year
    start = pd.Timestamp(year=year, month=1, day=1)
    end = pd.Timestamp(year=year + 1, month=1, day=1)
    return year + (t - start) / (end - start)


class GnssCouplingSource(FeatureSource):
    name = "gnss_coupling"
    columns = ["gnss_rate_change"]
    data_spec = {
        "requires": f"{GDIR}/<STATION>.tenv3 (NGL IGS14) + {STATIONS} [station,lon,lat]",
        "fetch": "sources/fetch_data.py gnss  (Nevada Geodetic Lab, free, no key)",
        "format": "NGL .tenv3 whitespace: col2=decimal_year, col8=east(m), col10=north(m)",
        "provenance": "Nevada Geodetic Lab http://geodesy.unr.edu/gps_timeseries/tenv3/IGS14/",
        "why": "Time-varying deformation (locking transients) is the precursor the static "
               "strain grid cannot carry; most relevant to the Princes Islands segment.",
    }

    def available(self):
        if not STATIONS.exists():
            return (False, f"missing {STATIONS}")
        n = len(list(GDIR.glob("*.tenv3")))
        return (n > 0, "" if n > 0 else f"no .tenv3 in {GDIR}")

    def _load_stations(self):
        st = pd.read_csv(STATIONS)
        series = {}
        for _, r in st.iterrows():
            f = GDIR / f"{r['station']}.tenv3"
            if not f.exists():
                continue
            a = np.loadtxt(f, usecols=(2, 8, 10), skiprows=1)   # IGS20 tenv3 has a header
            if a.ndim == 1:
                a = a[None, :]
            series[r["station"]] = {"lon": float(r["lon"]), "lat": float(r["lat"]),
                                    "yr": a[:, 0], "disp": np.hypot(a[:, 1], a[:, 2])}
        return series

    def add_columns(self, cells: pd.DataFrame) -> pd.DataFrame:
        S = self._load_stations()
        names = list(S)
        slon = np.array([S[n]["lon"] for n in names])
        slat = np.array([S[n]["lat"] for n in names])
        clon = cells["cell_lon"].to_numpy(); clat = cells["cell_lat"].to_numpy()
        dkm = np.sqrt(((clon[:, None] - slon[None, :]) * 85.0) ** 2
                      + ((clat[:, None] - slat[None, :]) * 111.0) ** 2)
        nearest = np.argmin(dkm, axis=1)
        dist = dkm[np.arange(len(cells)), nearest]
        t0yr = np.array([_decimal_year(t) for t in pd.to_datetime(cells["t0"])])

        rate_change = np.full(len(cells), np.nan)
        for r in range(len(cells)):
            s = S[names[nearest[r]]]
            yr, disp = s["yr"], s["disp"]
            past = yr < t0yr[r]                     # causal
            if past.sum() < 30:
                continue
            recent = past & (yr >= t0yr[r] - 1.0)
            longt = past & (yr < t0yr[r] - 1.0)
            if recent.sum() < 10 or longt.sum() < 30:
                continue
            rate_recent = np.polyfit(yr[recent], disp[recent], 1)[0]
            rate_long = np.polyfit(yr[longt], disp[longt], 1)[0]
            rate_change[r] = rate_recent - rate_long
        # no long-record station -> 0 (no detected transient), and null out cells with
        # no station within R_KM so distance-driven artifacts don't enter
        rate_change[dist > R_KM] = 0.0
        rate_change[np.isnan(rate_change)] = 0.0
        return pd.DataFrame({"gnss_rate_change": rate_change}, index=cells.index)
