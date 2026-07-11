"""GNSS trajectory channel (gnss_traj): a properly-engineered replacement for the
broken single-scalar gnss_rate_change (gnss_coupling, kept for the audit trail).

Design decisions that keep the channel causal and physically meaningful:
  (1) East and North are modelled separately (direction matters when removing secular
      plate motion); the norm is taken only after detrending, so the secular trend does
      not dominate the "rate".
  (2) known equipment/earthquake step offsets (NGL steps.txt) are subtracted inside the
      trajectory model.
  (3) annual and semiannual seasonal terms are fit.
  (4) stations within 60 km are combined by inverse-distance^2 weights (>= 2, else 0),
      not the single nearest station.
  (5) the trajectory is fit on epochs < t0-365d and every feature is derived from the
      residuals of [t0-365d, t0) against the extrapolated model. Each feature for window
      t0 therefore uses only data < t0, so recompute_truncated is bit-for-bit trivial
      (the leakage tripwire).

Trajectory model (per station, per component, t in decimal years):
  x(t) = a + b t + sum_k c_k H(t - t_k) + s1 sin2pi t + s2 cos2pi t
                                        + s3 sin4pi t + s4 cos4pi t
fit by least squares on epochs < cutoff, cutoff = decimal_year(month_start(t0) - 365d),
cached per (station, calendar-month) so t0s within a month share the fit (strictly
causal: month_start <= t0 => fit sees only < t0-365d). Step terms use only steps
< cutoff (a step is identifiable only with data on both sides).

Common-mode filter (2.3): at each daily epoch (aligned by MJD) subtract the median
residual across all Marmara stations, per component, before features.

Features (per cell), IDW 1/d^2 over stations <= 60 km (>= 2 else 0), null beyond 60 km:
  gnss_resid_rate_90, gnss_resid_rate_365 : slope (mm/yr) of the CMF residual NORM
  gnss_resid_max_365                       : max CMF residual norm (mm), last 365 d
  gnss_strain_rate_365                     : 2nd invariant of the horizontal strain-rate
        tensor from Delaunay-triangle station residual velocities (containing/nearest
        triangle); gnss_strain_fallback = 1.0 where < 3 usable triangles.
All-NaN -> 0 (absence != signal), matching the v1 convention.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401
from scipy.spatial import Delaunay

from .base import DATA, FeatureSource

GDIR = DATA / "gnss_traj"
STATIONS = GDIR / "gnss_stations.csv"
STEPS = GDIR / "steps.txt"
R_KM = 60.0
KM_PER_DEG_LAT = 111.0
MIN_FIT_EPOCHS = 100          # need a decent secular+seasonal fit
DAYS_YEAR = 365.25


def _decimal_year(ts) -> float:
    t = pd.Timestamp(ts)
    y0 = pd.Timestamp(year=t.year, month=1, day=1)
    y1 = pd.Timestamp(year=t.year + 1, month=1, day=1)
    return t.year + (t - y0) / (y1 - y0)


def _design(t: np.ndarray, steps: np.ndarray) -> np.ndarray:
    cols = [np.ones_like(t), t]
    for tk in steps:
        cols.append((t >= tk).astype(float))
    two_pi = 2.0 * np.pi
    cols += [np.sin(two_pi * t), np.cos(two_pi * t),
             np.sin(2 * two_pi * t), np.cos(2 * two_pi * t)]
    return np.column_stack(cols)


class GnssTrajSource(FeatureSource):
    name = "gnss_traj"
    columns = ["gnss_resid_rate_90", "gnss_resid_rate_365", "gnss_resid_max_365",
               "gnss_strain_rate_365", "gnss_strain_fallback"]
    data_spec = {
        "requires": f"{GDIR}/<STATION>.tenv3 (NGL IGS20) + {STATIONS} [station,lon,lat] + {STEPS}",
        "fetch": "sources/fetch_data.py gnss_traj  (Nevada Geodetic Lab, free, no key)",
        "format": "NGL .tenv3 whitespace: col2=decimal_year, col3=MJD, col8=east(m), col10=north(m)",
        "provenance": "NGL https://geodesy.unr.edu/gps_timeseries/IGS20/tenv3/IGS20/ + steps.txt",
        "why": "Trajectory-modelled, step-corrected, common-mode-filtered transient "
               "deformation, the time-varying precursor the static strain grid cannot carry.",
    }

    def available(self):
        if not STATIONS.exists():
            return (False, f"missing {STATIONS}")
        n = len(list(GDIR.glob("*.tenv3")))
        return (n > 0, "" if n > 0 else f"no .tenv3 in {GDIR}")

    # ---- data loading -------------------------------------------------------
    def _load(self):
        st = pd.read_csv(STATIONS)
        series, pos = {}, {}
        for _, r in st.iterrows():
            f = GDIR / f"{r['station']}.tenv3"
            if not f.exists():
                continue
            a = np.loadtxt(f, usecols=(2, 3, 8, 10), skiprows=1)   # yr, MJD, E(m), N(m)
            if a.ndim == 1:
                a = a[None, :]
            o = np.argsort(a[:, 0])
            series[r["station"]] = {"yr": a[o, 0], "mjd": np.round(a[o, 1]).astype(int),
                                    "E": a[o, 2] * 1000.0, "N": a[o, 3] * 1000.0}  # mm
            pos[r["station"]] = (float(r["lon"]), float(r["lat"]))
        steps = self._load_steps(set(series))
        return series, pos, steps

    def _load_steps(self, keep: set) -> dict:
        out = {s: [] for s in keep}
        if not STEPS.exists():
            return out
        for line in STEPS.read_text().splitlines():
            f = line.split()
            if len(f) < 2 or f[0] not in keep:
                continue
            try:
                dy = _decimal_year(pd.to_datetime(f[1], format="%y%b%d"))
            except Exception:
                continue
            out[f[0]].append(dy)
        return {s: np.array(sorted(v)) for s, v in out.items()}

    # ---- per-station trajectory fit (cached per station+month) --------------
    def _fit(self, cache, sta, s, steps, cutoff_dy):
        key = (sta, round(cutoff_dy, 5))
        if key in cache:
            return cache[key]
        m = s["yr"] < cutoff_dy
        if m.sum() < MIN_FIT_EPOCHS:
            cache[key] = None
            return None
        st_in = steps[np.asarray(steps) < cutoff_dy] if len(steps) else np.array([])
        t = s["yr"][m]
        D = _design(t, st_in)
        try:
            cE = np.linalg.lstsq(D, s["E"][m], rcond=None)[0]
            cN = np.linalg.lstsq(D, s["N"][m], rcond=None)[0]
        except np.linalg.LinAlgError:
            cache[key] = None
            return None
        cache[key] = {"steps": st_in, "cE": cE, "cN": cN}
        return cache[key]

    def _residuals(self, s, fit, lo_dy, hi_dy):
        """CMF-input residuals (E, N) at epochs in [lo_dy, hi_dy)."""
        m = (s["yr"] >= lo_dy) & (s["yr"] < hi_dy)
        if not m.any():
            return None
        t = s["yr"][m]
        D = _design(t, fit["steps"])
        rE = s["E"][m] - D @ fit["cE"]
        rN = s["N"][m] - D @ fit["cN"]
        return {"mjd": s["mjd"][m], "yr": t, "rE": rE, "rN": rN}

    # ---- main compute -------------------------------------------------------
    def _compute(self, cells_in: pd.DataFrame, trunc_dy: float | None = None) -> pd.DataFrame:
        cells = cells_in.reset_index(drop=True)        # guarantee positional 0..N-1
        series, pos, steps = self._load()
        names = list(series)
        if trunc_dy is not None:                       # recompute_truncated: drop >= cutoff
            for s in names:
                mm = series[s]["yr"] < trunc_dy
                for k in ("yr", "mjd", "E", "N"):
                    series[s][k] = series[s][k][mm]
        cols = {c: np.zeros(len(cells)) for c in self.columns}
        if len(names) == 0:
            return pd.DataFrame(cols, index=cells_in.index)
        slon = np.array([pos[s][0] for s in names]); slat = np.array([pos[s][1] for s in names])
        clon = cells["cell_lon"].to_numpy(); clat = cells["cell_lat"].to_numpy()
        coslat = np.cos(np.radians(float(np.mean(slat))))
        # cell<->station distance (km) and station positions in a local km frame
        dkm = np.sqrt(((clon[:, None] - slon[None, :]) * KM_PER_DEG_LAT * coslat) ** 2
                      + ((clat[:, None] - slat[None, :]) * KM_PER_DEG_LAT) ** 2)
        lon0, lat0 = float(np.mean(slon)), float(np.mean(slat))
        sx = (slon - lon0) * KM_PER_DEG_LAT * coslat
        sy = (slat - lat0) * KM_PER_DEG_LAT

        fitcache = {}
        for _, sub in cells.groupby("window"):
            pos_idx = sub.index.to_numpy()             # positional rows (0..N-1)
            t0 = pd.Timestamp(sub["t0"].iloc[0])
            t0_dy = _decimal_year(t0)
            month_start = pd.Timestamp(year=t0.year, month=t0.month, day=1)
            cutoff_fit = _decimal_year(month_start - pd.Timedelta(days=365))
            lo90 = _decimal_year(t0 - pd.Timedelta(days=90))
            lo365 = _decimal_year(t0 - pd.Timedelta(days=365))

            res = {}                                   # station index -> residual series
            for si, sta in enumerate(names):
                fit = self._fit(fitcache, sta, series[sta], steps.get(sta, np.array([])), cutoff_fit)
                if fit is None:
                    continue
                r = self._residuals(series[sta], fit, lo365, t0_dy)
                if r is None or r["mjd"].size < 5:
                    continue
                res[si] = r
            if len(res) == 0:
                continue
            self._common_mode(res)                     # subtract per-epoch cross-station median
            rate90 = {}; rate365 = {}; maxr = {}; velx = {}; vely = {}
            for si, r in res.items():
                norm = np.hypot(r["rE"], r["rN"])
                rate365[si] = _slope(r["yr"], norm)
                m90 = r["yr"] >= lo90
                rate90[si] = _slope(r["yr"][m90], norm[m90]) if m90.sum() >= 3 else rate365[si]
                maxr[si] = float(np.max(norm)) if norm.size else 0.0
                velx[si] = _slope(r["yr"], r["rE"]); vely[si] = _slope(r["yr"], r["rN"])

            dsub = dkm[pos_idx]
            for col, feat in (("gnss_resid_rate_90", rate90),
                              ("gnss_resid_rate_365", rate365),
                              ("gnss_resid_max_365", maxr)):
                cols[col][pos_idx] = self._idw(dsub, feat)
            inv, fb = self._strain(sx, sy, velx, vely, clon[pos_idx], clat[pos_idx],
                                   lon0, lat0, coslat)
            cols["gnss_strain_rate_365"][pos_idx] = inv
            cols["gnss_strain_fallback"][pos_idx] = fb
        return pd.DataFrame(cols, index=cells_in.index)

    def _common_mode(self, res: dict):
        # gather residuals by MJD across stations, subtract per-epoch median
        from collections import defaultdict
        byE = defaultdict(list); byN = defaultdict(list)
        for r in res.values():
            for j, mjd in enumerate(r["mjd"]):
                byE[mjd].append(r["rE"][j]); byN[mjd].append(r["rN"][j])
        medE = {k: np.median(v) for k, v in byE.items()}
        medN = {k: np.median(v) for k, v in byN.items()}
        for r in res.values():
            r["rE"] = r["rE"] - np.array([medE[m] for m in r["mjd"]])
            r["rN"] = r["rN"] - np.array([medN[m] for m in r["mjd"]])

    def _idw(self, dsub, feat: dict):
        """Inverse-distance^2 weighting over stations <=60 km (>=2 else 0)."""
        n = dsub.shape[0]
        out = np.zeros(n)
        sids = list(feat)                       # station indices with a valid feature
        if len(sids) == 0:
            return out
        d = dsub[:, sids]
        within = d <= R_KM
        w = np.where(within, 1.0 / np.clip(d, 0.5, None) ** 2, 0.0)
        fv = np.array([feat[si] for si in sids])
        num = w @ fv
        den = w.sum(axis=1)
        enough = within.sum(axis=1) >= 2
        with np.errstate(invalid="ignore", divide="ignore"):
            out = np.where(enough & (den > 0), num / np.where(den > 0, den, 1), 0.0)
        return out

    def _strain(self, sx, sy, velx, vely, cx, cy, lon0, lat0, coslat):
        """2nd invariant of the strain-rate tensor per cell via Delaunay triangles."""
        n = cx.size
        sids = [si for si in velx]              # stations with a velocity
        if len(sids) < 3:
            return np.zeros(n), np.ones(n)      # fallback flag = 1
        P = np.column_stack([sx[sids], sy[sids]])
        try:
            tri = Delaunay(P)
        except Exception:
            return np.zeros(n), np.ones(n)
        simplices = tri.simplices
        if len(simplices) < 3:
            return np.zeros(n), np.ones(n)
        vx = np.array([velx[si] for si in sids]); vy = np.array([vely[si] for si in sids])
        inv_tri = np.zeros(len(simplices))
        for ti, s in enumerate(simplices):
            X = P[s]; V = np.column_stack([vx[s], vy[s]])
            A = np.column_stack([np.ones(3), X[:, 0], X[:, 1]])   # [1, x, y]
            try:
                cx_ = np.linalg.solve(A, V[:, 0]); cy_ = np.linalg.solve(A, V[:, 1])
            except np.linalg.LinAlgError:
                inv_tri[ti] = 0.0; continue
            exx, exy = cx_[1], 0.5 * (cx_[2] + cy_[1]); eyy = cy_[2]
            inv_tri[ti] = float(np.sqrt(exx ** 2 + 2 * exy ** 2 + eyy ** 2))
        # cell projected to the same local km frame
        cxk = (cx - lon0) * KM_PER_DEG_LAT * coslat
        cyk = (cy - lat0) * KM_PER_DEG_LAT
        which = tri.find_simplex(np.column_stack([cxk, cyk]))
        # nearest triangle (by centroid) for cells outside the hull
        cent = P[simplices].mean(axis=1)
        out = np.zeros(n)
        for i in range(n):
            ti = which[i]
            if ti < 0:
                ti = int(np.argmin((cent[:, 0] - cxk[i]) ** 2 + (cent[:, 1] - cyk[i]) ** 2))
            out[i] = inv_tri[ti]
        return out, np.zeros(n)

    # ---- FeatureSource API --------------------------------------------------
    def add_columns(self, cells: pd.DataFrame) -> pd.DataFrame:
        return self._compute(cells, trunc_dy=None)

    def recompute_truncated(self, cells: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
        return self._compute(cells, trunc_dy=_decimal_year(cutoff))


def _slope(t: np.ndarray, y: np.ndarray) -> float:
    if t.size < 2 or np.ptp(t) < 1e-6:
        return 0.0
    return float(np.polyfit(t, y, 1)[0])
