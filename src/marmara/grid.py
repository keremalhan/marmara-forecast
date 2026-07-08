"""Spatiotemporal feature grid (baseline).

Builds one row per (cell, 30-day window). EVERY feature uses ONLY events with
time < t0; targets use [t0, t0+30d). The per-window feature function is
vectorized over all 1219 cells (no per-row loops) and is the single source of
truth, so the leakage self-test can truncate the catalog to < t0 and recompute
each row exactly.

Grid: lon centers 25.65..30.85 step 0.1 (53), lat 39.65..41.85 (23) -> 1219 cells.
Windows: t0 = 2005-01-01 + k*30d while t0+30d <= catalog end.

Output: results/grid.parquet  (+ results/grid_report.json)

Run:  "<venv>/bin/python3" -m marmara.grid
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401
from scipy.ndimage import convolve, maximum_filter

from marmara.catalog import b_positive
from marmara.etas_model import expected_counts
from marmara.stress import CoulombStressModel, nearest_segment_receivers

OUT = RESULTS
# SEG_PATH from marmara.paths
# STRAIN_NPZ from marmara.paths

# Grid geometry.
LON_C = np.round(25.65 + 0.1 * np.arange(53), 2)   # 25.65 .. 30.85
LAT_C = np.round(39.65 + 0.1 * np.arange(23), 2)   # 39.65 .. 41.85
NLON, NLAT = len(LON_C), len(LAT_C)                # 53, 23
NCELLS = NLON * NLAT                               # 1219
LON_EDGE0, LAT_EDGE0 = 25.60, 39.60                # lower edges

REF = pd.Timestamp("2003-01-01")                   # time origin (day 0)
T0_START = pd.Timestamp("2005-01-01")
STEP_D = 30.0
HORIZON_D = 30.0
DAYS_CAP = 3650.0
YEAR_PERIODS = 365.0 / 30.0                        # 12.1667

# Feature column order (fixed).
FEATURES = [
    "cnt30", "cnt90", "cnt365",
    "nbr3_cnt30", "nbr3_cnt365", "nbr5_cnt30", "nbr5_cnt365",
    "days_since_m35_25km", "days_since_m45_25km",
    "maxmag365_25km", "mean_depth90", "rate_ratio",
    "b_pos_50km_730d", "b_pos_is_fallback",
    "dist_fault_km", "dcfs_perm", "dcfs_decay25", "strain_inv", "etas_rate",
]
# Seismicity-only subset (used by Task 9 on ETAS simulations; no stress/strain/fault).
SEISMICITY_FEATURES = [
    "cnt30", "cnt90", "cnt365",
    "nbr3_cnt30", "nbr3_cnt365", "nbr5_cnt30", "nbr5_cnt365",
    "days_since_m35_25km", "days_since_m45_25km",
    "maxmag365_25km", "rate_ratio", "b_pos_50km_730d", "b_pos_is_fallback",
    "etas_rate",
]

K3 = np.ones((3, 3))
K5 = np.ones((5, 5))
K9 = np.ones((9, 9))
NEG = -1.0e18


def _to_days(ts) -> np.ndarray:
    return (pd.to_datetime(ts) - REF) / pd.Timedelta(days=1)


def cell_index(lon: np.ndarray, lat: np.ndarray):
    """(ir, ic) grid indices for points; -1 where outside the grid."""
    ic = np.floor((np.asarray(lon) - LON_EDGE0) / 0.1).astype(int)
    ir = np.floor((np.asarray(lat) - LAT_EDGE0) / 0.1).astype(int)
    bad = (ic < 0) | (ic >= NLON) | (ir < 0) | (ir >= NLAT)
    ir = np.where(bad, -1, ir)
    ic = np.where(bad, -1, ic)
    return ir, ic


# --------------------------------------------------------------------------- #
# Event bundle: global time-sorted arrays for the magnitude subsets we need.
# --------------------------------------------------------------------------- #
def build_event_bundle(cat: pd.DataFrame, mc: float) -> dict:
    """Time-sorted arrays for events that fall on the grid. `cat` may be any
    truncation of the model catalog (the leakage test passes a < t0 slice)."""
    df = cat.copy()
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    ir, ic = cell_index(df["longitude"].to_numpy(), df["latitude"].to_numpy())
    on = (ir >= 0)
    df = df[on].reset_index(drop=True)
    ir, ic = ir[on], ic[on]
    t = _to_days(df["datetime_utc"]).to_numpy()
    order = np.argsort(t, kind="mergesort")

    t = t[order]; ir = ir[order]; ic = ic[order]
    mag = df["mag_w"].to_numpy()[order]
    depth = df["depth_km"].to_numpy()[order]
    lon = df["longitude"].to_numpy()[order]
    lat = df["latitude"].to_numpy()[order]

    def sub(mask):
        return dict(t=t[mask], ir=ir[mask], ic=ic[mask], mag=mag[mask],
                    depth=depth[mask], lon=lon[mask], lat=lat[mask])

    return {
        "all": sub(np.ones(len(t), bool)),
        "mc": sub(mag >= mc),
        "e30": sub(mag >= 3.0),          # Phase 3: y30 target (M>=3.0)
        "e35": sub(mag >= 3.5),
        "e45": sub(mag >= 4.5),
        "e25": sub(mag >= 2.5),
        "mc_val": mc,
    }


def _count_grid(ir, ic) -> np.ndarray:
    return np.bincount(ir * NLON + ic, minlength=NCELLS).reshape(NLAT, NLON).astype(float)


# --------------------------------------------------------------------------- #
# Static per-cell context (time-invariant): fault distance, receiver planes,
# strain invariant. Built once.
# --------------------------------------------------------------------------- #
def build_static_context() -> dict:
    LO, LA = np.meshgrid(LON_C, LAT_C)             # (NLAT, NLON)
    cell_lon = LO.ravel(); cell_lat = LA.ravel()
    strike, dip, rake, dist = nearest_segment_receivers(cell_lon, cell_lat, SEG_PATH)

    z = np.load(STRAIN_NPZ)
    xs = np.unique(z["x"]); ys = np.unique(z["y"])
    inv = z["invariant"]
    # z['x'],z['y'] are meshgrid(x,y): infer orientation and reshape inv onto (ys, xs).
    if z["x"].shape == (len(ys), len(xs)) and np.allclose(z["x"][0, :], xs):
        grid_inv = inv                              # already (ny, nx)
    else:
        grid_inv = inv.T
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((ys, xs), grid_inv, bounds_error=False,
                                     fill_value=None)  # extrapolate at edges
    strain_inv = interp(np.column_stack([cell_lat, cell_lon]))

    return {
        "cell_lon": cell_lon, "cell_lat": cell_lat,
        "strike": strike, "dip": dip, "rake": rake,
        "dist_fault_km": dist.reshape(NLAT, NLON),
        "strain_inv": strain_inv.reshape(NLAT, NLON),
        "stress_model": CoulombStressModel(),
    }


# --------------------------------------------------------------------------- #
# Per-window features (all cells at once). Only events with t < t0 are used.
# --------------------------------------------------------------------------- #
def features_at_window(EV: dict, t0d: float, t0_dt: pd.Timestamp,
                       ctx: dict, params, subset: bool = False) -> dict:
    mc = EV["mc"]
    out: dict[str, np.ndarray] = {}

    # ---- counts (events >= Mc), exact trailing day-windows ----
    tmc = mc["t"]
    hi = np.searchsorted(tmc, t0d, "left")

    def cnt(D):
        lo = np.searchsorted(tmc, t0d - D, "left")
        return _count_grid(mc["ir"][lo:hi], mc["ic"][lo:hi])

    C30, C90, C365 = cnt(30.0), cnt(90.0), cnt(365.0)
    out["cnt30"], out["cnt90"], out["cnt365"] = C30, C90, C365
    out["nbr3_cnt30"] = convolve(C30, K3, mode="constant", cval=0.0)
    out["nbr3_cnt365"] = convolve(C365, K3, mode="constant", cval=0.0)
    out["nbr5_cnt30"] = convolve(C30, K5, mode="constant", cval=0.0)
    out["nbr5_cnt365"] = convolve(C365, K5, mode="constant", cval=0.0)
    out["rate_ratio"] = C30 / (C365 / YEAR_PERIODS + 0.1)

    # ---- days since last M>=3.5 / >=4.5 in 5x5 (25 km) neighborhood ----
    def days_since(sub):
        h = np.searchsorted(sub["t"], t0d, "left")
        last = np.full((NLAT, NLON), NEG)
        np.maximum.at(last, (sub["ir"][:h], sub["ic"][:h]), sub["t"][:h])
        nb = maximum_filter(last, size=5, mode="constant", cval=NEG)
        ds = np.clip(t0d - nb, 0.0, DAYS_CAP)
        ds[nb <= NEG / 2] = DAYS_CAP
        return ds

    out["days_since_m35_25km"] = days_since(EV["e35"])
    out["days_since_m45_25km"] = days_since(EV["e45"])

    # ---- max magnitude (all events) in 5x5, trailing 365 d ----
    al = EV["all"]
    lo = np.searchsorted(al["t"], t0d - 365.0, "left")
    hia = np.searchsorted(al["t"], t0d, "left")
    cellmax = np.full((NLAT, NLON), NEG)
    np.maximum.at(cellmax, (al["ir"][lo:hia], al["ic"][lo:hia]), al["mag"][lo:hia])
    nbmax = maximum_filter(cellmax, size=5, mode="constant", cval=NEG)
    out["maxmag365_25km"] = np.where(nbmax <= NEG / 2, 0.0, nbmax)

    # ---- mean depth (events >= Mc) in cell, trailing 90 d ----
    lo90 = np.searchsorted(tmc, t0d - 90.0, "left")
    dep = mc["depth"][lo90:hi]
    ir90, ic90 = mc["ir"][lo90:hi], mc["ic"][lo90:hi]
    sumd = np.zeros((NLAT, NLON)); cntd = np.zeros((NLAT, NLON))
    np.add.at(sumd, (ir90, ic90), dep)
    np.add.at(cntd, (ir90, ic90), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cellmean = np.where(cntd > 0, sumd / np.where(cntd > 0, cntd, 1), np.nan)
    if len(dep) > 0:
        regional = float(dep.mean())
    elif hi > 0:
        regional = float(mc["depth"][:hi].mean())
    else:
        regional = 10.0
    out["mean_depth90"] = np.where(cntd > 0, cellmean, regional)

    # ---- b-positive within ~50 km (9x9) & last 730 d ----
    lo730 = np.searchsorted(tmc, t0d - 730.0, "left")
    ir7, ic7, mg7 = mc["ir"][lo730:hi], mc["ic"][lo730:hi], mc["mag"][lo730:hi]
    cnt730 = _count_grid(ir7, ic7)
    nbcnt = convolve(cnt730, K9, mode="constant", cval=0.0)
    reg_b = b_positive(mg7) if len(mg7) else None      # mg7 is chronological
    bpos = np.full((NLAT, NLON), np.nan)
    isfb = np.ones((NLAT, NLON))
    if len(mg7):
        for i, j in np.argwhere(nbcnt >= 31):
            m = (np.abs(ir7 - i) <= 4) & (np.abs(ic7 - j) <= 4)
            b = b_positive(mg7[m])
            if b is not None:
                bpos[i, j] = b
                isfb[i, j] = 0.0
    fill = reg_b if reg_b is not None else 1.0
    out["b_pos_50km_730d"] = np.where(np.isnan(bpos), fill, bpos)
    out["b_pos_is_fallback"] = isfb

    # ---- ETAS expected M>=mc counts (history >= 2.5, last 3650 d) ----
    e25 = EV["e25"]
    a = np.searchsorted(e25["t"], t0d - 3650.0, "left")
    bnd = np.searchsorted(e25["t"], t0d, "left")
    hist = pd.DataFrame({
        "datetime": REF + pd.to_timedelta(e25["t"][a:bnd], unit="D"),
        "latitude": e25["lat"][a:bnd], "longitude": e25["lon"][a:bnd],
        "mag": e25["mag"][a:bnd],
    })
    cells = pd.DataFrame({"cell_lon": ctx["cell_lon"], "cell_lat": ctx["cell_lat"]})
    ec = expected_counts(params, hist, cells, t0_dt, horizon_days=HORIZON_D)
    out["etas_rate"] = np.asarray(ec).reshape(NLAT, NLON)

    if subset:
        return {k: out[k] for k in SEISMICITY_FEATURES}

    # ---- stress + static (catalog-independent; identical under truncation) ----
    sm = ctx["stress_model"]
    out["dcfs_perm"] = sm.delta_cfs(
        ctx["cell_lon"], ctx["cell_lat"], at_time=t0_dt, half_life_years=None,
        receiver_strike=ctx["strike"], receiver_dip=ctx["dip"],
        receiver_rake=ctx["rake"]).reshape(NLAT, NLON)
    out["dcfs_decay25"] = sm.delta_cfs(
        ctx["cell_lon"], ctx["cell_lat"], at_time=t0_dt, half_life_years=25.0,
        receiver_strike=ctx["strike"], receiver_dip=ctx["dip"],
        receiver_rake=ctx["rake"]).reshape(NLAT, NLON)
    out["dist_fault_km"] = ctx["dist_fault_km"]
    out["strain_inv"] = ctx["strain_inv"]
    return out


def targets_at_window(EV: dict, t0d: float) -> tuple[np.ndarray, np.ndarray]:
    def y(sub):
        lo = np.searchsorted(sub["t"], t0d, "left")
        hi = np.searchsorted(sub["t"], t0d + HORIZON_D, "left")
        g = np.zeros((NLAT, NLON))
        g[sub["ir"][lo:hi], sub["ic"][lo:hi]] = 1.0
        return g
    return y(EV["e35"]), y(EV["e45"])


def y30_at_window(EV: dict, t0d: float) -> np.ndarray:
    """Phase 3: M>=3.0 occurrence in [t0, t0+30) (same causal logic as
    targets_at_window; separate fn so targets_at_window's signature is unchanged)."""
    sub = EV["e30"]
    lo = np.searchsorted(sub["t"], t0d, "left")
    hi = np.searchsorted(sub["t"], t0d + HORIZON_D, "left")
    g = np.zeros((NLAT, NLON))
    g[sub["ir"][lo:hi], sub["ic"][lo:hi]] = 1.0
    return g


# ========================================================================== #
# Generic (arbitrary-geometry) seismicity feature builder.
# Used by Task 9 for both the model-box ETAS simulations and the wide-box real
# catalog. Produces exactly the SEISMICITY_FEATURES; the model-box results are
# identical to features_at_window(subset=True) (verified in Task 9's checks).
# ========================================================================== #
class GridSpec:
    def __init__(self, min_lon, max_lon, min_lat, max_lat):
        self.nlon = int(round((max_lon - min_lon) / 0.1))
        self.nlat = int(round((max_lat - min_lat) / 0.1))
        self.lon_c = np.round(min_lon + 0.05 + 0.1 * np.arange(self.nlon), 2)
        self.lat_c = np.round(min_lat + 0.05 + 0.1 * np.arange(self.nlat), 2)
        self.ncells = self.nlon * self.nlat
        self.lon_edge0 = round(min_lon, 2)
        self.lat_edge0 = round(min_lat, 2)


MODEL_SPEC = GridSpec(25.6, 30.9, 39.6, 41.9)     # matches the model-box grid
WIDE_SPEC = GridSpec(25.0, 31.5, 39.0, 42.5)      # Task 9 real-M6 checks


def cell_index_spec(lon, lat, spec: GridSpec):
    ic = np.floor((np.asarray(lon) - spec.lon_edge0) / 0.1).astype(int)
    ir = np.floor((np.asarray(lat) - spec.lat_edge0) / 0.1).astype(int)
    bad = (ic < 0) | (ic >= spec.nlon) | (ir < 0) | (ir >= spec.nlat)
    return np.where(bad, -1, ir), np.where(bad, -1, ic)


def build_bundle(t_days, lon, lat, mag, depth, spec: GridSpec, mc: float) -> dict:
    """Time-sorted event arrays on `spec`. t_days = float days since REF."""
    t = np.asarray(t_days, float); lon = np.asarray(lon, float)
    lat = np.asarray(lat, float); mag = np.asarray(mag, float)
    depth = np.asarray(depth, float)
    ir, ic = cell_index_spec(lon, lat, spec)
    on = ir >= 0
    t, ir, ic, mag, depth, lon, lat = (a[on] for a in (t, ir, ic, mag, depth, lon, lat))
    o = np.argsort(t, kind="mergesort")
    t, ir, ic, mag, depth, lon, lat = (a[o] for a in (t, ir, ic, mag, depth, lon, lat))

    def sub(m):
        return dict(t=t[m], ir=ir[m], ic=ic[m], mag=mag[m], depth=depth[m],
                    lon=lon[m], lat=lat[m])

    return {"all": sub(np.ones(len(t), bool)), "mc": sub(mag >= mc),
            "e30": sub(mag >= 3.0), "e35": sub(mag >= 3.5), "e45": sub(mag >= 4.5),
            "e55": sub(mag >= 5.5), "e25": sub(mag >= 2.5)}


def _cgrid(ir, ic, spec):
    return np.bincount(ir * spec.nlon + ic,
                       minlength=spec.ncells).reshape(spec.nlat, spec.nlon).astype(float)


def seismicity_features(EV: dict, t0d: float, t0_dt: pd.Timestamp,
                        spec: GridSpec, params, etas_hist_days: float = 3650.0) -> dict:
    """The 14 SEISMICITY_FEATURES on `spec`, causal (events < t0).

    etas_hist_days controls the ETAS history window (3650 matches the model-box
    grid; Task 9 uses 730 for tractability on 1000-simulation-scale work)."""
    nlat, nlon, neg = spec.nlat, spec.nlon, NEG
    mc = EV["mc"]; tmc = mc["t"]
    hi = np.searchsorted(tmc, t0d, "left")
    out = {}

    def cnt(D):
        lo = np.searchsorted(tmc, t0d - D, "left")
        return _cgrid(mc["ir"][lo:hi], mc["ic"][lo:hi], spec)

    C30, C90, C365 = cnt(30.0), cnt(90.0), cnt(365.0)
    out["cnt30"], out["cnt90"], out["cnt365"] = C30, C90, C365
    out["nbr3_cnt30"] = convolve(C30, K3, mode="constant", cval=0.0)
    out["nbr3_cnt365"] = convolve(C365, K3, mode="constant", cval=0.0)
    out["nbr5_cnt30"] = convolve(C30, K5, mode="constant", cval=0.0)
    out["nbr5_cnt365"] = convolve(C365, K5, mode="constant", cval=0.0)
    out["rate_ratio"] = C30 / (C365 / YEAR_PERIODS + 0.1)

    def days_since(sub):
        h = np.searchsorted(sub["t"], t0d, "left")
        last = np.full((nlat, nlon), neg)
        np.maximum.at(last, (sub["ir"][:h], sub["ic"][:h]), sub["t"][:h])
        nb = maximum_filter(last, size=5, mode="constant", cval=neg)
        ds = np.clip(t0d - nb, 0.0, DAYS_CAP)
        ds[nb <= neg / 2] = DAYS_CAP
        return ds

    out["days_since_m35_25km"] = days_since(EV["e35"])
    out["days_since_m45_25km"] = days_since(EV["e45"])

    al = EV["all"]
    lo = np.searchsorted(al["t"], t0d - 365.0, "left")
    hia = np.searchsorted(al["t"], t0d, "left")
    cellmax = np.full((nlat, nlon), neg)
    np.maximum.at(cellmax, (al["ir"][lo:hia], al["ic"][lo:hia]), al["mag"][lo:hia])
    nbmax = maximum_filter(cellmax, size=5, mode="constant", cval=neg)
    out["maxmag365_25km"] = np.where(nbmax <= neg / 2, 0.0, nbmax)

    lo730 = np.searchsorted(tmc, t0d - 730.0, "left")
    ir7, ic7, mg7 = mc["ir"][lo730:hi], mc["ic"][lo730:hi], mc["mag"][lo730:hi]
    nbcnt = convolve(_cgrid(ir7, ic7, spec), K9, mode="constant", cval=0.0)
    reg_b = b_positive(mg7) if len(mg7) else None
    bpos = np.full((nlat, nlon), np.nan); isfb = np.ones((nlat, nlon))
    if len(mg7):
        for i, j in np.argwhere(nbcnt >= 31):
            b = b_positive(mg7[(np.abs(ir7 - i) <= 4) & (np.abs(ic7 - j) <= 4)])
            if b is not None:
                bpos[i, j] = b; isfb[i, j] = 0.0
    fill = reg_b if reg_b is not None else 1.0
    out["b_pos_50km_730d"] = np.where(np.isnan(bpos), fill, bpos)
    out["b_pos_is_fallback"] = isfb

    e25 = EV["e25"]
    a = np.searchsorted(e25["t"], t0d - etas_hist_days, "left")
    bnd = np.searchsorted(e25["t"], t0d, "left")
    hist = pd.DataFrame({
        "datetime": REF + pd.to_timedelta(e25["t"][a:bnd], unit="D"),
        "latitude": e25["lat"][a:bnd], "longitude": e25["lon"][a:bnd],
        "mag": e25["mag"][a:bnd]})
    LO, LA = np.meshgrid(spec.lon_c, spec.lat_c)
    cells = pd.DataFrame({"cell_lon": LO.ravel(), "cell_lat": LA.ravel()})
    ec = expected_counts(params, hist, cells, t0_dt, horizon_days=HORIZON_D)
    out["etas_rate"] = np.asarray(ec).reshape(nlat, nlon)
    return {k: out[k] for k in SEISMICITY_FEATURES}


def build_static_context_spec(spec: GridSpec) -> dict:
    """build_static_context generalized to an arbitrary GridSpec (strain is
    extrapolated for cells outside the model-box strain grid — documented)."""
    LO, LA = np.meshgrid(spec.lon_c, spec.lat_c)
    cell_lon = LO.ravel(); cell_lat = LA.ravel()
    strike, dip, rake, dist = nearest_segment_receivers(cell_lon, cell_lat, SEG_PATH)
    z = np.load(STRAIN_NPZ)
    xs = np.unique(z["x"]); ys = np.unique(z["y"]); inv = z["invariant"]
    grid_inv = inv if (z["x"].shape == (len(ys), len(xs)) and np.allclose(z["x"][0, :], xs)) else inv.T
    from scipy.interpolate import RegularGridInterpolator
    interp = RegularGridInterpolator((ys, xs), grid_inv, bounds_error=False, fill_value=None)
    strain_inv = interp(np.column_stack([cell_lat, cell_lon]))
    return {"cell_lon": cell_lon, "cell_lat": cell_lat, "strike": strike, "dip": dip,
            "rake": rake, "dist_fault_km": dist.reshape(spec.nlat, spec.nlon),
            "strain_inv": strain_inv.reshape(spec.nlat, spec.nlon),
            "stress_model": CoulombStressModel()}


def features_full_spec(EV: dict, t0d: float, t0_dt: pd.Timestamp, spec: GridSpec,
                       ctx: dict, params, etas_hist_days: float = 3650.0) -> dict:
    """All 19 FEATURES on an arbitrary GridSpec (seismicity + mean_depth + stress/
    static), same column set/order as features_at_window. Used for the wide-box y45."""
    out = dict(seismicity_features(EV, t0d, t0_dt, spec, params, etas_hist_days))
    mc = EV["mc"]; tmc = mc["t"]; hi = np.searchsorted(tmc, t0d, "left")
    lo90 = np.searchsorted(tmc, t0d - 90.0, "left")
    dep = mc["depth"][lo90:hi]; ir90, ic90 = mc["ir"][lo90:hi], mc["ic"][lo90:hi]
    sumd = np.zeros((spec.nlat, spec.nlon)); cntd = np.zeros((spec.nlat, spec.nlon))
    np.add.at(sumd, (ir90, ic90), dep); np.add.at(cntd, (ir90, ic90), 1.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cellmean = np.where(cntd > 0, sumd / np.where(cntd > 0, cntd, 1), np.nan)
    regional = float(dep.mean()) if len(dep) else (float(mc["depth"][:hi].mean()) if hi > 0 else 10.0)
    out["mean_depth90"] = np.where(cntd > 0, cellmean, regional)
    sm = ctx["stress_model"]
    out["dcfs_perm"] = sm.delta_cfs(ctx["cell_lon"], ctx["cell_lat"], at_time=t0_dt,
        half_life_years=None, receiver_strike=ctx["strike"], receiver_dip=ctx["dip"],
        receiver_rake=ctx["rake"]).reshape(spec.nlat, spec.nlon)
    out["dcfs_decay25"] = sm.delta_cfs(ctx["cell_lon"], ctx["cell_lat"], at_time=t0_dt,
        half_life_years=25.0, receiver_strike=ctx["strike"], receiver_dip=ctx["dip"],
        receiver_rake=ctx["rake"]).reshape(spec.nlat, spec.nlon)
    out["dist_fault_km"] = ctx["dist_fault_km"]
    out["strain_inv"] = ctx["strain_inv"]
    return out


def target_neighborhood(EV: dict, t0d: float, spec: GridSpec, thr: float,
                        horizon: float, nbhd: int) -> np.ndarray:
    """1 where any event >= thr occurs in the nbhd x nbhd block in [t0, t0+horizon)."""
    key = {3.5: "e35", 4.5: "e45", 5.5: "e55"}[thr]
    sub = EV[key]
    lo = np.searchsorted(sub["t"], t0d, "left")
    hi = np.searchsorted(sub["t"], t0d + horizon, "left")
    g = np.zeros((spec.nlat, spec.nlon))
    g[sub["ir"][lo:hi], sub["ic"][lo:hi]] = 1.0
    if nbhd > 1:
        g = (maximum_filter(g, size=nbhd, mode="constant", cval=0.0) > 0).astype(float)
    return g


def window_starts(cat_end: pd.Timestamp) -> list[pd.Timestamp]:
    starts = []
    k = 0
    while True:
        t0 = T0_START + pd.Timedelta(days=STEP_D * k)
        if t0 + pd.Timedelta(days=HORIZON_D) > cat_end:
            break
        starts.append(t0)
        k += 1
    return starts


def load_mc() -> float:
    with open(OUT / "catalog_report.json") as f:
        return float(json.load(f)["mc"])


def load_params():
    with open(OUT / "etas_params.pkl", "rb") as f:
        return pickle.load(f)


def build() -> dict:
    OUT.mkdir(exist_ok=True)
    mc = load_mc()
    params = load_params()
    cat = pd.read_csv(OUT / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])

    EV = build_event_bundle(cat, mc)
    ctx = build_static_context()
    cat_end = cat["datetime_utc"].max()
    starts = window_starts(cat_end)

    LO, LA = np.meshgrid(LON_C, LAT_C)
    flat_lon = LO.ravel(); flat_lat = LA.ravel()
    ir_flat = np.repeat(np.arange(NLAT), NLON)
    ic_flat = np.tile(np.arange(NLON), NLAT)

    rows = []
    for k, t0_dt in enumerate(starts):
        t0d = float(_to_days(t0_dt))
        feats = features_at_window(EV, t0d, t0_dt, ctx, params)
        y35, y45 = targets_at_window(EV, t0d)
        block = {
            "window": np.full(NCELLS, k),
            "t0": np.full(NCELLS, np.datetime64(t0_dt), dtype="datetime64[ns]"),
            "cell_lon": flat_lon, "cell_lat": flat_lat,
            "ir": ir_flat, "ic": ic_flat,
        }
        for name in FEATURES:
            block[name] = feats[name].ravel()
        block["y35"] = y35.ravel()
        block["y45"] = y45.ravel()
        rows.append(pd.DataFrame(block))
        if (k + 1) % 30 == 0:
            print(f"  window {k+1}/{len(starts)}  {t0_dt.date()}")

    grid = pd.concat(rows, ignore_index=True)
    grid.to_parquet(OUT / "grid.parquet", index=False)

    report = {
        "n_cells": NCELLS, "n_windows": len(starts),
        "n_rows": int(len(grid)), "mc": mc,
        "window_range": [str(starts[0].date()), str(starts[-1].date())],
        "catalog_end": str(cat_end),
        "y35_positive_rate": float(grid["y35"].mean()),
        "y45_positive_rate": float(grid["y45"].mean()),
        "features": FEATURES,
        "seismicity_features": SEISMICITY_FEATURES,
    }
    with open(OUT / "grid_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print("grid built:", json.dumps(report, indent=2))
    return {"grid": grid, "report": report}


if __name__ == "__main__":
    build()
