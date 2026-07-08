"""Task 8 baselines. Each returns a Poisson rate lambda per grid row for a
given target threshold (3.5 or 4.5). All are strictly causal and scored through
the same code path as the model (marmara.metrics).

1. Poisson climatology     -- per-cell train-era mean occurrence, time-invariant.
2. Fault-proximity clim.   -- per-cell rate = mean rate of its dist_fault decile.
3. Smoothed seismicity     -- per-t0 Gaussian smoothing of events<t0, scaled to
                              the causal regional rate.
4. ETAS                    -- etas_rate feature rescaled mc->thr by GR.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401
from scipy.ndimage import gaussian_filter

from marmara.grid import LAT_C, LON_C, NLAT, NLON, REF, cell_index

# km per 0.1-degree cell (mean model-box latitude ~40.75).
KM_LAT = 11.1
KM_LON = 111.0 * 0.1 * np.cos(np.radians(40.75))  # ~8.4


def _ycol(thr: float) -> str:
    if abs(thr - 3.0) < 1e-6:
        return "y30"
    return "y35" if abs(thr - 3.5) < 1e-6 else "y45"


def poisson_clim(grid: pd.DataFrame, train_mask: np.ndarray, thr: float) -> np.ndarray:
    """Per-cell mean occurrence over train windows -> time-invariant lambda."""
    y = _ycol(thr)
    tr = grid[train_mask]
    rate = tr.groupby(["ir", "ic"])[y].mean()
    key = list(zip(grid["ir"].to_numpy(), grid["ic"].to_numpy()))
    default = float(tr[y].mean())
    lut = rate.to_dict()
    return np.array([lut.get(k, default) for k in key], float)


def fault_prox_clim(grid: pd.DataFrame, train_mask: np.ndarray, thr: float,
                    n_dec: int = 10) -> np.ndarray:
    """Bin cells by dist_fault_km decile; lambda(cell) = decile mean of the
    per-cell train rate."""
    y = _ycol(thr)
    tr = grid[train_mask]
    cell_rate = tr.groupby(["ir", "ic"])[y].mean()
    cells = cell_rate.reset_index()
    # static per-cell distance
    dist = grid.groupby(["ir", "ic"])["dist_fault_km"].first()
    cells = cells.merge(dist.reset_index(), on=["ir", "ic"])
    cells["dec"] = pd.qcut(cells["dist_fault_km"], n_dec, labels=False, duplicates="drop")
    dec_rate = cells.groupby("dec")[y].mean()
    cells["lam"] = cells["dec"].map(dec_rate)
    lut = {(r.ir, r.ic): r.lam for r in cells.itertuples()}
    default = float(cells["lam"].mean())
    key = list(zip(grid["ir"].to_numpy(), grid["ic"].to_numpy()))
    return np.array([lut.get(k, default) for k in key], float)


def etas_baseline(grid: pd.DataFrame, thr: float, b_train: float,
                  mc_etas: float) -> np.ndarray:
    """lambda_thr = etas_rate * 10^(-b_train*(thr - mc_etas))."""
    scale = 10.0 ** (-b_train * (thr - mc_etas))
    return grid["etas_rate"].to_numpy() * scale


def smoothed_seismicity(grid: pd.DataFrame, cat: pd.DataFrame, mc: float,
                        thr: float, sigma_km: float) -> np.ndarray:
    """Per-t0 Gaussian-smoothed seismicity, scaled so sum(lambda) equals the
    causal regional mean >=thr count per 30 d. lambda per grid row."""
    cat = cat.copy()
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    ir, ic = cell_index(cat["longitude"].to_numpy(), cat["latitude"].to_numpy())
    on = (ir >= 0) & (ic >= 0)
    cat = cat[on].reset_index(drop=True)
    ir, ic = ir[on], ic[on]
    tdays = ((cat["datetime_utc"] - REF) / pd.Timedelta(days=1)).to_numpy()
    magw = cat["mag_w"].to_numpy()
    t_first = tdays.min()

    sig = (sigma_km / KM_LAT, sigma_km / KM_LON)
    is_mc = magw >= mc
    is_thr = magw >= thr

    grid = grid.reset_index(drop=True)   # positional indexing into `out`
    out = np.zeros(len(grid), float)
    for w, sub in grid.groupby("window"):
        t0 = pd.Timestamp(sub["t0"].iloc[0])
        t0d = float((t0 - REF) / pd.Timedelta(days=1))
        past = tdays < t0d
        # spatial density from all events >= Mc before t0
        cnt = np.zeros((NLAT, NLON))
        m = past & is_mc
        np.add.at(cnt, (ir[m], ic[m]), 1.0)
        dens = gaussian_filter(cnt, sigma=sig, mode="constant")
        s = dens.sum()
        if s <= 0:
            dens = np.ones((NLAT, NLON)) / (NLAT * NLON)
        else:
            dens = dens / s
        # causal regional rate: >=thr events before t0 per 30 d
        n_thr = float((past & is_thr).sum())
        n_periods = max((t0d - t_first) / 30.0, 1.0)
        reg_rate = n_thr / n_periods
        lam_grid = dens * reg_rate
        idx = sub.index.to_numpy()
        out[idx] = lam_grid[sub["ir"].to_numpy(), sub["ic"].to_numpy()]
    return out
