"""Conditional cascade Monte-Carlo forecaster.

Replaces the first-generation `expected_counts` with a full forward simulation of
[t0, t0+H): new background + offspring of ALL history events (residual Omori over
the window) + recursive cascades of everything spawned in-window. Vectorized across
all K sims of a window at once (sim_id carried through the generations).

Per (cell): lam_sim(M>=3.5), lam_sim(M>=4.5), and P_sim(M>=X) = fraction of sims
with >=1 such event in the cell, X in {3.5,4.5,5,5.5,6}. The M5/M6 product is the
sim fraction directly (no GR-extrapolation formula); b-uncertainty via the caller
running two b-values.

Reuses the protected etas_model helpers unmodified.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401

from pathlib import Path

from marmara.etas_model import (BackgroundField, EtasParams, _project_km,
                           _sample_gr, _unproject_km, branching_ratio)
from marmara.stress import nearest_segment_receivers

MMAX = 7.6
X_LEVELS = (3.5, 4.5, 5.0, 5.5, 6.0)
ASPECT = 3.0                                   # elliptical aspect ratio for M>=5.5
BIG_M = 5.5
# SEG_PATH from marmara.paths


def _omori_H(s, c, p):
    return 1.0 - (1.0 + s / c) ** (1.0 - p)


def _strike_angle(lons, lats):
    """Major-axis math angle (rad, CCW from east) = nearest fault-segment strike."""
    strike, _, _, _ = nearest_segment_receivers(np.atleast_1d(lons), np.atleast_1d(lats), SEG_PATH)
    return np.radians(90.0 - strike)           # azimuth (from N, CW) -> math angle


def _pca_angle(dx, dy):
    """Major-axis angle of an aftershock cloud (>=10 offsets), else NaN."""
    if dx.size < 10:
        return np.nan
    Sxx = float(np.mean(dx * dx)); Syy = float(np.mean(dy * dy)); Sxy = float(np.mean(dx * dy))
    return 0.5 * float(np.arctan2(2 * Sxy, Sxx - Syy))


def _history_orientations(t_hist, lon_hist, lat_hist, mag_hist):
    """Orientation (rad) per history parent: PCA of its first-72h M>=mc aftershocks
    (>=10 events) else nearest fault-segment strike. NaN for parents below BIG_M."""
    orient = np.full(mag_hist.size, np.nan)
    big = np.where(mag_hist >= BIG_M)[0]
    for j in big:
        aft = (t_hist > t_hist[j]) & (t_hist <= t_hist[j] + 3.0)
        dx, dy = _project_km(lon_hist[aft], lat_hist[aft], lon_hist[j], lat_hist[j])
        near = (dx * dx + dy * dy) <= 30.0 ** 2
        a = _pca_angle(dx[near], dy[near])
        orient[j] = a if np.isfinite(a) else float(_strike_angle(lon_hist[j], lat_hist[j])[0])
    return orient


def _place_offspring(px, py, pm, params, rng, orient=None):
    """Power-law spatial offset around each parent (km-space); for parents with
    m >= BIG_M and a finite orientation, an area-preserving 3:1 ELLIPTICAL kernel
    aligned to `orient` (aftershock-cloud / fault strike). Isotropic otherwise."""
    reg = params.region
    Dp = params.d * np.power(10.0, params.gamma * (pm - params.mc))
    n = px.size
    lo = np.empty(n); la = np.empty(n)
    todo = np.arange(n)
    ref_lon, ref_lat = params.background_xy.ref_lon, params.background_xy.ref_lat
    sx, sy = np.sqrt(ASPECT), 1.0 / np.sqrt(ASPECT)     # area-preserving stretch
    aniso = np.zeros(n, bool) if orient is None else (np.isfinite(orient) & (pm >= BIG_M))
    for _ in range(30):
        if todo.size == 0:
            break
        v = rng.random(todo.size)
        r = np.sqrt(Dp[todo] * (v ** (-1.0 / (params.q - 1.0)) - 1.0))
        th = rng.uniform(0.0, 2 * np.pi, todo.size)
        dx = r * np.cos(th); dy = r * np.sin(th)
        am = aniso[todo]
        if am.any():
            a = orient[todo][am]
            xf, yf = dx[am] * sx, dy[am] * sy
            dx[am] = xf * np.cos(a) - yf * np.sin(a)
            dy[am] = xf * np.sin(a) + yf * np.cos(a)
        lot, lat_ = _unproject_km(px[todo] + dx, py[todo] + dy, ref_lon, ref_lat)
        ok = ((lot >= reg["min_lon"]) & (lot <= reg["max_lon"])
              & (lat_ >= reg["min_lat"]) & (lat_ <= reg["max_lat"]))
        lo[todo[ok]] = lot[ok]; la[todo[ok]] = lat_[ok]
        todo = todo[~ok]
    if todo.size:
        lot, lat_ = _unproject_km(px[todo], py[todo], ref_lon, ref_lat)
        lo[todo] = np.clip(lot, reg["min_lon"], reg["max_lon"])
        la[todo] = np.clip(lat_, reg["min_lat"], reg["max_lat"])
    return lo, la


def simulate_window(params: EtasParams, hist_t, hist_lon, hist_lat, hist_mag,
                    t0d: float, horizon: float, K: int, rng, b: float,
                    max_events: int = 4_000_000, anisotropy: bool = True,
                    per_sim_cap: int | None = None, stats: dict | None = None,
                    preserve_branching: bool = True, sim_n_target: float | None = None):
    """One batched cascade over K sims. Returns (sim_id, lon, lat, mag) of all
    events that occur in [t0d, t0d+horizon), pooled across sims. t0d in float days.

    per_sim_cap: computational guard for long horizons. A sim that reaches this many
    events stops spawning (near-critical branching can explode over 365 d). If `stats`
    is given, stats['n_capped'] and stats['K'] are filled so the caller can report the
    capped fraction. simulate_window is already per-sim-correct (sim_id is inherited
    through generations), so the cap only bounds runaway tail sims."""
    bg: BackgroundField = params.background_xy
    reg = params.region
    c, p, mc, k, alpha = params.c, params.p, params.mc, params.k, params.alpha
    # --- branching-ratio-preserving rescale (SIMULATED parents only) ------------
    # Offspring magnitudes are DRAWN at the operational b (`b`), usually != params.b.
    # Averaged over that GR(b) draw the productivity gives branching ratio
    # n(b_op) = k*beta/(beta-alpha) = 1.21 > 1 (supercritical) with the fitted k, so a
    # cascade of SIMULATED events explodes and is bounded only by per_sim_cap/max_events.
    # We therefore damp productivity to k_sim for SIMULATED parents, holding the
    # simulated sub-cascade at the fitted mmax-truncated branching ratio 0.95 (or an
    # explicit sim_n_target, for the cap-sensitivity sweep).
    # A REAL history parent has an OBSERVED magnitude (e.g. Kumburgaz Mw6.2): its
    # expected direct-offspring count kappa(m)=k*exp(alpha*(m-mc)) is the MLE-estimated
    # productivity and does NOT depend on b, so real parents KEEP the fitted k. (Damping
    # them would wrongly suppress observed aftershock sequences by ~22%.)
    k_sim = k
    if preserve_branching and abs(b - params.b) > 1e-12:
        target = branching_ratio(params, mmax=MMAX) if sim_n_target is None else float(sim_n_target)
        per_unit = branching_ratio(replace(params, b=b, k=1.0), mmax=MMAX)
        k_sim = target / per_unit
    end = t0d + horizon

    # --- new background over the window, across K sims (total-Poisson trick:
    # sum of K iid Poisson(l) is Poisson(K*l); each event's sim is uniform) ---
    tot_bg = int(rng.poisson(params.mu_total * horizon * K))
    bg_sim = rng.integers(0, K, tot_bg)
    bg_t = t0d + rng.uniform(0.0, horizon, tot_bg)
    bg_lon, bg_lat = bg.sample(tot_bg, rng)
    bg_mag = _sample_gr(tot_bg, rng, mc, b, MMAX)

    # --- offspring of history events, across K sims ---
    a = t0d - hist_t                                  # age of each history event
    good = (a > 0) & (hist_mag >= mc - 1e-9)
    a, hlon, hlat, hmag = a[good], hist_lon[good], hist_lat[good], hist_mag[good]
    # anisotropy orientation for M>=5.5 history parents (from their real aftershocks)
    horient = (_history_orientations(hist_t[good], hlon, hlat, hmag)
               if (anisotropy and np.any(hmag >= BIG_M)) else np.full(hmag.size, np.nan))
    Hn = _omori_H(a + horizon, c, p) - _omori_H(a, c, p)     # Omori mass in window
    kap = k * np.exp(alpha * (hmag - mc))             # REAL history parents: fitted k (obs mag)
    lam_j = np.clip(kap * Hn, 0.0, None)              # expected offspring per sim
    tot_j = rng.poisson(lam_j * K)                    # total offspring across K sims
    nz = int(tot_j.sum())
    hi_sim = np.zeros(0, int); hi_t = np.zeros(0); hi_lon = np.zeros(0)
    hi_lat = np.zeros(0); hi_mag = np.zeros(0)
    if nz > 0:
        src = np.repeat(np.arange(a.size), tot_j)
        sim = rng.integers(0, K, nz)
        Ha = _omori_H(a[src], c, p); Hb = _omori_H(a[src] + horizon, c, p)
        u = rng.random(nz)
        Hval = Ha + u * (Hb - Ha)
        s = c * ((1.0 - Hval) ** (1.0 / (1.0 - p)) - 1.0)     # age of offspring
        hi_t = t0d + (s - a[src])                              # absolute time in window
        px, py = _project_km(hlon[src], hlat[src], bg.ref_lon, bg.ref_lat)
        hi_lon, hi_lat = _place_offspring(px, py, hmag[src], params, rng, orient=horient[src])
        hi_mag = _sample_gr(nz, rng, mc, b, MMAX)
        hi_sim = sim

    # --- assemble the in-window frontier (background + history offspring) ---
    f_sim = np.concatenate([bg_sim, hi_sim.astype(int)])
    f_t = np.concatenate([bg_t, hi_t])
    f_lon = np.concatenate([bg_lon, hi_lon])
    f_lat = np.concatenate([bg_lat, hi_lat])
    f_mag = np.concatenate([bg_mag, hi_mag])

    A_sim = [f_sim]; A_lon = [f_lon]; A_lat = [f_lat]; A_mag = [f_mag]
    total = f_sim.size
    sim_count = np.bincount(f_sim, minlength=K) if per_sim_cap else None
    # --- recursive cascade within the window (vectorized across sims) ---
    while f_sim.size and total < max_events:
        kap = k_sim * np.exp(alpha * (f_mag - mc))    # simulated parents: damped k_sim
        noff = rng.poisson(kap)
        if noff.sum() == 0:
            break
        psim = np.repeat(f_sim, noff); pt = np.repeat(f_t, noff)
        pm = np.repeat(f_mag, noff)
        px, py = _project_km(np.repeat(f_lon, noff), np.repeat(f_lat, noff),
                             bg.ref_lon, bg.ref_lat)
        u = rng.random(pt.size)
        dt = c * (u ** (-1.0 / (p - 1.0)) - 1.0)
        ct = pt + dt
        keep = ct < end
        if not np.any(keep):
            break
        psim, ct, pm, px, py = psim[keep], ct[keep], pm[keep], px[keep], py[keep]
        if per_sim_cap is not None:                 # drop offspring for capped sims
            allowed = sim_count[psim] < per_sim_cap
            psim, ct, pm, px, py = psim[allowed], ct[allowed], pm[allowed], px[allowed], py[allowed]
            if ct.size == 0:
                break
            np.add.at(sim_count, psim, 1)
        # simulated M>=5.5 parents: orient offspring along nearest fault strike
        corient = None
        if anisotropy and np.any(pm >= BIG_M):
            corient = np.full(pm.size, np.nan)
            bigp = pm >= BIG_M
            plon, plat = _unproject_km(px[bigp], py[bigp], bg.ref_lon, bg.ref_lat)
            corient[bigp] = _strike_angle(plon, plat)
        clon, clat = _place_offspring(px, py, pm, params, rng, orient=corient)
        cmag = _sample_gr(ct.size, rng, mc, b, MMAX)
        if total + ct.size > max_events:
            ct = ct[: max_events - total]
            psim, clon, clat, cmag = (psim[:ct.size], clon[:ct.size],
                                      clat[:ct.size], cmag[:ct.size])
        A_sim.append(psim); A_lon.append(clon); A_lat.append(clat); A_mag.append(cmag)
        total += ct.size
        f_sim, f_t, f_lon, f_lat, f_mag = psim, ct, clon, clat, cmag  # new frontier
    if stats is not None:
        stats["n_capped"] = int((sim_count >= per_sim_cap).sum()) if per_sim_cap else 0
        stats["K"] = K
        stats["k_eff"] = float(k_sim)
        stats["k_hist"] = float(k)
        stats["n_branching"] = float(branching_ratio(replace(params, b=b, k=k_sim), mmax=MMAX))
    return (np.concatenate(A_sim), np.concatenate(A_lon),
            np.concatenate(A_lat), np.concatenate(A_mag))


def cascade_forecast(params: EtasParams, history_df, t0d: float, horizon: float,
                     lon_c, lat_c, K: int, seed: int, b: float | None = None,
                     anisotropy: bool = True, per_sim_cap: int | None = None,
                     preserve_branching: bool = True, sim_n_target: float | None = None,
                     return_events: bool = False):
    """Per-cell cascade forecast on the grid (lon_c, lat_c centers).
    history_df: columns datetime/lat/lon/mag OR arrays already; must be causal.
    Returns dict of (nlat, nlon) arrays: lam35, lam45, P{level}; plus scalars
    'K', 'capped_fraction', and regional expected counts 'exp_count{level}'."""
    import pandas as pd
    from marmara.grid import REF
    b = params.b if b is None else b
    nlon, nlat = len(lon_c), len(lat_c)
    lon_edge0 = round(float(lon_c[0]) - 0.05, 2); lat_edge0 = round(float(lat_c[0]) - 0.05, 2)

    # history arrays (strictly < t0). Accept either (datetime_utc,longitude,latitude,mag_w)
    # or (datetime/time,lon,lat,mag) column naming.
    h = history_df
    tcol = "datetime_utc" if "datetime_utc" in h else ("datetime" if "datetime" in h else "time")
    tvals = h[tcol]
    if np.issubdtype(np.asarray(tvals).dtype, np.number):
        td = np.asarray(tvals, float)
    else:
        td = ((pd.to_datetime(tvals) - REF) / pd.Timedelta(days=1)).to_numpy()
    hlon = (h["longitude"] if "longitude" in h else h["lon"]).to_numpy()
    hlat = (h["latitude"] if "latitude" in h else h["lat"]).to_numpy()
    hmag = (h["mag_w"] if "mag_w" in h else h["mag"]).to_numpy()
    causal = td < t0d
    td, hlon, hlat, hmag = td[causal], hlon[causal], hlat[causal], hmag[causal]

    rng = np.random.default_rng(seed)
    meta = {}
    sim, lon, lat, mag = simulate_window(params, td, hlon, hlat, hmag, t0d, horizon, K,
                                         rng, b, anisotropy=anisotropy,
                                         per_sim_cap=per_sim_cap, stats=meta,
                                         preserve_branching=preserve_branching,
                                         sim_n_target=sim_n_target)

    # map to cells
    ic = np.floor((lon - lon_edge0) / 0.1).astype(int)
    ir = np.floor((lat - lat_edge0) / 0.1).astype(int)
    on = (ic >= 0) & (ic < nlon) & (ir >= 0) & (ir < nlat)
    if return_events:   # native per-sim on-grid events, for clustered-catalogue CSEP (task #6)
        return {"sim": sim[on], "lon": lon[on], "lat": lat[on], "mag": mag[on],
                "K": K, "b": float(b), "n_branching": meta.get("n_branching")}
    sim, mag, cflat = sim[on], mag[on], (ir[on] * nlon + ic[on])
    ncells = nlon * nlat

    out = {"K": K, "capped_fraction": (meta.get("n_capped", 0) / K) if per_sim_cap else 0.0,
           "b": float(b), "k_eff": meta.get("k_eff"), "k_hist": meta.get("k_hist"),
           "n_branching": meta.get("n_branching")}
    # M>=3.0 (=mc) total rate for the Phase-3 y30 target (all simulated events are >=mc)
    out["lam30"] = (np.bincount(cflat, minlength=ncells) / K).reshape(nlat, nlon)
    for lvl, name in ((3.5, "lam35"), (4.5, "lam45")):
        m = mag >= lvl
        out[name] = (np.bincount(cflat[m], minlength=ncells) / K).reshape(nlat, nlon)
    for lvl in X_LEVELS:
        m = mag >= lvl
        out[f"exp_count{lvl}"] = float(m.sum()) / K              # regional expected count
        # per-cell expected count (continuous rate, rankable even where P is ~0)
        out[f"lam{lvl}"] = (np.bincount(cflat[m], minlength=ncells) / K).reshape(nlat, nlon)
        # regional exceedance = fraction of sims with >=1 event >= lvl anywhere in box
        out[f"Preg{lvl}"] = float(np.unique(sim[m]).size) / K if m.sum() else 0.0
        if m.sum() == 0:
            out[f"P{lvl}"] = np.zeros((nlat, nlon)); continue
        key = sim[m].astype(np.int64) * ncells + cflat[m]
        uniq = np.unique(key)
        per_cell = np.bincount(uniq % ncells, minlength=ncells)  # #distinct sims/cell
        out[f"P{lvl}"] = (per_cell / K).reshape(nlat, nlon)
    return out
