"""Phase 1.1 — modern_etas: gridded 30-day forecast from the INDEPENDENT
third-party lmizrahi/etas 3.0.0 inversion, via first-generation INTENSITY
INTEGRATION (the blueprint's sanctioned alternative to simulation).

Why intensity integration, not our cascade or their simulator:
  * The Mizrahi best fit is a TAPERED Omori with p = 1+omega = 0.81 (< 1) and
    tau ~ 1229 d — structurally incompatible with our pure-Omori cascade
    (feeding its params into cascade.py would misrepresent it).
  * The package's ETASSimulation continues the single inversion catalogue forward
    from the fit end; it is not built for 51 leakage-clean per-window backtests
    (that would need per-window re-inversion, ~7 min each).
So we evaluate the Mizrahi model the same way our `firstgen_etas` baseline is
evaluated: the FIRST-GENERATION expected M>=mc count per (cell, 30-day window),
computed analytically from the inverted params + the package's exact kernels
(offspring of the causal history < t0 only, no secondary triggering), then
GR-scaled to M>=thr. This makes modern_etas directly comparable to firstgen_etas
(our first-gen ETAS) and cascade — an independent-implementation reference, not a
strawman.

Mizrahi kernels (etas.inversion.triggering_kernel / expected_aftershocks):
  productivity  kappa(m)   = k0 * exp(a*(m-mc))
  temporal      g(dt)      = exp(-dt/tau) / (dt+c)^(1+omega)
  time integral over [dt0,dt1] = exp(c/tau)*tau^(-omega) *
                    [UGE(-omega,(dt0+c)/tau) - UGE(-omega,(dt1+c)/tau)]
                    where UGE(a>0,x)=gammaincc(a,x)*Gamma(a)
  spatial       f(r2;m)    = (r2 + d*exp(gamma*(m-mc)))^-(1+rho)   (unnormalised)
  cell count from source j  = kappa(m_j)*time_integral_j*(r_cj^2+D_j)^-(1+rho)*A_c
    (sums over cells to kappa*time*area_factor = the package's expected_aftershocks)

Ranking metrics (PR-AUC/ROC-AUC) are invariant to the global GR factor
10^(-b*(thr-mc)); only IG/Brier see Mizrahi's absolute calibration (b, mu).
Background: uniform at the fitted mu (per day per km^2) — a small spatially-flat
floor for M>=3 (documented; Mizrahi's spatial background KDE is not reused).

Runs in the pinned MAIN env (no etas import). Reads results/etas_mizrahi_fit.json.
Output: results/rates_modern_etas.parquet [window, ir, ic, lam35, lam45]
Run:  "<venv>/bin/python" -m marmara.etas_modern [--validate]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401
from scipy.special import gammaincc, gamma as gamma_func

from marmara import grid as G

OUT = RESULTS
MC = 3.0
KM_PER_DEG = 111.19
TRAIN_END = pd.Timestamp("2022-01-01")
VAL_END = pd.Timestamp("2024-01-01")
TEST_TARGET_END = pd.Timestamp("2026-03-31")


def _uge(a, x):
    """upper_gamma_ext for a>0 (our case: a=-omega>0), matching etas.inversion."""
    return gammaincc(a, x) * gamma_func(a)


def _load_params():
    d = json.load(open(OUT / "etas_mizrahi_fit.json"))
    p = d["param_dict"]
    return {
        "k0": 10.0 ** p["log10_k0"], "a": p["a"],
        "c": 10.0 ** p["log10_c"], "omega": p["omega"], "tau": 10.0 ** p["log10_tau"],
        "d": 10.0 ** p["log10_d"], "gamma": p["gamma"], "rho": p["rho"],
        "mu": 10.0 ** p["log10_mu"],
        "b": (d["beta"] / np.log(10.0)) if d.get("beta") else 1.0,
    }


def _time_integral(dt0, dt1, c, omega, tau):
    """Integral of the tapered-Omori temporal kernel over [dt0, dt1] (days)."""
    base = np.exp(c / tau) * tau ** (-omega)
    return base * (_uge(-omega, (dt0 + c) / tau) - _uge(-omega, (dt1 + c) / tau))


def valtest_window_ids(starts):
    out = []
    for k, t0 in enumerate(starts):
        tgt_end = t0 + pd.Timedelta(days=30)
        if ((t0 >= TRAIN_END) & (t0 < VAL_END)) or ((t0 >= VAL_END) & (tgt_end <= TEST_TARGET_END)):
            out.append(k)
    return out


def lam_mc_window(P, t0d, hist_t, hist_lon, hist_lat, hist_mag, cell_lon, cell_lat, area_c):
    """First-generation Mizrahi expected M>=mc count per cell for [t0, t0+30)."""
    c, omega, tau, d, gamma, rho, k0, a = (P["c"], P["omega"], P["tau"], P["d"],
                                           P["gamma"], P["rho"], P["k0"], P["a"])
    ncell = cell_lon.size
    lam = P["mu"] * area_c * G.HORIZON_D                      # uniform background floor
    causal = hist_t < t0d
    ht, hlon, hlat, hmag = hist_t[causal], hist_lon[causal], hist_lat[causal], hist_mag[causal]
    if ht.size == 0:
        return lam
    dt0 = t0d - ht                                            # age at window start (>0)
    dt1 = dt0 + G.HORIZON_D
    tint = _time_integral(dt0, dt1, c, omega, tau)           # per-source time factor
    numf = k0 * np.exp(a * (hmag - MC))                       # productivity
    Dj = d * np.exp(gamma * (hmag - MC))                     # spatial scale
    w = numf * tint                                          # per-source weight
    big = w > 1e-12
    ht2, hlon, hlat, w, Dj = ht[big], hlon[big], hlat[big], w[big], Dj[big]
    if w.size == 0:
        return lam
    meanlat = float(cell_lat.mean())
    coslat = np.cos(np.radians(meanlat))
    chunk = max(1, int(4e6 // ncell))
    for s in range(0, w.size, chunk):
        sl = slice(s, s + chunk)
        dx = (cell_lon[None, :] - hlon[sl, None]) * KM_PER_DEG * coslat
        dy = (cell_lat[None, :] - hlat[sl, None]) * KM_PER_DEG
        r2 = dx * dx + dy * dy
        space = (r2 + Dj[sl, None]) ** (-(1.0 + rho))
        lam = lam + (w[sl, None] * space * area_c[None, :]).sum(axis=0)
    return lam


def compute():
    P = _load_params()
    cat = pd.read_csv(OUT / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    cat = cat[cat["mag_w"] >= MC - 1e-9].reset_index(drop=True)
    hist_t = ((cat["datetime_utc"] - G.REF) / pd.Timedelta(days=1)).to_numpy()
    hist_lon = cat["longitude"].to_numpy(); hist_lat = cat["latitude"].to_numpy()
    hist_mag = cat["mag_w"].to_numpy()

    LO, LA = np.meshgrid(G.LON_C, G.LAT_C)
    cell_lon = LO.ravel(); cell_lat = LA.ravel()
    area_c = (0.1 * KM_PER_DEG) * (0.1 * KM_PER_DEG * np.cos(np.radians(cell_lat)))
    ir_flat = np.repeat(np.arange(G.NLAT), G.NLON)
    ic_flat = np.tile(np.arange(G.NLON), G.NLAT)

    starts = G.window_starts(cat["datetime_utc"].max())
    ks = valtest_window_ids(starts)
    s35 = 10.0 ** (-P["b"] * (3.5 - MC)); s45 = 10.0 ** (-P["b"] * (4.5 - MC))
    rows = []
    t_all = time.time()
    for i, k in enumerate(ks):
        t0d = float(G._to_days(starts[k]))
        lam_mc = lam_mc_window(P, t0d, hist_t, hist_lon, hist_lat, hist_mag,
                               cell_lon, cell_lat, area_c)
        rows.append(pd.DataFrame({
            "window": np.full(G.NCELLS, k), "ir": ir_flat, "ic": ic_flat,
            "lam35": lam_mc * s35, "lam45": lam_mc * s45}))
        if (i + 1) % 10 == 0:
            print(f"  modern_etas window {i+1}/{len(ks)} (k={k}) ({time.time()-t_all:.0f}s)", flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(OUT / "rates_modern_etas.parquet", index=False)
    print(f"wrote rates_modern_etas.parquet: {len(out)} rows, {len(ks)} windows, "
          f"b_miz={P['b']:.3f}, lam35 mean {out['lam35'].mean():.5f} ({time.time()-t_all:.0f}s)")


def validate():
    """Grid-summed expected count for a test event must match the package's
    expected_aftershocks (integrates spatial analytically). Uses a fine local grid."""
    P = _load_params()
    m, dt0, dt1 = 5.0, 2.0, 32.0
    numf = P["k0"] * np.exp(P["a"] * (m - MC))
    Dj = P["d"] * np.exp(P["gamma"] * (m - MC))
    tint = _time_integral(dt0, dt1, P["c"], P["omega"], P["tau"])
    # analytic area_factor = pi*(d*exp(gamma*dm))^-rho / rho
    area_factor = np.pi * Dj ** (-P["rho"]) / P["rho"]
    ea_analytic = numf * area_factor * tint
    # fine grid sum
    xs = np.linspace(-2000, 2000, 4001)
    X, Y = np.meshgrid(xs, xs)
    r2 = X * X + Y * Y
    dA = (xs[1] - xs[0]) ** 2
    ea_grid = numf * tint * ((r2 + Dj) ** (-(1.0 + P["rho"])) * dA).sum()
    print(f"validate: analytic EA={ea_analytic:.5e}  grid-sum EA={ea_grid:.5e}  "
          f"ratio={ea_grid/ea_analytic:.4f} (expect ~1.0)")


if __name__ == "__main__":
    if "--validate" in sys.argv:
        validate()
    else:
        compute()
