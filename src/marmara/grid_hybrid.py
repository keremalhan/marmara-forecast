"""Hybrid feature grid: the 19 features recomputed at base Mc=3.0 with the ETAS
params, PLUS cascade forecaster outputs (lam_sim, P_sim levels) and 30-day cell
COUNTS (Poisson-regression targets). Same 261 windows / 1219 cells as the baseline grid.

Cascade columns are computed at the operational b (b_op) for the y35/y45 hybrid;
the M5/M6 b-ensemble product is produced separately in forecast / synthetic.

Output: results/grid_hybrid.parquet (+ grid_hybrid_report.json)
Run:  "<venv>/bin/python3" -m marmara.grid_hybrid
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401

from marmara import grid as G
from marmara.cascade import cascade_forecast

OUT = RESULTS
BASE_MC = 3.0
K_BACKTEST = 500
CASCADE_COLS = ["lam35_sim", "lam45_sim", "Psim3.5", "Psim5.0", "Psim5.5", "Psim6.0"]


def load_b_op() -> float:
    r = json.load(open(OUT / "etas_fit_report.json"))
    return float(r["operational_b_for_cascade"])


def count_grid(EV, t0d, spec, thr):
    """Actual event count (not capped) per cell in [t0, t0+30) for mag_w>=thr."""
    sub = {3.0: EV["e30"], 3.5: EV["e35"], 4.5: EV["e45"]}[thr]
    lo = np.searchsorted(sub["t"], t0d, "left")
    hi = np.searchsorted(sub["t"], t0d + G.HORIZON_D, "left")
    g = np.zeros((spec.nlat, spec.nlon))
    np.add.at(g, (sub["ir"][lo:hi], sub["ic"][lo:hi]), 1.0)
    return g


def build(b_op: float):
    OUT.mkdir(exist_ok=True)
    with open(OUT / "etas_params.pkl", "rb") as f:
        params = pickle.load(f)
    cat = pd.read_csv(OUT / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    spec = G.MODEL_SPEC

    EV = G.build_event_bundle(cat, BASE_MC)          # features/counts at Mc=3.0
    ctx = G.build_static_context()
    starts = G.window_starts(cat["datetime_utc"].max())
    hist_all = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]

    LO, LA = np.meshgrid(G.LON_C, G.LAT_C)
    flat_lon, flat_lat = LO.ravel(), LA.ravel()
    ir_flat = np.repeat(np.arange(G.NLAT), G.NLON)
    ic_flat = np.tile(np.arange(G.NLON), G.NLAT)

    rows = []
    t_all = time.time()
    for k, t0_dt in enumerate(starts):
        t0d = float(G._to_days(t0_dt))
        feats = G.features_at_window(EV, t0d, t0_dt, ctx, params)   # 19 features (Mc=3.0)
        casc = cascade_forecast(params, hist_all[cat["datetime_utc"] < t0_dt],
                                t0d, G.HORIZON_D, spec.lon_c, spec.lat_c,
                                K=K_BACKTEST, seed=1000 + k, b=b_op)
        y35, y45 = G.targets_at_window(EV, t0d)
        block = {"window": np.full(G.NCELLS, k),
                 "t0": np.full(G.NCELLS, np.datetime64(t0_dt), dtype="datetime64[ns]"),
                 "cell_lon": flat_lon, "cell_lat": flat_lat,
                 "ir": ir_flat, "ic": ic_flat}
        for name in G.FEATURES:
            block[name] = feats[name].ravel()
        block["lam30_sim"] = casc["lam30"].ravel()          # y30 (M>=3.0)
        block["lam35_sim"] = casc["lam35"].ravel()
        block["lam45_sim"] = casc["lam45"].ravel()
        for lvl in (3.5, 5.0, 5.5, 6.0):
            block[f"Psim{lvl}"] = casc[f"P{lvl}"].ravel()
        block["count30"] = count_grid(EV, t0d, spec, 3.0).ravel()
        block["count35"] = count_grid(EV, t0d, spec, 3.5).ravel()
        block["count45"] = count_grid(EV, t0d, spec, 4.5).ravel()
        block["y30"] = G.y30_at_window(EV, t0d).ravel()
        block["y35"] = y35.ravel()
        block["y45"] = y45.ravel()
        rows.append(pd.DataFrame(block))
        if (k + 1) % 40 == 0:
            print(f"  window {k+1}/{len(starts)} {t0_dt.date()} ({time.time()-t_all:.0f}s)")

    grid = pd.concat(rows, ignore_index=True)
    grid.to_parquet(OUT / "grid_hybrid.parquet", index=False)
    report = {
        "base_mc": BASE_MC, "b_op": b_op, "K_backtest": K_BACKTEST,
        "n_rows": int(len(grid)), "n_windows": len(starts),
        "features_19": G.FEATURES, "cascade_cols": CASCADE_COLS,
        "count35_total": int(grid["count35"].sum()),
        "count45_total": int(grid["count45"].sum()),
        "lam35_sim_mean": float(grid["lam35_sim"].mean()),
        "runtime_s": round(time.time() - t_all, 1),
    }
    json.dump(report, open(OUT / "grid_hybrid_report.json", "w"), indent=2)
    print("hybrid grid built:", json.dumps(report, indent=2)[:400])
    return grid


if __name__ == "__main__":
    build(load_b_op())
