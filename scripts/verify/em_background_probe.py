"""EM-background split probe (reviewer steer, 2026-07-12): does swapping the
sv-ETAS EM-declustered background into the cascade collapse the quiet-period
over-prediction, or is it a genuine temporal non-stationarity?

The surgical cascade (first-gen params) over/under-predicts in a canceling way:
quiet/background OVER (pre-Kumburgaz 735 vs 501, ~1.5x), aftershock zone UNDER
(339 vs 692, ~0.5x). sv-ETAS has NEAR-IDENTICAL triggering (k,alpha,c,p,d,q,gamma)
but a different, EM-estimated background (mu_total 0.6834 vs 0.7027; EM-KDE spatial
structure). So running the cascade with sv-ETAS params isolates the BACKGROUND lever.

Recomputes ONLY the split table (no rankings, no bootstrap). Decision:
  * quiet over-prediction collapses toward 1x  -> EM-mu cascade is canonical v2;
  * it persists                                -> temporal non-stationarity residual
    (2024-26 quieter than the 2003-21 stationary fit), stationary-mu stays canonical
    with the EM row as a sensitivity, and the residual ties to the eastward-migration
    claim.

Both configs use the SURGICAL fix (preserve_branching=True: real-history parents keep
each fit's own k; only simulated parents are damped). Writes results/em_background_probe.{json,md}.
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd

from marmara.paths import RESULTS
from marmara import grid as G
from marmara.train import split_masks
from marmara.cascade import cascade_forecast

OUT = RESULTS
B_OP = 1.15
K = 500
KUMBURGAZ = (28.23, 40.84)
SPLIT_DATE = pd.Timestamp("2025-04-23")
R_ZONE_KM = 60.0
KM_PER_DEG = 111.19


def run_config(params, windows):
    F_cell = np.zeros((G.NLAT, G.NLON))
    F_pre = F_post = F30 = F35 = 0.0
    for (w, t0_dt, t0d, h) in windows:
        casc = cascade_forecast(params, h, t0d, G.HORIZON_D, spec.lon_c, spec.lat_c,
                                K=K, seed=1000 + w, b=B_OP, preserve_branching=True)
        F_cell += casc["lam30"]
        s = float(casc["lam30"].sum())
        F30 += s
        F35 += float(casc["lam35"].sum())
        if t0_dt < SPLIT_DATE:
            F_pre += s
        else:
            F_post += s
    return F_cell, F_pre, F_post, F30, F35


if __name__ == "__main__":
    t_start = time.time()
    grid = pd.read_parquet(OUT / "grid" / "grid_hybrid.parquet")
    test_wins = [int(w) for w in np.sort(grid.loc[split_masks(grid)["test"], "window"].unique())]
    win_t0 = grid.groupby("window")["t0"].first()
    cat = pd.read_csv(OUT / "catalog" / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    spec = G.MODEL_SPEC
    hist_all = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]
    windows = [(w, pd.Timestamp(win_t0.loc[w]), float(G._to_days(pd.Timestamp(win_t0.loc[w]))),
                hist_all[cat["datetime_utc"] < pd.Timestamp(win_t0.loc[w])]) for w in test_wins]

    # observed splits (config-independent)
    t0s = pd.to_datetime([win_t0.loc[w] for w in test_wins])
    lo, hi = t0s.min(), t0s.max() + pd.Timedelta(days=G.HORIZON_D)
    ev = cat[(cat["datetime_utc"] >= lo) & (cat["datetime_utc"] < hi) & (cat["mag_w"] >= 3.0 - 1e-9)]
    iir, iic = G.cell_index(ev["longitude"].to_numpy(), ev["latitude"].to_numpy())
    on = iir >= 0; ev = ev[on]; iir, iic = iir[on], iic[on]
    O_cell = np.zeros((G.NLAT, G.NLON)); np.add.at(O_cell, (iir, iic), 1.0)
    O_pre = int((ev["datetime_utc"] < SPLIT_DATE).sum()); O_post = int((ev["datetime_utc"] >= SPLIT_DATE).sum())
    LO, LA = np.meshgrid(G.LON_C, G.LAT_C)
    dx = (LO - KUMBURGAZ[0]) * KM_PER_DEG * np.cos(np.radians(KUMBURGAZ[1])); dy = (LA - KUMBURGAZ[1]) * KM_PER_DEG
    zone = np.sqrt(dx * dx + dy * dy) < R_ZONE_KM
    O_zone = int(O_cell[zone].sum()); O_bg = int(O_cell[~zone].sum())

    configs = {"firstgen_stationary_mu": OUT / "etas" / "etas_params.pkl",
               "sv_etas_EM_background": OUT / "etas" / "etas_sv_params.pkl"}
    out = {"meta": {"b_op": B_OP, "K": K, "n_windows": len(test_wins),
                    "split_date": str(SPLIT_DATE.date()), "zone_km": R_ZONE_KM},
           "observed": {"total": int(O_cell.sum()), "pre": O_pre, "post": O_post,
                        "zone": O_zone, "background": O_bg},
           "configs": {}}
    for name, pkl in configs.items():
        with open(pkl, "rb") as f:
            params = pickle.load(f)
        F_cell, F_pre, F_post, F30, F35 = run_config(params, windows)
        out["configs"][name] = {
            "mu_total": round(float(params.mu_total), 4),
            "F_total_M3.0": round(F30, 1), "F_total_M3.5": round(F35, 1),
            "pre":  {"f": round(F_pre, 1),  "ratio": round(F_pre / max(O_pre, 1), 2)},
            "post": {"f": round(F_post, 1), "ratio": round(F_post / max(O_post, 1), 2)},
            "zone": {"f": round(float(F_cell[zone].sum()), 1),  "ratio": round(F_cell[zone].sum() / max(O_zone, 1), 2)},
            "background": {"f": round(float(F_cell[~zone].sum()), 1), "ratio": round(F_cell[~zone].sum() / max(O_bg, 1), 2)},
        }
        print(f"{name}: mu={params.mu_total:.4f} F30={F30:.0f} "
              f"pre {F_pre:.0f}/{O_pre}={F_pre/O_pre:.2f} post {F_post:.0f}/{O_post}={F_post/O_post:.2f} "
              f"zone {F_cell[zone].sum():.0f}/{O_zone}={F_cell[zone].sum()/O_zone:.2f} "
              f"bg {F_cell[~zone].sum():.0f}/{O_bg}={F_cell[~zone].sum()/O_bg:.2f} ({time.time()-t_start:.0f}s)")

    out["meta"]["runtime_s"] = round(time.time() - t_start, 1)
    (OUT / "em_background_probe.json").write_text(json.dumps(out, indent=2))
    lines = ["# EM-background split probe: does re-estimating the background fix the misallocation?", "",
             f"Observed (M>=3.0, {len(test_wins)} windows): total {int(O_cell.sum())}, "
             f"pre-Kumburgaz {O_pre}, post {O_post}, aftershock-zone {O_zone}, background {O_bg}.", "",
             "| config | mu_total | F total | pre f/o | post f/o | zone f/o | bg f/o |",
             "|---|---|---|---|---|---|---|"]
    for name, c in out["configs"].items():
        lines.append(f"| {name} | {c['mu_total']} | {c['F_total_M3.0']} | "
                     f"{c['pre']['f']}/{O_pre}={c['pre']['ratio']} | {c['post']['f']}/{O_post}={c['post']['ratio']} | "
                     f"{c['zone']['f']}/{O_zone}={c['zone']['ratio']} | {c['background']['f']}/{O_bg}={c['background']['ratio']} |")
    (OUT / "em_background_probe.md").write_text("\n".join(lines))
    print("\n" + json.dumps(out["configs"], indent=2))
