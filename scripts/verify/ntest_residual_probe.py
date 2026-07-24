"""Characterize the surviving M>=3.0 residual after the SURGICAL branching fix
(follow-up to task 3; supersedes the earlier global-rescale probe).

The surgical cascade (real-history parents keep fitted k; only simulated parents are
damped to hold the simulated sub-cascade at n=0.95) forecasts ~1306 M>=3.0 over the
2024-26 test period vs 1383 observed -- a narrow under-prediction. This script asks
WHERE the ~77-event residual lives, with three cheap splits + one sensitivity row:

  (1) sim-cascade branching sweep sim_n_target in {0, 0.95, 0.99, 0.999} -- how much
      the simulated sub-cascade's cap contributes (real parents always at fitted k);
      n=0.999 is the MLE-preferred (re-pinned) value, reported as SENSITIVITY not
      canonical (it is pre-test information and only weakly identified).
  (2) TIME split: windows before vs after the 23 Apr 2025 Kumburgaz Mw6.2. A deficit
      in the quiet (pre) windows is a background NON-STATIONARITY signal; a deficit in
      the post window is aftershock under-triggering.
  (3) SPACE split: cells within R_ZONE_KM of Kumburgaz (aftershock zone) vs the rest
      (background). Same logic, spatially.

Writes results/ntest_residual_probe.{json,md}. Reads only.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/verify/ntest_residual_probe.py
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
KUMBURGAZ = (28.23, 40.84)                 # lon, lat of the 23 Apr 2025 Mw6.2
SPLIT_DATE = pd.Timestamp("2025-04-23")
R_ZONE_KM = 60.0
KM_PER_DEG = 111.19
OBS_RAW = 1383


def main():
    t_start = time.time()
    grid = pd.read_parquet(OUT / "grid" / "grid_hybrid.parquet")
    test_wins = [int(w) for w in np.sort(grid.loc[split_masks(grid)["test"], "window"].unique())]
    win_t0 = grid.groupby("window")["t0"].first()

    with open(OUT / "etas" / "etas_params.pkl", "rb") as f:
        params = pickle.load(f)
    cat = pd.read_csv(OUT / "catalog" / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    spec = G.MODEL_SPEC
    hist_all = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]

    windows = [(w, pd.Timestamp(win_t0.loc[w]), float(G._to_days(pd.Timestamp(win_t0.loc[w]))),
                hist_all[cat["datetime_utc"] < pd.Timestamp(win_t0.loc[w])]) for w in test_wins]

    # ---- canonical surgical pass (sim_n_target=None -> 0.95): splits ----
    F_cell = np.zeros((G.NLAT, G.NLON))
    F_pre = F_post = 0.0
    F30_canon = F35_canon = 0.0
    for (w, t0_dt, t0d, h) in windows:
        casc = cascade_forecast(params, h, t0d, G.HORIZON_D, spec.lon_c, spec.lat_c,
                                K=K, seed=1000 + w, b=B_OP, preserve_branching=True)
        F_cell += casc["lam30"]
        s = float(casc["lam30"].sum())
        F30_canon += s
        F35_canon += float(casc["lam35"].sum())
        if t0_dt < SPLIT_DATE:
            F_pre += s
        else:
            F_post += s
    print(f"canonical surgical: F30={F30_canon:.1f} F35={F35_canon:.1f} ({time.time()-t_start:.0f}s)")

    # ---- observed per cell + pre/post ----
    t0s = pd.to_datetime([win_t0.loc[w] for w in test_wins])
    lo, hi = t0s.min(), t0s.max() + pd.Timedelta(days=G.HORIZON_D)
    ev = cat[(cat["datetime_utc"] >= lo) & (cat["datetime_utc"] < hi) & (cat["mag_w"] >= 3.0 - 1e-9)]
    iir, iic = G.cell_index(ev["longitude"].to_numpy(), ev["latitude"].to_numpy())
    on = iir >= 0
    ev = ev[on]; iir, iic = iir[on], iic[on]
    O_cell = np.zeros((G.NLAT, G.NLON))
    np.add.at(O_cell, (iir, iic), 1.0)
    O_pre = int((ev["datetime_utc"] < SPLIT_DATE).sum())
    O_post = int((ev["datetime_utc"] >= SPLIT_DATE).sum())

    # ---- space masks (distance of each grid cell centre to Kumburgaz) ----
    LO, LA = np.meshgrid(G.LON_C, G.LAT_C)
    dx = (LO - KUMBURGAZ[0]) * KM_PER_DEG * np.cos(np.radians(KUMBURGAZ[1]))
    dy = (LA - KUMBURGAZ[1]) * KM_PER_DEG
    zone = np.sqrt(dx * dx + dy * dy) < R_ZONE_KM

    def rr(x):
        return round(float(x), 1)
    splits = {
        "time": {
            "pre_Kumburgaz": {"forecast": rr(F_pre), "observed": O_pre,
                              "ratio": rr(F_pre / max(O_pre, 1))},
            "post_Kumburgaz": {"forecast": rr(F_post), "observed": O_post,
                               "ratio": rr(F_post / max(O_post, 1))},
        },
        "space": {
            "aftershock_zone_<60km": {"forecast": rr(F_cell[zone].sum()),
                                      "observed": int(O_cell[zone].sum()),
                                      "ratio": rr(F_cell[zone].sum() / max(O_cell[zone].sum(), 1))},
            "background_rest": {"forecast": rr(F_cell[~zone].sum()),
                                "observed": int(O_cell[~zone].sum()),
                                "ratio": rr(F_cell[~zone].sum() / max(O_cell[~zone].sum(), 1))},
        },
    }

    # ---- sim-cascade branching sweep (real parents stay at fitted k) ----
    sweep = [{"n": 0.95, "F_M3.0": rr(F30_canon), "F_M3.5": rr(F35_canon), "note": "canonical"}]
    for n_t in (0.0, 0.99, 0.999):
        F30 = F35 = 0.0
        for (w, t0_dt, t0d, h) in windows:
            casc = cascade_forecast(params, h, t0d, G.HORIZON_D, spec.lon_c, spec.lat_c,
                                    K=K, seed=1000 + w, b=B_OP, preserve_branching=True,
                                    sim_n_target=n_t)
            F30 += float(casc["lam30"].sum()); F35 += float(casc["lam35"].sum())
        note = ("no simulated sub-cascade (bg + real-parent 1st-gen only)" if n_t == 0.0
                else "MLE-preferred, SENSITIVITY only" if n_t == 0.999 else "sensitivity")
        sweep.append({"n": n_t, "F_M3.0": rr(F30), "F_M3.5": rr(F35), "note": note})
        print(f"  sim_n={n_t}: F30={F30:.1f} F35={F35:.1f} ({time.time()-t_start:.0f}s)")
    sweep.sort(key=lambda s: s["n"])

    result = {
        "meta": {"b_op": B_OP, "K": K, "n_windows": len(test_wins),
                 "split_date": str(SPLIT_DATE.date()), "zone_km": R_ZONE_KM,
                 "runtime_s": round(time.time() - t_start, 1)},
        "observed": {"raw_total": OBS_RAW, "on_grid_total": int(O_cell.sum()),
                     "pre_Kumburgaz": O_pre, "post_Kumburgaz": O_post},
        "canonical_surgical_M3.0": rr(F30_canon),
        "sim_branching_sweep": sweep,
        "deficit_splits": splits,
    }
    (OUT / "ntest_residual_probe.json").write_text(json.dumps(result, indent=2))
    _write_md(result, OUT / "ntest_residual_probe.md")
    print("\n" + json.dumps({"sweep": sweep, "splits": splits}, indent=2))


def _write_md(r, path):
    sp = r["deficit_splits"]
    L = ["# Surgical-cascade M>=3.0 residual: where does the ~77-event deficit live?", "",
         f"b_op={r['meta']['b_op']}, K={r['meta']['K']}, {r['meta']['n_windows']} test windows. "
         f"Canonical surgical forecast **{r['canonical_surgical_M3.0']}** vs observed "
         f"{r['observed']['raw_total']} (on-grid {r['observed']['on_grid_total']}).", "",
         "## Simulated-sub-cascade branching sweep (real parents fixed at fitted k)", "",
         "| sim n | forecast M>=3.0 | forecast M>=3.5 | note |", "|---|---|---|---|"]
    for s in r["sim_branching_sweep"]:
        L.append(f"| {s['n']} | {s['F_M3.0']} | {s['F_M3.5']} | {s['note']} |")
    L += ["",
          "## Time split (Kumburgaz Mw6.2 = 2025-04-23)", "",
          "| window set | forecast | observed | fcast/obs |", "|---|---|---|---|",
          f"| pre-Kumburgaz | {sp['time']['pre_Kumburgaz']['forecast']} | "
          f"{sp['time']['pre_Kumburgaz']['observed']} | {sp['time']['pre_Kumburgaz']['ratio']} |",
          f"| post-Kumburgaz | {sp['time']['post_Kumburgaz']['forecast']} | "
          f"{sp['time']['post_Kumburgaz']['observed']} | {sp['time']['post_Kumburgaz']['ratio']} |",
          "",
          f"## Space split (< {r['meta']['zone_km']} km of Kumburgaz)", "",
          "| region | forecast | observed | fcast/obs |", "|---|---|---|---|",
          f"| aftershock zone | {sp['space']['aftershock_zone_<60km']['forecast']} | "
          f"{sp['space']['aftershock_zone_<60km']['observed']} | {sp['space']['aftershock_zone_<60km']['ratio']} |",
          f"| background rest | {sp['space']['background_rest']['forecast']} | "
          f"{sp['space']['background_rest']['observed']} | {sp['space']['background_rest']['ratio']} |"]
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
