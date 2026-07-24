"""Run 13 — b_op count-calibration on ALL pre-test windows, not the [::3] subsample.

WHY. marmara/calibrate_b.py runs the sweep on every third pre-test window (77 of 231) -- a speed
choice written into the code before the sweep was ever run, but one that §3 does not disclose and
that leaves the selected b_op resting on a third of the available evidence. The subsample's realized
rate (189.6/yr) also sits ~11% under the catalogue's (213.6/yr), so the subsample is not a perfectly
representative slice.

WHAT. Re-run the sweep over ALL 231 pre-test windows at the bracketing candidates
{1.10, 1.12, 1.15, 1.18} and check whether the argmin moves off 1.15. Everything else is identical
to calibrate_b.py: same params, same K=300, same per-window seed 5000+i, same slope estimator
(regression through the origin of predicted on realized per-window M>=3.5 counts).

Reported unconditionally. Writes results/round4/r13_bop_full_sweep.json. Does not touch
etas_fit_report.json or any frozen artifact.

Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/sensitivity/bop_sweep_full.py
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd

from marmara import grid as G
from marmara.cascade import cascade_forecast
from marmara.paths import RESULTS

K_CAL = 300
CANDIDATES = [1.10, 1.12, 1.15, 1.18]
R4 = RESULTS / "round4"
R4.mkdir(exist_ok=True)


def sweep(params, cat, hist, e35, spec, windows, label):
    out = {}
    for b in CANDIDATES:
        pred, real = [], []
        for i, t0 in enumerate(windows):
            t0d = float(G._to_days(t0))
            c = cascade_forecast(params, hist[cat["datetime_utc"] < t0], t0d, 30.0,
                                 spec.lon_c, spec.lat_c, K=K_CAL, seed=5000 + i, b=b)
            lo = np.searchsorted(e35["t"], t0d, "left")
            hi = np.searchsorted(e35["t"], t0d + 30.0, "left")
            pred.append(float(c["lam35"].sum())); real.append(float(hi - lo))
        pred, real = np.array(pred), np.array(real)
        slope = float((pred * real).sum() / max((pred * pred).sum(), 1e-12))
        out[b] = {"slope": round(slope, 4), "abs_slope_minus_1": round(abs(slope - 1.0), 4),
                  "pred_total": round(float(pred.sum()), 1), "real_total": round(float(real.sum()), 1)}
        print(f"  [{label}] b={b:.2f}: slope {slope:.4f}  pred {pred.sum():.0f} "
              f"real {real.sum():.0f}", flush=True)
    return out


def main():
    t0 = time.time()
    params = pickle.load(open(RESULTS / "etas" / "etas_params.pkl", "rb"))
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    spec = G.MODEL_SPEC
    EV = G.build_event_bundle(cat, 3.0)
    e35 = EV["e35"]
    hist = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]
    starts = G.window_starts(cat["datetime_utc"].max())
    pre = [t for t in starts if t + pd.Timedelta(days=30) <= pd.Timestamp("2024-01-01")]
    sub = pre[::3]
    print(f"pre-test windows: {len(pre)} (all) vs {len(sub)} (the shipped [::3] subsample)\n",
          flush=True)

    res_sub = sweep(params, cat, hist, e35, spec, sub, "subsample [::3]")
    print(flush=True)
    res_all = sweep(params, cat, hist, e35, spec, pre, "ALL windows")

    pick_sub = min(CANDIDATES, key=lambda b: res_sub[b]["abs_slope_minus_1"])
    pick_all = min(CANDIDATES, key=lambda b: res_all[b]["abs_slope_minus_1"])
    out = {
        "why": ("calibrate_b.py sweeps every third pre-test window (a speed choice in code, "
                "predating the sweep); this re-runs the bracketing candidates on all of them"),
        "n_windows": {"all_pre_test": len(pre), "shipped_subsample": len(sub)},
        "candidates": CANDIDATES,
        "subsample_[::3]": {str(b): res_sub[b] for b in CANDIDATES},
        "all_windows": {str(b): res_all[b] for b in CANDIDATES},
        "argmin_subsample": pick_sub,
        "argmin_all_windows": pick_all,
        "argmin_moves": bool(pick_sub != pick_all),
        "shipped_b_op": 1.15,
        "verdict": ("the argmin does NOT move: b_op = 1.15 is selected on all 231 pre-test windows "
                    "exactly as on the [::3] subsample" if pick_all == 1.15 else
                    f"the argmin MOVES to {pick_all} on all windows -- the subsample mattered"),
        "runtime_s": round(time.time() - t0, 1),
    }
    json.dump(out, open(R4 / "r13_bop_full_sweep.json", "w"), indent=2)
    print(f"\n{'b':>6} {'slope [::3]':>12} {'slope ALL':>11} {'|s-1| [::3]':>12} {'|s-1| ALL':>11}")
    for b in CANDIDATES:
        print(f"{b:>6.2f} {res_sub[b]['slope']:>12.4f} {res_all[b]['slope']:>11.4f} "
              f"{res_sub[b]['abs_slope_minus_1']:>12.4f} {res_all[b]['abs_slope_minus_1']:>11.4f}")
    print(f"\nargmin on [::3] subsample : b_op = {pick_sub}")
    print(f"argmin on ALL windows     : b_op = {pick_all}")
    print(f"VERDICT: {out['verdict']}  ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
