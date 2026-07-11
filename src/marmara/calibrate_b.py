"""Pick the operational cascade b (b_op) by real-window calibration.

b-positive (1.54) is inflated by the Md/ML magnitude mixture; using it in the
cascade under-predicts M>=3.5 counts. We choose b_op so the cascade's predicted
per-window M>=3.5 count matches realized counts (regression slope near 1) over the
reviewed windows, then record it in etas_fit_report.json for the hybrid grid.

Run:  "<venv>/bin/python3" -m marmara.calibrate_b
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401

from marmara import grid as G
from marmara.cascade import cascade_forecast

OUT = RESULTS
B_CANDIDATES = [0.9, 1.0, 1.05, 1.1, 1.12, 1.15, 1.18, 1.2, 1.542]


def main():
    with open(OUT / "etas_params.pkl", "rb") as f:
        params = pickle.load(f)
    cat = pd.read_csv(OUT / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    spec = G.MODEL_SPEC
    EV = G.build_event_bundle(cat, 3.0)
    hist = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]
    starts = G.window_starts(cat["datetime_utc"].max())
    # calibrate on pre-test windows only (target fully before the test start 2024-01-01),
    # every 3rd for speed, so the operational b never sees the test period.
    starts = [t for t in starts if t + pd.Timedelta(days=30) <= pd.Timestamp("2024-01-01")][::3]

    e35 = EV["e35"]
    results = {}
    for b in B_CANDIDATES:
        pred, real = [], []
        for i, t0 in enumerate(starts):
            t0d = float(G._to_days(t0))
            out = cascade_forecast(params, hist[cat["datetime_utc"] < t0], t0d, 30.0,
                                   spec.lon_c, spec.lat_c, K=300, seed=5000 + i, b=b)
            lo = np.searchsorted(e35["t"], t0d, "left"); hi = np.searchsorted(e35["t"], t0d + 30.0, "left")
            pred.append(float(out["lam35"].sum())); real.append(float(hi - lo))
        pred = np.array(pred); real = np.array(real)
        slope = float((pred * real).sum() / max((pred * pred).sum(), 1e-12))
        results[b] = {"slope": slope, "pred_total": float(pred.sum()), "real_total": float(real.sum())}
        print(f"  b={b:.3f}: slope {slope:.3f}  pred {pred.sum():.0f} real {real.sum():.0f}")

    # choose b_op minimizing |slope-1| within [0.8,1.2]; else closest to 1
    best = min(B_CANDIDATES, key=lambda b: abs(results[b]["slope"] - 1.0))
    r = json.load(open(OUT / "etas_fit_report.json"))
    r["operational_b_for_cascade"] = float(best)
    r["b_calibration"] = {str(b): results[b] for b in B_CANDIDATES}
    json.dump(r, open(OUT / "etas_fit_report.json", "w"), indent=2)
    print(f"\nchosen operational b_op = {best} (slope {results[best]['slope']:.3f})")


if __name__ == "__main__":
    main()
