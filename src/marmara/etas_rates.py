"""Per-(cell, window) 30-day cascade rates for an ARBITRARY ETAS parameter set,
on the model-box grid, for the val+test windows train.py evaluates.

Turns the sv-ETAS fit and the Mizrahi-inverted params (fallback: modern_params_cascade) into rankable predictors that plug into train.py
exactly like the published `cascade` (lam35_sim): same cascade machinery, same
K=500, same operational b, same per-window seed (1000+k) as grid_hybrid.py.
Only the ETAS PARAMETERS differ, so a sv_etas-vs-cascade (or modern-vs-cascade)
comparison attributes any difference to the parameters alone.

SELF-CHECK: run with the first-gen params (etas_params.pkl, label `firstgen_check`)
and it reproduces grid_hybrid's lam35_sim/lam45_sim on the val+test windows
(bit-close); this confirms the rate computation is faithful to the published cascade.

Memory: one window at a time (peak ~tens of MB); safe on an 8 GB machine. No
concurrency; run these sequentially.

Run:  "<venv>/bin/python" -m marmara.etas_rates <params_pkl> <label>
Output: results/rates_<label>.parquet  [window, ir, ic, lam35, lam45]
"""
from __future__ import annotations

import json
import pickle
import sys
import time

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401

from marmara import grid as G
from marmara.cascade import cascade_forecast

OUT = RESULTS
K_BACKTEST = 500          # matches grid_hybrid.py
B_OP = float(json.load(open(OUT / "etas_fit_report.json"))["operational_b_for_cascade"])  # matches grid_hybrid.py
SEED_BASE = 1000          # per-window seed = SEED_BASE + k (matches grid_hybrid.py)

# splits (train.py): recomputed here without loading the big grid.
TRAIN_END = pd.Timestamp("2022-01-01")
VAL_END = pd.Timestamp("2024-01-01")
TEST_TARGET_END = pd.Timestamp("2026-03-31")


def valtest_window_ids(starts: list[pd.Timestamp]) -> list[int]:
    """Window indices k whose t0 is in the val or test split (train.py convention)."""
    out = []
    for k, t0 in enumerate(starts):
        tgt_end = t0 + pd.Timedelta(days=30)
        is_val = (t0 >= TRAIN_END) & (t0 < VAL_END)
        is_test = (t0 >= VAL_END) & (tgt_end <= TEST_TARGET_END)
        if is_val or is_test:
            out.append(k)
    return out


def compute(params_pkl: str, label: str, b: float = B_OP, K: int = K_BACKTEST):
    with open(params_pkl, "rb") as f:
        params = pickle.load(f)
    cat = pd.read_csv(OUT / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    spec = G.MODEL_SPEC
    starts = G.window_starts(cat["datetime_utc"].max())
    ks = valtest_window_ids(starts)
    hist_all = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]

    ir_flat = np.repeat(np.arange(G.NLAT), G.NLON)
    ic_flat = np.tile(np.arange(G.NLON), G.NLAT)

    rows = []
    t0_all = time.time()
    for i, k in enumerate(ks):
        t0_dt = starts[k]
        t0d = float(G._to_days(t0_dt))
        casc = cascade_forecast(params, hist_all[cat["datetime_utc"] < t0_dt],
                                t0d, G.HORIZON_D, spec.lon_c, spec.lat_c,
                                K=K, seed=SEED_BASE + k, b=b)
        rows.append(pd.DataFrame({
            "window": np.full(G.NCELLS, k),
            "ir": ir_flat, "ic": ic_flat,
            "lam30": casc["lam30"].ravel(),        # M>=3.0 (y30 target)
            "lam35": casc["lam35"].ravel(), "lam45": casc["lam45"].ravel(),
        }))
        del casc
        if (i + 1) % 10 == 0:
            print(f"  {label}: window {i+1}/{len(ks)} (k={k}) "
                  f"({time.time()-t0_all:.0f}s)", flush=True)

    out = pd.concat(rows, ignore_index=True)
    path = OUT / f"rates_{label}.parquet"
    out.to_parquet(path, index=False)
    print(f"wrote {path.name}: {len(out)} rows, {len(ks)} windows, "
          f"lam35 mean {out['lam35'].mean():.4f} ({time.time()-t0_all:.0f}s)")
    return out


def main():
    if len(sys.argv) < 3:
        print("usage: python -m marmara.etas_rates <params_pkl> <label> [b]")
        sys.exit(2)
    params_pkl = sys.argv[1]
    label = sys.argv[2]
    b = float(sys.argv[3]) if len(sys.argv) > 3 else B_OP
    compute(params_pkl, label, b=b)


if __name__ == "__main__":
    main()
