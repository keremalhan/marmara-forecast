"""Exhaustive-window version of the leakage self-test (referee item: check a whole window,
not a 26-row sample).

The gate test_grid_leakage spot-checks ~26 (cell, t0) rows. Here we take ONE full forecast
window and recompute EVERY feature for ALL 1,219 model-box cells from a catalogue truncated to
< t0, requiring an exact match to the stored grid (integer counts exact, continuous features to
1e-6). Because every feature is causal, truncation must be a no-op for all cells at once. We pick
the most active test-period window so the count/recency/b-positive families are all exercised.

Writes results/audit/leakage_full_window.json. Reads only.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/verify/leakage_full_window.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from marmara.paths import RESULTS, ROOT
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from marmara.grid import (FEATURES, NLON, REF, build_event_bundle, build_static_context,
                          features_at_window, load_mc, load_params)

CNT_COLS = {"cnt30", "cnt90", "cnt365", "nbr3_cnt30", "nbr3_cnt365",
            "nbr5_cnt30", "nbr5_cnt365", "b_pos_is_fallback"}


def main():
    grid = pd.read_parquet(RESULTS / "grid" / "grid.parquet")
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    mc = load_mc(); params = load_params(); ctx = build_static_context()

    # most active window (max total 30-day count) -> exercises every feature family
    activity = grid.groupby("window")["cnt30"].sum()
    w = int(activity.idxmax())
    gw = grid[grid.window == w].sort_values(["ir", "ic"]).reset_index(drop=True)
    t0_dt = pd.Timestamp(gw["t0"].iloc[0])
    t0d = float((t0_dt - REF) / pd.Timedelta(days=1))

    trunc = cat[cat.datetime_utc < t0_dt]
    EVt = build_event_bundle(trunc, mc)
    feats = features_at_window(EVt, t0d, t0_dt, ctx, params)   # all 1,219 cells

    flat = (gw["ir"].to_numpy() * NLON + gw["ic"].to_numpy())
    n_fail = 0; max_dev = {}; fails = []
    for col in FEATURES:
        got = feats[col].ravel()[flat]
        exp = gw[col].to_numpy(float)
        dev = np.abs(got - exp)
        max_dev[col] = float(dev.max())
        bad = (got != exp) if col in CNT_COLS else ~np.isclose(got, exp, rtol=1e-6, atol=1e-6)
        if bad.any():
            n_fail += int(bad.sum())
            fails.append({"feature": col, "n_bad": int(bad.sum()), "max_dev": float(dev.max())})

    out = {
        "window": w, "t0": str(t0_dt.date()), "n_cells": int(len(gw)),
        "n_features": len(FEATURES), "total_checks": int(len(gw) * len(FEATURES)),
        "window_total_cnt30": int(activity.loc[w]),
        "passed": n_fail == 0, "n_failures": n_fail, "failures": fails,
        "max_abs_deviation_per_feature": {c: max_dev[c] for c in FEATURES},
        "note": ("all 1,219 model-box cells of one full window recomputed from a <t0 truncated "
                 "catalogue and matched to the stored grid; counts exact, continuous to 1e-6."),
    }
    (RESULTS / "audit" / "leakage_full_window.json").write_text(json.dumps(out, indent=2))
    print(f"window {w} (t0 {t0_dt.date()}), {len(gw)} cells x {len(FEATURES)} features "
          f"= {len(gw)*len(FEATURES)} checks")
    print("PASS" if n_fail == 0 else f"FAIL ({n_fail})")
    print("max |deviation| over all cells:", max(max_dev.values()))


if __name__ == "__main__":
    main()
