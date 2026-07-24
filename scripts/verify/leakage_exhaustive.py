"""Run 2 — EXHAUSTIVE leakage self-test.

The shipped gate spot-checks 26 (cell, t0) rows (results/audit/leakage_ok.json); the round-3
extension checked ONE full window (1,219 cells x 19 features = 23,161 checks). This run
recomputes EVERY stored cell-window in train/val/test -- all 19 features, every cell, every
window -- from a catalogue truncated to < t0, and requires an exact match to the stored grid.

Because every feature is causal by construction, truncation must be a no-op EVERYWHERE, not
just on a sample. This upgrades the paper's claim from "spot-check" to "exhaustive".

Gates (pre-registered for this run):
  * integer features (counts / flags): max deviation == 0
  * real-valued features:              max deviation <= 1e-6
  * max |feature-target correlation| < 0.999   (a feature that IS the target would leak)

Writes results/round4/r2_leakage_exhaustive.json. Reads only.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/verify/leakage_exhaustive.py [n_windows_limit]
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from marmara.paths import RESULTS, ROOT
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from marmara.grid import (FEATURES, NLON, REF, build_event_bundle, build_static_context,
                          features_at_window, load_mc, load_params)
from marmara.train import split_masks

# integer-valued families: must match EXACTLY (deviation 0), no tolerance
INT_COLS = {"cnt30", "cnt90", "cnt365", "nbr3_cnt30", "nbr3_cnt365",
            "nbr5_cnt30", "nbr5_cnt365", "b_pos_is_fallback"}
REAL_COLS = [c for c in FEATURES if c not in INT_COLS]

OUT = RESULTS / "round4"
OUT.mkdir(exist_ok=True)


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    t_all = time.time()

    # grid_hybrid.parquet is the grid the models actually consume (marmara.train reads it);
    # it is a superset of grid.parquet (one extra forecast-tail window).
    #
    # Mc: grid_hybrid.build() counts features at its own BASE_MC (3.0, per §3 "the 19 grid
    # features (at Mc=3.0)"), NOT at catalog_report's Mc=3.65 that grid.build()/load_mc()
    # use. Recomputing at the wrong Mc produces a spurious mismatch, so we take the Mc
    # from the report of the grid under test.
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    mc = float(json.load(open(RESULTS / "grid" / "grid_hybrid_report.json"))["base_mc"])
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    params = load_params(); ctx = build_static_context()
    print(f"grid under test: grid_hybrid.parquet   base_mc={mc} "
          f"(catalog_report mc={load_mc()}, used by grid.parquet)", flush=True)

    masks = split_masks(grid)
    in_split = masks["train"] | masks["val"] | masks["test"]
    split_of = np.where(masks["train"], "train",
                        np.where(masks["val"], "val",
                                 np.where(masks["test"], "test", "tail")))
    grid = grid.assign(_split=split_of)
    gs = grid[in_split].copy()
    wins = np.sort(gs["window"].unique())
    if limit:
        wins = wins[:limit]
    print(f"grid rows {len(grid)}, in-split {int(in_split.sum())}, windows {len(wins)}", flush=True)

    # pre-sort catalogue once; truncation is then a searchsorted slice of a sorted frame
    cat = cat.sort_values("datetime_utc").reset_index(drop=True)
    cat_t = cat["datetime_utc"].to_numpy()

    max_int_dev = 0.0
    max_real_dev = 0.0
    per_feature_max = {c: 0.0 for c in FEATURES}
    n_checks = 0
    n_fail = 0
    failures = []
    per_split_windows = {"train": 0, "val": 0, "test": 0}
    worst = {"int": None, "real": None}

    for wi, w in enumerate(wins):
        gw = gs[gs.window == w].sort_values(["ir", "ic"]).reset_index(drop=True)
        t0_dt = pd.Timestamp(gw["t0"].iloc[0])
        t0d = float((t0_dt - REF) / pd.Timedelta(days=1))
        per_split_windows[gw["_split"].iloc[0]] += 1

        cut = int(np.searchsorted(cat_t, np.datetime64(t0_dt), side="left"))
        trunc = cat.iloc[:cut]
        EVt = build_event_bundle(trunc, mc)
        feats = features_at_window(EVt, t0d, t0_dt, ctx, params)

        flat = (gw["ir"].to_numpy() * NLON + gw["ic"].to_numpy())
        for col in FEATURES:
            got = feats[col].ravel()[flat]
            exp = gw[col].to_numpy(float)
            dev = np.abs(got - exp)
            dmax = float(dev.max()) if dev.size else 0.0
            if dmax > per_feature_max[col]:
                per_feature_max[col] = dmax
            if col in INT_COLS:
                bad = (got != exp)
                if dmax > max_int_dev:
                    max_int_dev = dmax
                    worst["int"] = {"feature": col, "window": int(w), "t0": str(t0_dt.date()),
                                    "dev": dmax}
            else:
                bad = ~np.isclose(got, exp, rtol=1e-6, atol=1e-6)
                if dmax > max_real_dev:
                    max_real_dev = dmax
                    worst["real"] = {"feature": col, "window": int(w), "t0": str(t0_dt.date()),
                                     "dev": dmax}
            n_checks += int(len(gw))
            if bad.any():
                n_fail += int(bad.sum())
                if len(failures) < 50:
                    failures.append({"feature": col, "window": int(w), "t0": str(t0_dt.date()),
                                     "n_bad": int(bad.sum()), "max_dev": dmax})
        if (wi + 1) % 20 == 0:
            print(f"  {wi+1}/{len(wins)} windows, {n_checks:,} checks, "
                  f"max_int={max_int_dev:g} max_real={max_real_dev:g} "
                  f"({time.time()-t_all:.0f}s)", flush=True)

    # ---- feature-target correlation tripwire ----
    # A feature that IS (a monotone image of) the target would leak. Computed over the same
    # in-split rows, against every scored target present on the grid.
    print("feature-target correlations ...", flush=True)
    tgt_cols = [c for c in ("y30", "y35", "y45", "count30", "count35", "count45")
                if c in gs.columns]
    corrs = {}
    max_abs_corr = 0.0
    max_corr_pair = None
    for tc in tgt_cols:
        tv = gs[tc].to_numpy(float)
        corrs[tc] = {}
        for c in FEATURES:
            fv = gs[c].to_numpy(float)
            if np.std(fv) < 1e-15 or np.std(tv) < 1e-15:
                r = 0.0
            else:
                r = float(np.corrcoef(fv, tv)[0, 1])
            r = 0.0 if not np.isfinite(r) else r
            corrs[tc][c] = round(r, 6)
            if abs(r) > max_abs_corr:
                max_abs_corr = abs(r)
                max_corr_pair = {"feature": c, "target": tc, "corr": round(r, 6)}

    # ---- documented (expected) difference: grid.parquet vs grid_hybrid.parquet ----
    # NOT a gate. The two grids count features at different Mc by design (3.65 vs 3.0), so
    # the Mc-sensitive families differ. Recorded so the difference is on the record and is
    # never mistaken for a leak.
    xg = {"checked": False}
    try:
        g0 = pd.read_parquet(RESULTS / "grid" / "grid.parquet")
        key = ["window", "ir", "ic"]
        a = g0.set_index(key)[FEATURES].sort_index()
        b = grid.set_index(key)[FEATURES].sort_index()
        common = a.index.intersection(b.index)
        per = {c: float(np.abs(a.loc[common, c].to_numpy(float)
                               - b.loc[common, c].to_numpy(float)).max()) for c in FEATURES}
        xg = {"checked": True, "n_common_rows": int(len(common)),
              "grid_parquet_mc": load_mc(), "grid_hybrid_mc": mc,
              "note": ("expected: the two grids count features at different Mc by design; "
                       "Mc-insensitive features must still agree exactly"),
              "features_differing": {c: v for c, v in per.items() if v > 1e-9},
              "features_identical": sorted([c for c, v in per.items() if v <= 1e-9])}
    except Exception as e:                                    # non-fatal: diagnostic only
        xg = {"checked": False, "error": str(e)}

    gate_int = (max_int_dev == 0.0)
    gate_real = (max_real_dev <= 1e-6)
    gate_corr = (max_abs_corr < 0.999)

    out = {
        "scope": ("every stored cell-window in train/val/test, all 19 features, recomputed "
                  "from a catalogue truncated to < t0 and matched against the stored grid"),
        "n_windows": int(len(wins)),
        "windows_per_split": per_split_windows,
        "n_cells_per_window": int(len(gs[gs.window == wins[0]])),
        "n_features": len(FEATURES),
        "total_checks": int(n_checks),
        "n_failures": int(n_fail),
        "failures": failures,
        "max_integer_deviation": max_int_dev,
        "max_real_deviation": max_real_dev,
        "worst_offenders": worst,
        "max_abs_feature_target_correlation": round(max_abs_corr, 6),
        "max_corr_pair": max_corr_pair,
        "base_mc_of_grid_under_test": mc,
        "grid_under_test": "results/grid/grid_hybrid.parquet",
        "cross_grid_note": xg,
        "gates": {
            "integer_deviation_is_0": bool(gate_int),
            "real_deviation_le_1e-6": bool(gate_real),
            "max_abs_corr_lt_0.999": bool(gate_corr),
            "ALL_PASS": bool(gate_int and gate_real and gate_corr),
        },
        "per_feature_max_deviation": {c: per_feature_max[c] for c in FEATURES},
        "int_features": sorted(INT_COLS),
        "real_features": REAL_COLS,
        "feature_target_correlations": corrs,
        "runtime_s": round(time.time() - t_all, 1),
    }
    json.dump(out, open(OUT / "r2_leakage_exhaustive.json", "w"), indent=2)
    print(f"\ntotal checks   : {n_checks:,}")
    print(f"failures       : {n_fail}")
    print(f"max int dev    : {max_int_dev:g}   (gate: == 0)      {'PASS' if gate_int else 'FAIL'}")
    print(f"max real dev   : {max_real_dev:g}  (gate: <= 1e-6)   {'PASS' if gate_real else 'FAIL'}")
    print(f"max |corr|     : {max_abs_corr:.6f} (gate: < 0.999)  {'PASS' if gate_corr else 'FAIL'}")
    print(f"  -> {max_corr_pair}")
    print(f"runtime        : {out['runtime_s']}s")
    print(f"\nwrote results/round4/r2_leakage_exhaustive.json")


if __name__ == "__main__":
    main()
