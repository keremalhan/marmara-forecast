"""Task 7 hard gate: leakage self-test for the feature grid.

For 20 random (cell, t0) rows spread over the period (plus the most active
rows, to exercise b-positive / days-since), rebuild every feature from a
catalog TRUNCATED to events strictly < t0 and require it to reproduce the
stored row (exact for counts, allclose for continuous features). Because every
feature is defined causally, truncation must be a no-op; any accidental use of
events >= t0 would change the value and fail here.

Also asserts no feature column correlates > 0.999 with either target.

Run:  "<venv>/bin/python3" python -m marmara.tests.test_grid_leakage.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marmara.grid import (  # noqa: E402
    FEATURES, LAT_C, LON_C, NLAT, NLON, REF, DAYS_CAP,
    build_event_bundle, build_static_context, cell_index,
    features_at_window, load_mc, load_params, targets_at_window,
)

OUT = RESULTS
RNG = np.random.default_rng(42)
CNT_COLS = {"cnt30", "cnt90", "cnt365", "nbr3_cnt30", "nbr3_cnt365",
            "nbr5_cnt30", "nbr5_cnt365", "b_pos_is_fallback"}


def _brute_counts(cat, mc, t0_dt, ir, ic):
    """Independent brute-force recomputation of count-family features + targets
    from the raw (truncated) catalog for one cell."""
    t0 = pd.Timestamp(t0_dt)
    d = cat.copy()
    d["datetime_utc"] = pd.to_datetime(d["datetime_utc"])
    jr, jc = cell_index(d["longitude"].to_numpy(), d["latitude"].to_numpy())
    d = d.assign(ir=jr, ic=jc)
    d = d[(d.ir >= 0) & (d.ic >= 0)]

    def in_cell(dd):
        return dd[(dd.ir == ir) & (dd.ic == ic)]

    def nbhd(dd, rad):
        return dd[(np.abs(dd.ir - ir) <= rad) & (np.abs(dd.ic - ic) <= rad)]

    past = d[d.datetime_utc < t0]
    mc_ev = past[past.mag_w >= mc]

    def cnt(days):
        w = mc_ev[mc_ev.datetime_utc >= t0 - pd.Timedelta(days=days)]
        return float(len(in_cell(w)))

    out = {"cnt30": cnt(30), "cnt90": cnt(90), "cnt365": cnt(365)}

    def days_since(thr):
        ev = nbhd(past[past.mag_w >= thr], 2)
        if len(ev) == 0:
            return DAYS_CAP
        last = ev.datetime_utc.max()
        return float(min((t0 - last) / pd.Timedelta(days=1), DAYS_CAP))

    out["days_since_m35_25km"] = days_since(3.5)
    out["days_since_m45_25km"] = days_since(4.5)

    all365 = nbhd(past[past.datetime_utc >= t0 - pd.Timedelta(days=365)], 2)
    out["maxmag365_25km"] = float(all365.mag_w.max()) if len(all365) else 0.0

    # targets
    fut = d[(d.datetime_utc >= t0) & (d.datetime_utc < t0 + pd.Timedelta(days=30))]
    out["y35"] = float((len(in_cell(fut[fut.mag_w >= 3.5])) > 0))
    out["y45"] = float((len(in_cell(fut[fut.mag_w >= 4.5])) > 0))
    return out


def main():
    grid = pd.read_parquet(OUT / "grid" / "grid.parquet")
    cat = pd.read_csv(OUT / "catalog" / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    mc = load_mc()
    params = load_params()
    ctx = build_static_context()

    # choose rows: 20 random spread across windows + the 6 most active rows.
    n_win = int(grid["window"].max()) + 1
    picks = []
    for w in np.linspace(0, n_win - 1, 20).astype(int):
        sub = grid[grid.window == w]
        picks.append(sub.iloc[int(RNG.integers(len(sub)))])
    active = grid.sort_values("cnt365", ascending=False).head(6)
    rows = pd.concat([pd.DataFrame(picks), active]).drop_duplicates(
        subset=["window", "ir", "ic"]).reset_index(drop=True)

    n_fail = 0
    max_dev = {}
    for _, r in rows.iterrows():
        t0_dt = pd.Timestamp(r["t0"])
        t0d = float((t0_dt - REF) / pd.Timedelta(days=1))
        ir, ic = int(r["ir"]), int(r["ic"])
        flat = ir * NLON + ic

        # (1) truncate catalog to < t0, rebuild features via the SAME function.
        trunc = cat[cat.datetime_utc < t0_dt]
        EVt = build_event_bundle(trunc, mc)
        feats = features_at_window(EVt, t0d, t0_dt, ctx, params)
        y35, y45 = targets_at_window(build_event_bundle(cat, mc), t0d)

        for col in FEATURES:
            got = float(feats[col].ravel()[flat])
            exp = float(r[col])
            if col in CNT_COLS:
                ok = (got == exp)
            else:
                ok = np.isclose(got, exp, rtol=1e-6, atol=1e-6)
            if not ok:
                n_fail += 1
                print(f"  LEAK/MISMATCH w{int(r['window'])} cell({ir},{ic}) "
                      f"{col}: stored={exp} recomputed={got}")
            max_dev[col] = max(max_dev.get(col, 0.0), abs(got - exp))

        # (2) independent brute-force counts/targets from truncated catalog.
        bf = _brute_counts(cat, mc, t0_dt, ir, ic)
        for col, val in bf.items():
            exp = float(r[col])
            if col.startswith("days_since") or col == "maxmag365_25km":
                ok = np.isclose(val, exp, atol=1e-6)
            else:
                ok = (val == exp)
            if not ok:
                n_fail += 1
                print(f"  BRUTE MISMATCH w{int(r['window'])} cell({ir},{ic}) "
                      f"{col}: grid={exp} brute={val}")

    print(f"\n  checked {len(rows)} rows; max abs deviation per feature:")
    for c in FEATURES:
        print(f"    {c:22s} {max_dev.get(c,0):.3e}")

    # (3) no feature ~identical to a target.
    print("\n  |corr(feature, target)|:")
    corr_fail = 0
    for tgt in ("y35", "y45"):
        yv = grid[tgt].to_numpy()
        for col in FEATURES:
            xv = grid[col].to_numpy()
            if np.std(xv) == 0:
                c = 0.0
            else:
                c = abs(np.corrcoef(xv, yv)[0, 1])
            if c > 0.999:
                corr_fail += 1
                print(f"    !! {col} vs {tgt}: {c:.4f} (>0.999)")
        top = sorted(((abs(np.corrcoef(grid[col], yv)[0, 1]) if np.std(grid[col]) > 0 else 0.0, col)
                      for col in FEATURES), reverse=True)[:3]
        print(f"    {tgt} top-3: " + ", ".join(f"{c:.3f} {n}" for c, n in top))

    total_fail = n_fail + corr_fail
    if total_fail == 0:
        import json
        with open(OUT / "audit" / "leakage_ok.json", "w") as f:
            json.dump({"passed": True, "n_rows_checked": int(len(rows))}, f, indent=2)
        print(f"\nPASS leakage self-test ({len(rows)} rows, exact counts, "
              f"no feature corr>0.999)")
        return 0
    (OUT / "audit" / "leakage_ok.json").unlink(missing_ok=True)
    print(f"\nFAIL leakage self-test: {total_fail} problems")
    return 1


if __name__ == "__main__":
    sys.exit(main())
