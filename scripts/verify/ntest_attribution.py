"""N-test attribution (reviewer note 3): cross the two candidate causes of the
M>=3.0 over-prediction (supercritical forecast inflation vs undercounted
observation) and report the full 2x2, plus the completeness-unambiguous M>=3.5
anchor.

Forecast side
  supercritical : the shipped cascade, n(b_op)=1.21. Read from grid_hybrid.parquet
                  lam{30,35}_sim summed over the CSEP test windows -- this is exactly
                  what csep_eval turned into N_forecast_mean=1498.1.
  rescaled      : the cascade re-run at the SAME b_op=1.15 with preserve_branching=True,
                  so the mmax-truncated branching ratio is held at the fitted 0.95.
Observation side
  raw           : on-grid M>=3.0 count in the test period (= 1383).
  corrected     : GR-extrapolated M>=3.0, anchoring on the completeness-unambiguous
                  [3.45,4.5) band and extrapolating the [3.0,3.45) deficit; a
                  Md/ML type-split FMD diagnostic is printed alongside (first pass;
                  the fully type-resolved band is the rigorous follow-up).

N-test = is the observed count inside the 95% CI of Poisson(forecast_mean)?
(the sum of the per-cell Poisson draws in csep_eval is Poisson(sum lam), so this
analytic form matches the in-house MC test exactly.)

Writes results/ntest_attribution.{json,md}. Reads only; mutates nothing else.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/verify/ntest_attribution.py
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd
from scipy.stats import poisson

from marmara.paths import RESULTS
from marmara import grid as G
from marmara.train import split_masks
from marmara.cascade import cascade_forecast
from marmara.etas_model import _aki_b

OUT = RESULTS
B_OP = 1.15
K = 500                       # matches grid_hybrid.K_BACKTEST
ALPHA = 0.05                  # N-test two-sided


def n_test(forecast_mean: float, observed: int) -> dict:
    """Two-sided Poisson N-test. delta1=P(X>=obs), delta2=P(X<=obs); pass if both>0.025."""
    d1 = float(poisson.sf(observed - 1, forecast_mean))   # P(X >= obs)
    d2 = float(poisson.cdf(observed, forecast_mean))      # P(X <= obs)
    return {"forecast_mean": round(float(forecast_mean), 1), "observed": int(observed),
            "delta1": round(d1, 4), "delta2": round(d2, 4),
            "pass": bool(min(d1, d2) > ALPHA / 2.0)}


def main():
    t_start = time.time()
    grid = pd.read_parquet(OUT / "grid" / "grid_hybrid.parquet")
    test_wins = np.sort(grid.loc[split_masks(grid)["test"], "window"].unique())
    tw = grid[grid["window"].isin(test_wins)]

    # ---- forecast side: supercritical baseline from the stored parquet ----------
    F_super30 = float(tw["lam30_sim"].sum())
    F_super35 = float(tw["lam35_sim"].sum())

    # ---- forecast side: rescaled cascade, re-run at the same b_op ----------------
    with open(OUT / "etas" / "etas_params.pkl", "rb") as f:
        params = pickle.load(f)
    cat = pd.read_csv(OUT / "catalog" / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    spec = G.MODEL_SPEC
    hist_all = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]

    win_t0 = grid.groupby("window")["t0"].first()
    F_resc30 = F_resc35 = 0.0
    n_real = []
    check = None
    for i, w in enumerate(test_wins):
        w = int(w)
        t0_dt = pd.Timestamp(win_t0.loc[w])
        t0d = float(G._to_days(t0_dt))
        h = hist_all[cat["datetime_utc"] < t0_dt]
        casc = cascade_forecast(params, h, t0d, G.HORIZON_D, spec.lon_c, spec.lat_c,
                                K=K, seed=1000 + w, b=B_OP, preserve_branching=True)
        F_resc30 += float(casc["lam30"].sum())
        F_resc35 += float(casc["lam35"].sum())
        n_real.append(casc["n_branching"])
        if check is None:   # reproduce the supercritical parquet for one window as a self-check
            c0 = cascade_forecast(params, h, t0d, G.HORIZON_D, spec.lon_c, spec.lat_c,
                                  K=K, seed=1000 + w, b=B_OP, preserve_branching=False)
            check = {"window": w,
                     "rerun_super_lam30": round(float(c0["lam30"].sum()), 3),
                     "stored_super_lam30": round(float(grid[grid["window"] == w]["lam30_sim"].sum()), 3),
                     "n_super": round(float(c0["n_branching"]), 4)}
        if (i + 1) % 5 == 0:
            print(f"  window {i+1}/{len(test_wins)} ({time.time()-t_start:.0f}s) "
                  f"running F_resc30={F_resc30:.0f}")

    # ---- observation side --------------------------------------------------------
    t0s = pd.to_datetime(win_t0.loc[[int(w) for w in test_wins]].values)
    lo, hi = t0s.min(), t0s.max() + pd.Timedelta(days=G.HORIZON_D)

    def on_grid(df):
        iir, iic = G.cell_index(df["longitude"].to_numpy(), df["latitude"].to_numpy())
        return df[iir >= 0]

    ev_all = on_grid(cat[(cat["datetime_utc"] >= lo) & (cat["datetime_utc"] < hi)])
    O_raw30 = int((ev_all["mag_w"] >= 3.0 - 1e-9).sum())
    O_35 = int((ev_all["mag_w"] >= 3.5 - 1e-9).sum())

    # completeness correction: anchor GR on the complete [3.45,4.5) band, extrapolate 3.0
    m = ev_all["mag_w"].to_numpy()
    N_complete = int(((m >= 3.45) & (m < 4.5)).sum())
    N_hi = int((m >= 3.45).sum())
    b_test = float(_aki_b(m[m >= 3.0], 3.0))
    b_lo, b_hi = b_test * 0.85, b_test * 1.15     # crude +-15% band on b
    O_corr30 = round(N_hi * 10.0 ** (b_test * 0.45))
    O_corr30_lo = round(N_hi * 10.0 ** (b_lo * 0.45))
    O_corr30_hi = round(N_hi * 10.0 ** (b_hi * 0.45))
    obs_3_345 = int(((m >= 3.0) & (m < 3.45)).sum())
    exp_3_345 = N_hi * (10.0 ** (b_test * 0.45) - 1.0)

    # Md/ML type-split diagnostic in the two bands
    mt = ev_all["mag_type"].astype(str).str.upper().str.strip()
    def band_types(lo_m, hi_m):
        sel = (ev_all["mag_w"] >= lo_m) & (ev_all["mag_w"] < hi_m)
        vc = mt[sel].value_counts()
        return {k: int(v) for k, v in vc.items()}
    types_3_345 = band_types(3.0, 3.45)
    types_345_45 = band_types(3.45, 4.5)

    # ---- 2x2 N-test + M>=3.5 anchor ---------------------------------------------
    table = {
        "M>=3.0": {
            "supercritical_vs_raw":       n_test(F_super30, O_raw30),
            "supercritical_vs_corrected": n_test(F_super30, O_corr30),
            "rescaled_vs_raw":            n_test(F_resc30, O_raw30),
            "rescaled_vs_corrected":      n_test(F_resc30, O_corr30),
        },
        "M>=3.5_anchor": {
            "supercritical_vs_raw": n_test(F_super35, O_35),
            "rescaled_vs_raw":      n_test(F_resc35, O_35),
        },
    }

    result = {
        "meta": {"test_period": [str(lo.date()), str(hi.date())],
                 "n_windows": int(len(test_wins)), "b_op": B_OP, "K": K,
                 "runtime_s": round(time.time() - t_start, 1)},
        "reproduction_check": check,
        "branching_ratio": {"fitted_b": round(float(params.b), 4),
                            "n_at_fitted_b": 0.95,
                            "n_at_b_op_supercritical": 1.212,
                            "n_at_b_op_rescaled_mean": round(float(np.mean(n_real)), 4)},
        "forecast_mean": {"supercritical_M3.0": round(F_super30, 1),
                          "rescaled_M3.0": round(F_resc30, 1),
                          "supercritical_M3.5": round(F_super35, 1),
                          "rescaled_M3.5": round(F_resc35, 1)},
        "observed": {"raw_M3.0": O_raw30, "M3.5": O_35,
                     "corrected_M3.0": O_corr30,
                     "corrected_M3.0_band": [O_corr30_lo, O_corr30_hi],
                     "b_test_aki": round(b_test, 3),
                     "N_complete_3.45_4.5": N_complete, "N_ge_3.45": N_hi,
                     "obs_[3.0,3.45)": obs_3_345, "gr_expected_[3.0,3.45)": round(exp_3_345, 1),
                     "deficit_[3.0,3.45)": round(exp_3_345 - obs_3_345, 1),
                     "types_[3.0,3.45)": types_3_345, "types_[3.45,4.5)": types_345_45},
        "n_test_2x2": table,
    }
    (OUT / "scoring" / "ntest_attribution.json").write_text(json.dumps(result, indent=2))
    _write_md(result, OUT / "scoring" / "ntest_attribution.md")
    print(json.dumps(result, indent=2))


def _write_md(r, path):
    fm, tb, ob = r["forecast_mean"], r["n_test_2x2"], r["observed"]
    def verdict(d): return ("PASS" if d["pass"] else "**FAIL**") + f" (d1 {d['delta1']}, d2 {d['delta2']})"
    L = ["# N-test attribution: forecast inflation vs observation undercount", "",
         f"Test period {r['meta']['test_period'][0]}..{r['meta']['test_period'][1]}, "
         f"{r['meta']['n_windows']} windows, b_op={r['meta']['b_op']}.", "",
         f"Reproduction check (window {r['reproduction_check']['window']}): re-run supercritical "
         f"lam30={r['reproduction_check']['rerun_super_lam30']} vs stored "
         f"{r['reproduction_check']['stored_super_lam30']} (n={r['reproduction_check']['n_super']}).", "",
         "## Branching ratio",
         f"- fitted b={r['branching_ratio']['fitted_b']}, n=0.95; at b_op unrescaled n=1.212 "
         f"(supercritical); rescaled mean n={r['branching_ratio']['n_at_b_op_rescaled_mean']}.", "",
         "## M>=3.0 attribution (2x2)", "",
         f"Forecast mean: supercritical **{fm['supercritical_M3.0']}**, rescaled **{fm['rescaled_M3.0']}**. "
         f"Observed: raw **{ob['raw_M3.0']}**, completeness-corrected **{ob['corrected_M3.0']}** "
         f"(band {ob['corrected_M3.0_band']}, Aki b={ob['b_test_aki']}).", "",
         "| forecast \\ observed | raw " + str(ob["raw_M3.0"]) + " | corrected " + str(ob["corrected_M3.0"]) + " |",
         "|---|---|---|",
         f"| supercritical {fm['supercritical_M3.0']} | {verdict(tb['M>=3.0']['supercritical_vs_raw'])} | {verdict(tb['M>=3.0']['supercritical_vs_corrected'])} |",
         f"| rescaled {fm['rescaled_M3.0']} | {verdict(tb['M>=3.0']['rescaled_vs_raw'])} | {verdict(tb['M>=3.0']['rescaled_vs_corrected'])} |",
         "",
         "## M>=3.5 anchor (completeness unambiguous, Mc~2.72)", "",
         f"| forecast | observed {ob['M3.5']} |", "|---|---|",
         f"| supercritical {fm['supercritical_M3.5']} | {verdict(tb['M>=3.5_anchor']['supercritical_vs_raw'])} |",
         f"| rescaled {fm['rescaled_M3.5']} | {verdict(tb['M>=3.5_anchor']['rescaled_vs_raw'])} |",
         "",
         "## Completeness diagnostic ([3.0,3.45) band)",
         f"- observed {ob['obs_[3.0,3.45)']} vs GR-expected {ob['gr_expected_[3.0,3.45)']} "
         f"(deficit {ob['deficit_[3.0,3.45)']}).",
         f"- types [3.0,3.45): {ob['types_[3.0,3.45)']}",
         f"- types [3.45,4.5): {ob['types_[3.45,4.5)']}"]
    path.write_text("\n".join(L))


if __name__ == "__main__":
    main()
