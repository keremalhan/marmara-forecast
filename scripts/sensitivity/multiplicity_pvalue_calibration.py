"""Run 25 — Item 7 of Amendment 8, PART B.1: the p-value CALIBRATION GATE (before any BH).

Governed by docs/preregistration/v2_analysis_amendment_8.md (SHA-256 c5684700..., hashed 2026-07-16T11:25:21Z).

The p-value construction (r24) is p = 2*min(win_rate, 1-win_rate) on the seed-42 block-bootstrap.
Before BH trusts it, test it on NEGATIVE CONTROLS with true diff = 0: same-model seed pairs
(cascade@seed_i vs cascade@seed_j), reusing scripts/verify/mult_calibration.py's exact construction
but capturing the win_rate the original discarded. 8 seeds -> C(8,2) = 28 null pairs.

Under a valid construction the 28 p-values are ~uniform. With n=28 we do NOT run a formal KS as
dispositive; we look for GROSS PATHOLOGY on each axis:
  * mass piled near 0  -> ANTICONSERVATIVE -> construction invalid -> STOP (BH must not run)
  * mass piled near 1  -> conservative     -> proceed; BH is then conservative too (only strengthens
                                              surviving findings)

Writes results/round4/r25_multiplicity_calibration.json.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/sensitivity/multiplicity_pvalue_calibration.py
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd

from marmara import bootstrap as BS
from marmara import grid as G
from marmara.cascade import cascade_forecast
from marmara.metrics import lambda_to_p
from marmara.paths import RESULTS
from marmara.train import split_masks

B_OP = 1.15
K = 500
SEEDS = [1000, 4000, 7000, 10000, 13000, 16000, 19000, 22000]   # 8 -> 28 null pairs


def p_from_winrate(wr):
    return round(2.0 * min(wr, 1.0 - wr), 5)


def gross_pathology(ps, axis):
    ps = np.array(ps)
    near0 = float((ps < 0.05).mean()); near1 = float((ps > 0.95).mean())
    # anticonservative if far more than 5% of true-nulls fall below 0.05
    antic = near0 > 0.15
    return {"axis": axis, "n": len(ps), "frac_p<0.05": round(near0, 3), "frac_p>0.95": round(near1, 3),
            "median_p": round(float(np.median(ps)), 3), "min_p": round(float(ps.min()), 4),
            "anticonservative": bool(antic),
            "verdict": ("ANTICONSERVATIVE -> STOP" if antic else
                        "conservative (BH conservative too)" if float(np.median(ps)) > 0.6 else
                        "no gross pathology -> proceed")}


def main():
    t0 = time.time()
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    m = split_masks(grid)
    tw = grid[grid["window"].isin(np.sort(grid.loc[m["test"], "window"].unique()))].copy()
    tw = tw.sort_values(["window", "ir", "ic"]).reset_index(drop=True)
    win_t0 = grid.groupby("window")["t0"].first()
    test_wins = [int(w) for w in np.sort(tw["window"].unique())]
    params = pickle.load(open(RESULTS / "etas" / "etas_params.pkl", "rb"))
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    hist = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]
    spec = G.MODEL_SPEC

    df = pd.DataFrame({"window": tw["window"].to_numpy(), "y": tw["y30"].to_numpy(float)})
    for sd in SEEDS:
        lam = np.zeros(len(tw))
        for w in test_wins:
            t0_dt = pd.Timestamp(win_t0.loc[w]); t0d = float(G._to_days(t0_dt))
            casc = cascade_forecast(params, hist[cat["datetime_utc"] < t0_dt], t0d, G.HORIZON_D,
                                    spec.lon_c, spec.lat_c, K=K, seed=sd + w, b=B_OP,
                                    preserve_branching=True)
            rows = (tw["window"] == w).to_numpy()
            lam[rows] = casc["lam30"][tw.loc[rows, "ir"].to_numpy(), tw.loc[rows, "ic"].to_numpy()]
        df[f"cascade_s{sd}"] = lambda_to_p(lam)
        print(f"  seed {sd}: predictor built ({time.time()-t0:.0f}s)", flush=True)

    models = [f"cascade_s{sd}" for sd in SEEDS]
    rng = np.random.default_rng(BS.SEED)
    full = BS.full_metrics(df, models)
    bs = BS.bootstrap_split(df, models, 2000, rng)

    p_ig, p_pr, rows = [], [], []
    for i in range(len(models)):
        for jx in range(i + 1, len(models)):
            a, b_ = models[i], models[jx]
            st = BS.pair_stats(bs, full, a, b_)
            pig = p_from_winrate(st["d_ig"]["win_rate"])
            ppr = p_from_winrate(st["d_pr_auc"]["win_rate"])
            p_ig.append(pig); p_pr.append(ppr)
            rows.append({"pair": f"{a}_vs_{b_}", "p_ig": pig, "p_pr": ppr})

    out = {"governed_by": {"amendment": "docs/preregistration/v2_analysis_amendment_8.md",
                           "sha256": "c5684700aa656949908640faa326c6b6f15b3a699052f627272bc26a1186e690"},
           "negative_controls": "cascade@seed_i vs cascade@seed_j (true diff=0)", "seeds": SEEDS,
           "n_null_pairs": len(rows), "pairs": rows,
           "calibration_IG": gross_pathology(p_ig, "count-scored ΔIG"),
           "calibration_PR": gross_pathology(p_pr, "ΔPR-AUC"),
           "runtime_s": round(time.time() - t0, 1)}
    ok = not (out["calibration_IG"]["anticonservative"] or out["calibration_PR"]["anticonservative"])
    out["GATE"] = "PASS -> BH may run" if ok else "FAIL -> BH must not run; fix the construction"
    json.dump(out, open(RESULTS / "round4" / "r25_multiplicity_calibration.json", "w"), indent=1, default=str)

    print(f"\n=== CALIBRATION GATE on {len(rows)} same-model null pairs ===")
    for k in ("calibration_IG", "calibration_PR"):
        c = out[k]
        print(f"  {c['axis']:16s}: frac p<0.05 {c['frac_p<0.05']} | median {c['median_p']} | "
              f"min {c['min_p']} -> {c['verdict']}")
    print(f"  GATE: {out['GATE']}  ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
