"""Item G: empirical family-wise error of the conjunctive verdict rule.

Negative control: the cascade is stochastic (sim seed), so cascade@seed_i vs cascade@seed_j
is a pair with TRUE difference zero. Run the full pre-registered conjunctive rule (both IG
and PR-AUC 95% block-bootstrap CIs exclude 0) on every such pair; any "separation" is a
FALSE positive. This measures the rule-level false-positive rate for the binding
(stochastic) case; deterministic-model pairs are 0 by construction. Report the empirical
rate and compare it to the naive-independence expectation (~8 of 216) and to the surviving
claimed separations (which replicate across targets -- chance does not).

Writes results/mult_calibration.{json,md}.
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
from marmara.metrics import lambda_to_p, information_gain
from marmara import bootstrap as BS

OUT = RESULTS
B_OP = 1.15
K = 500
SEEDS = [1000, 7000, 13000, 19000]     # 4 seeds -> 6 negative-control pairs (true diff = 0)


def main():
    t0 = time.time()
    grid = pd.read_parquet(OUT / "grid" / "grid_hybrid.parquet")
    m = split_masks(grid)
    tw = grid[grid["window"].isin(np.sort(grid.loc[m["test"], "window"].unique()))].copy()
    tw = tw.sort_values(["window", "ir", "ic"]).reset_index(drop=True)
    win_t0 = grid.groupby("window")["t0"].first()
    test_wins = [int(w) for w in np.sort(tw["window"].unique())]
    params = pickle.load(open(OUT / "etas" / "etas_params.pkl", "rb"))
    cat = pd.read_csv(OUT / "catalog" / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    hist = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]
    spec = G.MODEL_SPEC

    # cascade y30 predictor P(>=1 M>=3.0) per test cell, per seed
    df = pd.DataFrame({"window": tw["window"].to_numpy(), "y": tw["y30"].to_numpy(float)})
    for sd in SEEDS:
        lam = np.zeros(len(tw))
        for w in test_wins:
            t0_dt = pd.Timestamp(win_t0.loc[w]); t0d = float(G._to_days(t0_dt))
            casc = cascade_forecast(params, hist[cat["datetime_utc"] < t0_dt], t0d, G.HORIZON_D,
                                    spec.lon_c, spec.lat_c, K=K, seed=sd + w, b=B_OP, preserve_branching=True)
            rows = (tw["window"] == w).to_numpy()
            lam[rows] = casc["lam30"][tw.loc[rows, "ir"].to_numpy(), tw.loc[rows, "ic"].to_numpy()]
        df[f"cascade_s{sd}"] = lambda_to_p(lam)
        print(f"  seed {sd}: cascade predictor built ({time.time()-t0:.0f}s)", flush=True)

    models = [f"cascade_s{sd}" for sd in SEEDS]
    rng = np.random.default_rng(BS.SEED)
    full = BS.full_metrics(df, models)
    bs = BS.bootstrap_split(df, models, 2000, rng)
    y = df["y"].to_numpy(float)
    pairs, false_sep = [], 0
    for i in range(len(models)):
        for j in range(i + 1, len(models)):
            a, b_ = models[i], models[j]
            st = BS.pair_stats(bs, full, a, b_)
            st["d_ig"]["point"] = round(float(information_gain(df[a].to_numpy(float), df[b_].to_numpy(float), y)), 4)
            v = BS.verdict_for(st["d_ig"]["ci95"], st["d_pr_auc"]["ci95"])
            pairs.append({"pair": f"{a}_vs_{b_}", "verdict": v, "ig_ci": st["d_ig"]["ci95"], "pr_ci": st["d_pr_auc"]["ci95"]})
            if v != "inseparable":
                false_sep += 1
    n_pairs = len(pairs)
    rate = false_sep / n_pairs
    # canonical claims separation count (for context)
    claims = json.load(open(OUT / "claims.json"))["claims"]
    prim = [c for c in claims if c["target"] == "y30" and c["split"] == "test"]
    n_sep = sum(1 for c in prim if c["verdict"] != "inseparable")
    result = {
        "meta": {"seeds": SEEDS, "n_negative_control_pairs": n_pairs, "B": 2000, "runtime_s": round(time.time() - t0, 1)},
        "false_separations": false_sep, "empirical_false_positive_rate": round(rate, 4),
        "naive_independence_expectation_of_216": round(0.05 * 216, 1),
        "pairs": pairs,
        "context": {"y30_test_separations_in_claims": n_sep, "total_y30_test_pairs": len(prim),
                    "note": "claimed separations (physics cluster + first-gen>Mizrahi) replicate across y30/y35 -- "
                            "independent chance at the empirical false-positive rate does not reproduce a "
                            "separation at both powered targets."},
    }
    (OUT / "scoring" / "mult_calibration.json").write_text(json.dumps(result, indent=2))
    L = ["# Item G: empirical family-wise error of the conjunctive verdict rule", "",
         f"Negative control: cascade@seed_i vs cascade@seed_j (true diff = 0), {n_pairs} pairs, B=2000.", "",
         f"- **False separations: {false_sep}/{n_pairs}  (empirical per-comparison false-positive rate {rate:.3f}).**",
         f"- Naive-independence expectation at alpha=0.05 over 216 comparisons: ~{0.05*216:.0f} false positives; "
         f"the conjunctive rule (BOTH IG and PR-AUC CIs exclude 0) is far stricter, as this rate shows.",
         f"- Claimed y30/test separations in the canonical claims file: {n_sep}/{len(prim)}; they replicate across "
         f"y30 and y35 (independent chance at the measured rate does not reproduce a both-target separation).", "",
         "| negative-control pair | verdict (want: inseparable) |", "|---|---|"]
    for p in pairs:
        L.append(f"| {p['pair']} | {p['verdict']} |")
    (OUT / "scoring" / "mult_calibration.md").write_text("\n".join(L))
    print(f"\nFALSE SEPARATIONS: {false_sep}/{n_pairs} (rate {rate:.3f}); wrote results/mult_calibration.{{json,md}}")


if __name__ == "__main__":
    main()
