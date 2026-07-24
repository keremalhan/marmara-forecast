"""T0-5: extend the negative-control battery from 6 to 28 pairs (8 cascade seeds).

cascade@seed_i vs cascade@seed_j has TRUE difference zero; any 'separation' under the conjunctive
rule is a false positive. Isolated (writes results/round3/negcontrols.json only).
"""
import json
import pickle
import time

import numpy as np
import pandas as pd

from marmara.paths import RESULTS
from marmara import grid as G
from marmara import bootstrap as BS
from marmara.train import split_masks
from marmara.cascade import cascade_forecast
from marmara.metrics import information_gain, lambda_to_p

K = 500
B_OP = 1.15
SEEDS = [1000, 7000, 13000, 19000, 25000, 31000, 37000, 43000]   # 8 -> 28 pairs


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
                                    spec.lon_c, spec.lat_c, K=K, seed=sd + w, b=B_OP, preserve_branching=True)
            rows = (tw["window"] == w).to_numpy()
            lam[rows] = casc["lam30"][tw.loc[rows, "ir"].to_numpy(), tw.loc[rows, "ic"].to_numpy()]
        df[f"cascade_s{sd}"] = lambda_to_p(lam)
        print(f"  seed {sd}: built ({time.time()-t0:.0f}s)", flush=True)

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
            v = BS.verdict_for(st["d_ig"]["ci95"], st["d_pr_auc"]["ci95"])
            pairs.append({"pair": f"s{SEEDS[i]}_vs_s{SEEDS[j]}", "verdict": v})
            if v != "inseparable":
                false_sep += 1
    out = {"n_seeds": len(SEEDS), "n_pairs": len(pairs), "false_separations": false_sep,
           "empirical_false_positive_rate": round(false_sep / len(pairs), 4),
           "runtime_s": round(time.time() - t0, 1), "pairs": pairs}
    (RESULTS / "round3" / "negcontrols.json").write_text(json.dumps(out, indent=2))
    print(f"\nFALSE SEPARATIONS: {false_sep}/{len(pairs)} (rate {out['empirical_false_positive_rate']})")


if __name__ == "__main__":
    main()
