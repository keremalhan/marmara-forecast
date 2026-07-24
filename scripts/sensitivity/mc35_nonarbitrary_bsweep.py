"""Run 12, item A — re-run at NON-ARBITRARY b, because the Mc=3.5 b_op sweep is degenerate.

THE PROBLEM. At base Mc = 3.5 the count-calibration sweep cannot select: every candidate b from
0.90 to 1.542 returns slope 0.898-0.902, a spread of 0.004. The reason is structural. The
calibration target is the per-window M>=3.5 count, which at base Mc = 3.5 is the TOTAL simulated
count; and with preserve_branching=True the total is held at n = 0.95 regardless of b. So b has no
purchase on the calibrated quantity. `min(candidates, key=|slope-1|)` then picks noise -- it
returned b = 1.05 on a 0.004-wide spread. Any verdict from that arm could be a calibration mismatch
of the arm rather than a completeness result.

THE FIX. Re-run the arm at two b values that are chosen by principle rather than by a degenerate
argmin, and report all three:
    b = 1.05    -- what the degenerate sweep returned (retained for the record)
    b = 1.15    -- the shipped operational b_op, TRANSFERRED from the Mc = 3.0 pipeline
    b = 1.4698  -- the Mc = 3.5 fit's own b-positive (the arm's internal magnitude law)
If the verdict is the same at all three, it is not a b artifact.

The 19 features do not depend on b -- only the cascade column does -- so the Mc = 3.5 grid is
reused and only lam35_sim is recomputed per b. The ETAS fit (etas_params_mc35.pkl) is unchanged.

Reported unconditionally. No gate.

Writes results/round4/r12_item_A_bsweep.json.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/sensitivity/mc35_nonarbitrary_bsweep.py
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from marmara import grid as G
from marmara.bootstrap import verdict_for
from marmara.cascade import cascade_forecast
from marmara.grid import FEATURES
from marmara.metrics import lambda_to_p, p_to_lambda
from marmara.paths import RESULTS
from marmara.train import WEIGHTS, _monotonic, select_w_1se, split_masks

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "r12A", str(RESULTS.parent / "scripts" / "round4" / "r12_item_A_mc35.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
paired = _m.paired

EPS = 1e-9
R4 = RESULTS / "round4"
K_BACKTEST = 500
B_ARMS = [("degenerate_sweep_pick", 1.05), ("transferred_shipped_b_op", 1.15),
          ("mc35_own_b_positive", 1.4698)]
SHIPPED = {"hybrid_vs_cascade": "inseparable", "hybrid_vs_firstgen_etas": "inseparable",
           "cascade_vs_firstgen_etas": "inseparable"}


def main():
    t_all = time.time()
    params = pickle.load(open(R4 / "etas_params_mc35.pkl", "rb"))
    grid = pd.read_parquet(R4 / "grid_mc35.parquet")
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    hist = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]
    spec = G.MODEL_SPEC
    m = split_masks(grid)
    win_t0 = grid.groupby("window")["t0"].first()
    wins = [int(x) for x in np.sort(grid["window"].unique())]
    y = grid["y35"].to_numpy().astype(float)
    n = grid["count35"].to_numpy().astype(float)
    npos_test = float(y[m["test"]].sum())

    out = {"governed_by": {"amendment": "docs/preregistration/v2_analysis_amendment_5.md",
                           "sha256": "c97db8f54374ac4ff1b5fbfafc1a1e76c63d68077144b338319603170ce846c2",
                           "item": "A (b re-run)"},
           "why": ("the Mc=3.5 b_op sweep is degenerate (all slopes 0.898-0.902): at base Mc=3.5 "
                   "the calibration target IS the total simulated count, which preserve_branching "
                   "holds at n=0.95 independently of b, so b has no purchase and the argmin picks "
                   "noise. The arm is therefore re-run at explicitly motivated b values."),
           "arms": {}}

    for label, b in B_ARMS:
        # recompute ONLY the cascade column at this b (features are b-independent)
        lam = np.zeros(len(grid))
        for k, w in enumerate(wins):
            t0_dt = pd.Timestamp(win_t0.loc[w]); t0d = float(G._to_days(t0_dt))
            c = cascade_forecast(params, hist[cat["datetime_utc"] < t0_dt], t0d, G.HORIZON_D,
                                 spec.lon_c, spec.lat_c, K=K_BACKTEST, seed=1000 + w, b=b,
                                 preserve_branching=True)
            lam[grid["window"].to_numpy() == w] = c["lam35"].ravel()
        lam_sim = np.clip(lam, EPS, None)

        X = grid[FEATURES].copy()
        X["ln_lam_sim"] = np.log(lam + EPS)
        reg = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=400,
                                            max_depth=6,
                                            monotonic_cst=_monotonic(FEATURES + ["ln_lam_sim"]),
                                            random_state=42)
        reg.fit(X[m["train"]], n[m["train"]])
        lam_ml = np.clip(reg.predict(X), EPS, None)
        w_sel, wdiag = select_w_1se(lam_sim, lam_ml, y, m["val"], grid["window"].to_numpy(), WEIGHTS)
        P = {"hybrid": lambda_to_p(lam_sim ** (1 - w_sel) * lam_ml ** w_sel),
             "cascade": lambda_to_p(lam_sim),
             "firstgen_etas": lambda_to_p(np.clip(grid["etas_rate"].to_numpy(), EPS, None))}

        te = m["test"]
        yt = y[te]; wt = grid["window"].to_numpy()[te]
        rec = {"b": b, "selected_w": float(w_sel), "w_argmax": float(wdiag["w_argmax"]),
               "n_trees_fit": int(reg.n_iter_),
               "sum_lam_test": round(float(lam_sim[te].sum()), 2),
               "h_test": {k: round(float(np.clip(p_to_lambda(np.clip(v[te], 0, 1)), EPS, None).sum()
                                         / npos_test), 4) for k, v in P.items()},
               "table1_rows": {k: {"pr_auc": round(float(average_precision_score(yt, v[te])), 6),
                                   "roc_auc": round(float(roc_auc_score(yt, v[te])), 6),
                                   "brier": round(float(brier_score_loss(yt, v[te])), 6)}
                               for k, v in P.items()},
               "verdicts": {}}
        for a, bb in (("hybrid", "cascade"), ("hybrid", "firstgen_etas"),
                      ("cascade", "firstgen_etas")):
            st = paired(P[a][te], P[bb][te], yt, wt)
            v = verdict_for(st["d_ig"]["ci95"], st["d_pr_auc"]["ci95"])
            st["verdict"] = {"A_beats_B": f"{a} beats {bb}", "B_beats_A": f"{bb} beats {a}",
                             "inseparable": "inseparable"}[v]
            rec["verdicts"][f"{a}_vs_{bb}"] = st
        rec["diff_vs_shipped"] = [f"{k}: {SHIPPED[k]} -> {v['verdict']}"
                                  for k, v in rec["verdicts"].items() if v["verdict"] != SHIPPED[k]]
        out["arms"][label] = rec
        print(f"\n[{label}] b={b}  w={w_sel} (argmax {wdiag['w_argmax']})  trees={reg.n_iter_}  "
              f"sum_lam_test={rec['sum_lam_test']}  ({time.time()-t_all:.0f}s)", flush=True)
        print(f"   h: {rec['h_test']}")
        for k, st in rec["verdicts"].items():
            print(f"   {k:26s} dIG {st['d_ig']['point']:+.4f} {st['d_ig']['ci95']}  "
                  f"dPR {st['d_pr_auc']['point']:+.4f} {st['d_pr_auc']['ci95']}  -> {st['verdict']}")

    flips = {k: v["diff_vs_shipped"] for k, v in out["arms"].items()}
    out["flips_per_arm"] = flips
    same = len({tuple(sorted(v["verdicts"][k]["verdict"] for k in SHIPPED))
                for v in out["arms"].values()}) == 1
    out["verdicts_identical_across_b"] = bool(same)
    out["conclusion"] = ("the Mc=3.5 verdicts are the same at all three b, so they are NOT an "
                         "artifact of the degenerate sweep" if same else
                         "the Mc=3.5 verdicts DEPEND on b, which the sweep cannot select -> the "
                         "arm cannot be read as a completeness result")
    out["runtime_s"] = round(time.time() - t_all, 1)
    json.dump(out, open(R4 / "r12_item_A_bsweep.json", "w"), indent=2)
    print(f"\nflips per arm: {json.dumps(flips, indent=1)}")
    print(f"verdicts identical across b: {same}")
    print(f"CONCLUSION: {out['conclusion']}")


if __name__ == "__main__":
    main()
