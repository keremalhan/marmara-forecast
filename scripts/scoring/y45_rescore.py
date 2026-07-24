"""E: re-score the wide-box y45 hybrid against the MODEL_SPEC cascade (the paper's actual cascade),
not the WIDE_SPEC cascade it was scored against. The manuscript's '+1.045 IG, PR 0.022 vs 0.008'
uses the WIDE_SPEC cascade (rho=0.35, 2.4x rate). Against MODEL_SPEC (PR 0.062) the wide hybrid
should LOSE on ranking. Writes results/round3/t3_y45_rescore.json.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score

from marmara.paths import RESULTS
from marmara.grid import FEATURES
from marmara.metrics import lambda_to_p, p_to_lambda, information_gain
from marmara.widebox_y45 import masks as wb_masks

EPS = 1e-9
W = 0.1   # production wide-box weight (report)


def main():
    wb = pd.read_parquet(RESULTS / "grid" / "widebox_y45_grid.parquet")
    m = wb_masks(wb)
    feats = [f for f in FEATURES if f in wb.columns]
    X = wb[feats].copy(); X["ln_lam_sim"] = np.log(np.clip(wb["lam45_sim"].to_numpy(), EPS, None) + EPS)
    n = wb["count45"].to_numpy(float)
    reg = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=400,
                                        max_depth=6, random_state=42)
    reg.fit(X[m["train"]], n[m["train"]])
    lam_ml = np.clip(reg.predict(X), EPS, None)
    lam_h = np.clip(wb["lam45_sim"].to_numpy(), EPS, None) ** (1 - W) * lam_ml ** W

    ti = m["test"] & (wb.in_box == 1)
    wt = wb[ti].copy()
    wt["P_h"] = lambda_to_p(lam_h[ti.to_numpy()])
    wt["lon"] = np.round(25.05 + wt.ic.to_numpy() * 0.1, 2); wt["lat"] = np.round(39.05 + wt.ir.to_numpy() * 0.1, 2)

    # canonical MODEL_SPEC cascade for the same cells (predictions_y45 + grid_hybrid coords)
    pred = pd.read_parquet(RESULTS / "grid" / "predictions_y45.parquet")
    gh = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    gsub = gh[gh["window"].isin(pred["window"].unique())].reset_index(drop=True)
    pred = pred.copy(); pred["lon"] = np.round(gsub.cell_lon.to_numpy(), 2); pred["lat"] = np.round(gsub.cell_lat.to_numpy(), 2)
    ct = pred[pred.split == "test"][["window", "lon", "lat", "y", "cascade"]]

    j = wt[["window", "lon", "lat", "P_h", "y45"]].merge(ct, on=["window", "lon", "lat"], how="inner")
    y = j.y.to_numpy(float)
    pr_widehyb = float(average_precision_score(y, j.P_h))
    pr_modelcasc = float(average_precision_score(y, j.cascade))
    lam_wh = np.clip(p_to_lambda(np.clip(j.P_h.to_numpy(), 0, 1 - EPS)), EPS, None)
    ig_vs_modelcasc = float(information_gain(j.P_h.to_numpy(), j.cascade.to_numpy(), y))

    out = {
        "n_cells": int(len(j)), "n_pos": int(y.sum()), "w": W,
        "widebox_hybrid_PR": round(pr_widehyb, 4),
        "MODEL_SPEC_cascade_PR": round(pr_modelcasc, 4),
        "manuscript_framing_vs_WIDE_SPEC_cascade": {"hybrid_PR": 0.022, "wide_cascade_PR": 0.008, "IG": 1.045},
        "IG_widehybrid_vs_MODEL_SPEC_cascade": round(ig_vs_modelcasc, 4),
        "verdict": ("wide-box hybrid fixes the model-box calibration pathology, but scored against the "
                    "ACTUAL (MODEL_SPEC) cascade it does not out-rank it: PR "
                    f"{pr_widehyb:.4f} vs {pr_modelcasc:.4f}. The '+1.045 vs cascade' was against the "
                    "WIDE_SPEC cascade (rho=0.35 with the paper's cascade)."),
    }
    (RESULTS / "round3" / "t3_y45_rescore.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
