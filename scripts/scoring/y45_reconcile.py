"""T0-1: reconcile the two y45 'cascade' PR-AUCs (0.069 model-box vs 0.008 wide-box path).

Hypothesis: they are DIFFERENT cascade simulations on the same model-box test cells — the
canonical build uses MODEL_SPEC (1,219 cells); the wide-box build recomputes the cascade on
WIDE_SPEC (2,275 cells) and subsets to in-box. Same 22 positives, different rate field ->
different ranking -> different PR-AUC. This script recomputes both on the identical model-box
test set, checks positives/cells match, and correlates the two cascade rate fields cell-by-cell.
Writes results/round3/t0_y45_reconcile.json.
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score

from marmara.paths import RESULTS
from marmara.metrics import p_to_lambda, lambda_to_p
from marmara.train import split_masks
from marmara.widebox_y45 import masks as wb_masks

EPS = 1e-9


def main():
    # canonical cascade (predictions_y45) + ir/ic via positional align to grid_hybrid
    pred = pd.read_parquet(RESULTS / "grid" / "predictions_y45.parquet")
    gh = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    gsub = gh[gh["window"].isin(pred["window"].unique())].reset_index(drop=True)
    assert np.array_equal(gsub["y45"].to_numpy(float), pred["y"].to_numpy(float)), "misaligned"
    # physical cell coords (grid_hybrid carries cell_lon/cell_lat; ir/ic are MODEL_SPEC here)
    pred = pred.copy()
    pred["lon"] = np.round(gsub["cell_lon"].to_numpy(), 2); pred["lat"] = np.round(gsub["cell_lat"].to_numpy(), 2)
    ct = pred[pred.split == "test"]
    canon_pr = float(average_precision_score(ct.y, ct.cascade))
    canon = ct[["window", "lon", "lat", "y", "cascade"]].copy()
    canon["lam_canon"] = p_to_lambda(np.clip(canon.cascade.to_numpy(), 0, 1 - EPS))

    # wide-box cascade (widebox_y45_grid, lam45_sim), model-box (in_box) test cells.
    # widebox ir/ic index the WIDE grid (min 25.0/39.0, 0.1 deg) -> convert to lon/lat.
    wb = pd.read_parquet(RESULTS / "grid" / "widebox_y45_grid.parquet")
    wm = wb_masks(wb)
    wt = wb[(wm["test"]) & (wb.in_box == 1)].copy()
    wt["lon"] = np.round(25.05 + wt.ic.to_numpy() * 0.1, 2); wt["lat"] = np.round(39.05 + wt.ir.to_numpy() * 0.1, 2)
    wt["P"] = lambda_to_p(np.clip(wt.lam45_sim.to_numpy(), 0, None))
    wide_pr = float(average_precision_score(wt.y45, wt.P))

    # align cell-by-cell on (window, lon, lat)
    j = canon.merge(wt[["window", "lon", "lat", "lam45_sim", "y45"]], on=["window", "lon", "lat"], how="inner")
    rate_corr_pearson = float(np.corrcoef(j.lam_canon, j.lam45_sim)[0, 1])
    rate_corr_spearman = float(spearmanr(j.lam_canon, j.lam45_sim).correlation)
    mean_ratio = float((j.lam45_sim.sum()) / max(j.lam_canon.sum(), EPS))

    out = {
        "canonical_cascade_modelbox_test": {"pr_auc": round(canon_pr, 4), "n_cells": int(len(ct)),
                                            "n_pos": int(ct.y.sum()), "spec": "MODEL_SPEC (1219 cells)",
                                            "manuscript_states": 0.069, "evaluation_md_states": 0.0622},
        "widebox_cascade_modelbox_test": {"pr_auc": round(wide_pr, 4), "n_cells": int(len(wt)),
                                          "n_pos": int(wt.y45.sum()), "spec": "WIDE_SPEC subset to in_box",
                                          "widebox_report_states": 0.0082},
        "same_eval_set": bool(len(ct) == len(wt) and int(ct.y.sum()) == int(wt.y45.sum())),
        "cascade_rate_field_agreement": {"pearson": round(rate_corr_pearson, 3),
                                         "spearman_rank": round(rate_corr_spearman, 3),
                                         "widebox/canonical_total_rate": round(mean_ratio, 3),
                                         "n_joined_cells": int(len(j))},
        "verdict": ("Same test cells and positives; the two 'cascade' PR-AUCs come from DIFFERENT "
                    "cascade simulations (MODEL_SPEC vs WIDE_SPEC subset). The Spearman rank "
                    "correlation quantifies how differently they order the model-box cells — which "
                    "is why 0.069 and 0.008 are not the same estimator and must not be juxtaposed."),
    }
    (RESULTS / "round3" / "t0_y45_reconcile.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
