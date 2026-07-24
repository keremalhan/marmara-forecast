"""Run 12, item F — control arm, to disentangle a confound in F before interpreting it.

THE CONFOUND. The F arm differs from the shipped hybrid in TWO ways at once:
  (1) the tree count is chosen on a temporal tail rather than a random 10% (y30: 200 vs 147;
      y35: 100 vs 10), and
  (2) the effective training set differs. sklearn's `early_stopping='auto'` holds out
      `validation_fraction=0.1` and fits the FINAL model on the remaining 90% -- it does not refit
      on 100% afterwards. F, with early_stopping=False, fits on 100% of the training window.
So "F flips a verdict" could mean "temporal stopping matters" OR merely "any change to the ML fit
moves this pair". The two have very different consequences for the manuscript.

THE CONTROL. Fit at the SHIPPED tree count with early_stopping=False on 100% of training, and
re-select w by the 1-SE rule. This holds the tree count at its shipped value and changes only the
training-set size / stopping mechanism. Comparing three arms then isolates the cause:
    shipped        : random stop, 147 (y30) / 10 (y35) trees, 90% of train
    control        : no stop,     147 (y30) / 10 (y35) trees, 100% of train
    F (temporal)   : no stop,     200 (y30) / 100 (y35) trees, 100% of train
If control ~= shipped, the flip is driven by the TREE COUNT the temporal tail selects.
If control ~= F,       the flip is driven by the internal hold-out's removal, not by temporality.

Reported unconditionally, like every item in Amendment 5.

Writes results/round4/r12_item_F_control.json.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/sensitivity/early_stopping_control.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from marmara.bootstrap import verdict_for
from marmara.grid import FEATURES
from marmara.metrics import lambda_to_p
from marmara.paths import RESULTS
from marmara.train import WEIGHTS, _monotonic, select_w_1se, split_masks

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "r12F", str(RESULTS.parent / "scripts" / "round4" / "r12_item_F.py"))
_m = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_m)
paired = _m.paired

EPS = 1e-9
R4 = RESULTS / "round4"
SHIPPED_TREES = {"y30": 147, "y35": 10}
TARGETS = {"y30": ("count30", "lam30_sim", "y30"), "y35": ("count35", "lam35_sim", "y35")}


def main():
    t_all = time.time()
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    m = split_masks(grid)
    out = {"governed_by": {"amendment": "docs/preregistration/v2_analysis_amendment_5.md",
                           "sha256": "c97db8f54374ac4ff1b5fbfafc1a1e76c63d68077144b338319603170ce846c2",
                           "item": "F-control"},
           "design": ("shipped tree count, early_stopping=False, 100% of training -> isolates the "
                      "tree-count effect from the internal-holdout effect"),
           "targets": {}}

    for tgt, (count_col, lam_col, ycol) in TARGETS.items():
        X = grid[FEATURES].copy()
        X["ln_lam_sim"] = np.log(grid[lam_col].to_numpy() + EPS)
        n = grid[count_col].to_numpy().astype(float)
        y = grid[ycol].to_numpy().astype(float)
        lam_sim = np.clip(grid[lam_col].to_numpy(), EPS, None)
        mono = _monotonic(FEATURES + ["ln_lam_sim"])

        reg = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05,
                                            max_iter=SHIPPED_TREES[tgt], max_depth=6,
                                            monotonic_cst=mono, random_state=42,
                                            early_stopping=False)
        reg.fit(X[m["train"]], n[m["train"]])
        lam_ml = np.clip(reg.predict(X), EPS, None)
        w, wdiag = select_w_1se(lam_sim, lam_ml, y, m["val"], grid["window"].to_numpy(), WEIGHTS)
        P_hyb = lambda_to_p(lam_sim ** (1 - w) * lam_ml ** w)

        pred = pd.read_parquet(RESULTS / f"predictions_{tgt}.parquet")
        sel = np.where(m["val"] | m["test"])[0]
        te_mask = (pred["split"] == "test").to_numpy()
        idx_test = sel[te_mask]
        yt = y[idx_test]; wt = grid["window"].to_numpy()[idx_test]
        P_h = P_hyb[idx_test]
        rec = {"trees": SHIPPED_TREES[tgt], "early_stopping": False, "train_frac": 1.0,
               "selected_w": float(w), "shipped_w": {"y30": 0.8, "y35": 0.4}[tgt],
               "w_argmax": float(wdiag["w_argmax"]), "verdicts": {}}
        for other in ("cascade", "sv_etas", "firstgen_etas"):
            P_o = pred[other].to_numpy(float)[te_mask]
            st = paired(P_h, P_o, yt, wt)
            v = verdict_for(st["d_ig"]["ci95"], st["d_pr_auc"]["ci95"])
            st["verdict"] = {"A_beats_B": f"hybrid beats {other}",
                             "B_beats_A": f"{other} beats hybrid",
                             "inseparable": "inseparable"}[v]
            rec["verdicts"][f"hybrid_vs_{other}"] = st
        rec["max_abs_dev_vs_shipped_hybrid"] = float(
            np.abs(P_h - pred["hybrid"].to_numpy(float)[te_mask]).max())
        out["targets"][tgt] = rec
        print(f"[{tgt}] control: {SHIPPED_TREES[tgt]} trees, no stop, 100% train -> w={w} "
              f"(shipped {rec['shipped_w']})  ({time.time()-t_all:.0f}s)", flush=True)
        for k, st in rec["verdicts"].items():
            print(f"    {k:26s} dIG {st['d_ig']['point']:+.4f} {st['d_ig']['ci95']}  "
                  f"dPR {st['d_pr_auc']['point']:+.4f} {st['d_pr_auc']['ci95']}  -> {st['verdict']}")

    # three-arm comparison
    F = json.load(open(R4 / "r12_item_F.json"))
    ship = {"y30": {"hybrid_vs_cascade": "inseparable", "hybrid_vs_sv_etas": "inseparable",
                    "hybrid_vs_firstgen_etas": "inseparable"},
            "y35": {"hybrid_vs_cascade": "inseparable", "hybrid_vs_sv_etas": "inseparable",
                    "hybrid_vs_firstgen_etas": "inseparable"}}
    cmp = {}
    for tgt in TARGETS:
        cmp[tgt] = {}
        for k in ("hybrid_vs_cascade", "hybrid_vs_sv_etas", "hybrid_vs_firstgen_etas"):
            cmp[tgt][k] = {
                "shipped": ship[tgt][k],
                "control_shipped_trees_no_stop": out["targets"][tgt]["verdicts"][k]["verdict"],
                "F_temporal_stop": F["targets"][tgt]["verdicts"][k]["verdict"],
                "shipped_dPR": None,
                "control_dPR": out["targets"][tgt]["verdicts"][k]["d_pr_auc"]["point"],
                "F_dPR": F["targets"][tgt]["verdicts"][k]["d_pr_auc"]["point"],
            }
    out["three_arm_comparison"] = cmp
    ctrl_flips = [f"{t}/{k}" for t in cmp for k in cmp[t]
                  if cmp[t][k]["control_shipped_trees_no_stop"] != cmp[t][k]["shipped"]]
    out["control_flips_vs_shipped"] = ctrl_flips
    out["attribution"] = (
        "the flip is driven by the TREE COUNT selected on the temporal tail (control, at the "
        "shipped count, reproduces the shipped verdicts)" if not ctrl_flips else
        "the flip is NOT specific to temporal stopping: removing the internal hold-out alone "
        "(control, at the shipped tree count) already moves the verdict, so tree count and "
        "training-set size are entangled and F cannot isolate temporality")
    out["runtime_s"] = round(time.time() - t_all, 1)
    json.dump(out, open(R4 / "r12_item_F_control.json", "w"), indent=2)
    print(f"\ncontrol flips vs shipped: {ctrl_flips if ctrl_flips else 'NONE'}")
    print(f"ATTRIBUTION: {out['attribution']}")


if __name__ == "__main__":
    main()
