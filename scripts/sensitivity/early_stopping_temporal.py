"""Run 12, item F (Amendment 5, SHA-256 c97db8f5...): grouped-temporal early-stopping sensitivity.

DEFECT. `early_stopping='auto'` picks the tree count on scikit-learn's internal RANDOM 10% of the
training rows. That is within-training and cannot leak the test period, but a random split of
temporally autocorrelated rows puts near-duplicate neighbours on both sides, so the stop is decided
under optimistic conditions. The direction of that bias is NOT established by anything in this
repository -- the earlier claim that it "can only be toward more trees" was an intuition, not a
theorem, and is withdrawn. We measure it.

PROCEDURE (fixed before running). Disable the internal random stop. Select the tree count on a
temporal within-train tail: fit on windows with t0 <= 2020-12-31, score candidates
n_trees in {10, 25, 50, 100, 147, 200, 400} on the 2021 windows by the booster's own objective
(Poisson log-likelihood on counts). Refit on the FULL training window at the selected count.
Re-select w by the pre-registered 1-SE rule on the true validation split. Evaluate the test verdicts.

Only the ML stage changes; cascade and first-generation ETAS are physics and are taken from the
frozen predictions unchanged.

REPORTING. Unconditional: selected trees, selected w, and hybrid-vs-cascade / hybrid-vs-first-gen on
both axes, in whichever direction they move.

Writes results/round4/r12_item_F.json.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/sensitivity/early_stopping_temporal.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score

from marmara.bootstrap import MEAN_BLOCK, SEED, stationary_window_indices, verdict_for
from marmara.grid import FEATURES
from marmara.metrics import information_gain, lambda_to_p, p_to_lambda
from marmara.paths import RESULTS
from marmara.train import WEIGHTS, _monotonic, select_w_1se, split_masks

EPS = 1e-9
B_BOOT = 2000
R4 = RESULTS / "round4"
R4.mkdir(exist_ok=True)

INNER_END = pd.Timestamp("2021-01-01")     # inner-train: t0 < this
TAIL_END = pd.Timestamp("2022-01-01")      # temporal stopping tail: [INNER_END, TAIL_END)
CANDIDATES = [10, 25, 50, 100, 147, 200, 400]
TARGETS = {"y30": ("count30", "lam30_sim", "y30"), "y35": ("count35", "lam35_sim", "y35")}


def poisson_ll(n, lam):
    lam = np.clip(lam, EPS, None)
    return float(np.sum(n * np.log(lam) - lam))


def paired(P_a, P_b, y, win):
    wins = np.sort(np.unique(win))
    y_by, a_by, b_by, lla, llb, pos, nrow = [], [], [], [], [], [], []
    for w in wins:
        k = win == w
        yv = y[k]
        y_by.append(yv); a_by.append(P_a[k]); b_by.append(P_b[k])
        pos.append(float(yv.sum())); nrow.append(float(k.sum()))
        for P, acc in ((P_a[k], lla), (P_b[k], llb)):
            lam = np.clip(p_to_lambda(np.clip(P, 0.0, 1.0)), EPS, None)
            acc.append(float(np.sum(yv * np.log(lam) - lam)))
    lla, llb = np.array(lla), np.array(llb)
    pos, nrow = np.array(pos), np.array(nrow)
    rng = np.random.default_rng(SEED)
    seqs = stationary_window_indices(len(wins), B_BOOT, MEAN_BLOCK, rng)
    d_ig = np.empty(B_BOOT); d_pr = np.full(B_BOOT, np.nan)
    for i in range(B_BOOT):
        s = seqs[i]
        d_ig[i] = (lla[s].sum() - llb[s].sum()) / max(pos[s].sum(), 1.0)
        if 0.0 < pos[s].sum() < nrow[s].sum():
            yy = np.concatenate([y_by[j] for j in s])
            d_pr[i] = (average_precision_score(yy, np.concatenate([a_by[j] for j in s]))
                       - average_precision_score(yy, np.concatenate([b_by[j] for j in s])))
    def ci(v):
        v = v[np.isfinite(v)]
        return [round(float(np.percentile(v, 2.5)), 6), round(float(np.percentile(v, 97.5)), 6)]
    return {"d_ig": {"point": round(float(information_gain(P_a, P_b, y)), 6), "ci95": ci(d_ig)},
            "d_pr_auc": {"point": round(float(average_precision_score(y, P_a)
                                              - average_precision_score(y, P_b)), 6),
                         "ci95": ci(d_pr)}}


def main():
    t_all = time.time()
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    m = split_masks(grid)
    t0 = pd.to_datetime(grid["t0"])
    inner = m["train"] & (t0 < INNER_END).to_numpy()
    tail = m["train"] & (t0 >= INNER_END).to_numpy() & (t0 < TAIL_END).to_numpy()
    print(f"train {m['train'].sum()} rows | inner-train (t0 < {INNER_END.date()}) {inner.sum()} "
          f"| temporal tail (2021) {tail.sum()} | val {m['val'].sum()} | test {m['test'].sum()}",
          flush=True)
    print(f"tail windows: {pd.Series(grid['window'].to_numpy()[tail]).nunique()}", flush=True)

    out = {"governed_by": {"amendment": "docs/preregistration/v2_analysis_amendment_5.md",
                           "sha256": "c97db8f54374ac4ff1b5fbfafc1a1e76c63d68077144b338319603170ce846c2",
                           "item": "F"},
           "design": {"inner_train_end": str(INNER_END.date()), "tail": "2021 windows",
                      "candidates": CANDIDATES, "early_stopping": False,
                      "note": "only the ML stage changes; cascade and first-gen are frozen physics"},
           "targets": {}}

    for tgt, (count_col, lam_col, ycol) in TARGETS.items():
        feats = FEATURES
        X = grid[feats].copy()
        X["ln_lam_sim"] = np.log(grid[lam_col].to_numpy() + EPS)
        n = grid[count_col].to_numpy().astype(float)
        y = grid[ycol].to_numpy().astype(float)
        lam_sim = np.clip(grid[lam_col].to_numpy(), EPS, None)
        mono = _monotonic(feats + ["ln_lam_sim"])

        # --- select tree count on the temporal tail ---
        curve = {}
        for c in CANDIDATES:
            r = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=c,
                                              max_depth=6, monotonic_cst=mono, random_state=42,
                                              early_stopping=False)
            r.fit(X[inner], n[inner])
            lam_t = np.clip(r.predict(X[tail]), EPS, None)
            curve[c] = round(poisson_ll(n[tail], lam_t), 3)
            print(f"  [{tgt}] n_trees={c:3d}  tail Poisson-LL {curve[c]:.2f} "
                  f"({time.time()-t_all:.0f}s)", flush=True)
        best = max(curve, key=lambda k: curve[k])

        # --- refit on FULL training at the selected count, no internal stop ---
        reg = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=best,
                                            max_depth=6, monotonic_cst=mono, random_state=42,
                                            early_stopping=False)
        reg.fit(X[m["train"]], n[m["train"]])
        lam_ml = np.clip(reg.predict(X), EPS, None)
        w, wdiag = select_w_1se(lam_sim, lam_ml, y, m["val"], grid["window"].to_numpy(), WEIGHTS)
        P_hyb = lambda_to_p(lam_sim ** (1 - w) * lam_ml ** w)

        # --- verdicts on test, against the FROZEN physics ---
        pred = pd.read_parquet(RESULTS / f"predictions_{tgt}.parquet")
        sel = np.where(m["val"] | m["test"])[0]
        te_mask = (pred["split"] == "test").to_numpy()
        idx_test = sel[te_mask]
        yt = y[idx_test]; wt = grid["window"].to_numpy()[idx_test]
        P_h = P_hyb[idx_test]
        rec = {"tail_ll_curve": curve, "selected_trees": int(best),
               "shipped_trees": {"y30": 147, "y35": 10}[tgt],
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
        # shipped comparison, for the diff
        P_ship = pred["hybrid"].to_numpy(float)[te_mask]
        rec["max_abs_dev_vs_shipped_hybrid"] = float(np.abs(P_h - P_ship).max())
        out["targets"][tgt] = rec
        print(f"  [{tgt}] selected {best} trees (shipped {rec['shipped_trees']}), "
              f"w={w} (shipped {rec['shipped_w']})", flush=True)
        for k, st in rec["verdicts"].items():
            print(f"      {k:26s} dIG {st['d_ig']['point']:+.4f} {st['d_ig']['ci95']}  "
                  f"dPR {st['d_pr_auc']['point']:+.4f} {st['d_pr_auc']['ci95']}  -> {st['verdict']}")

    # shipped verdicts for comparison
    shipped = {"y30": {"hybrid_vs_cascade": "inseparable", "hybrid_vs_sv_etas": "inseparable",
                       "hybrid_vs_firstgen_etas": "inseparable"},
               "y35": {"hybrid_vs_cascade": "inseparable", "hybrid_vs_sv_etas": "inseparable",
                       "hybrid_vs_firstgen_etas": "inseparable"}}
    diffs = []
    for tgt, rec in out["targets"].items():
        for k, st in rec["verdicts"].items():
            if st["verdict"] != shipped[tgt][k]:
                diffs.append(f"{tgt}/{k}: {shipped[tgt][k]} -> {st['verdict']}")
    out["verdict_diff_vs_shipped"] = diffs
    out["any_verdict_changed"] = bool(diffs)
    out["runtime_s"] = round(time.time() - t_all, 1)
    json.dump(out, open(R4 / "r12_item_F.json", "w"), indent=2)
    print(f"\nverdict changes vs shipped: {diffs if diffs else 'NONE'}")
    print(f"runtime {out['runtime_s']}s -> results/round4/r12_item_F.json")


if __name__ == "__main__":
    main()
