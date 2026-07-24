"""Tier-1 booster battery: was the ML stage given a fair shot?

Reproduces the canonical hybrid, then runs:
  T1-3 argmax-w      : blend at the rejected argmax w (two-axis vs cascade)
  T1-4 drop-monotone : retrain booster monotonic_cst=None; val LL + test
  T1-6 offset        : Poisson-offset model lambda = lam_sim * nu(x) via exposure-weight trick
  T1-7 pure-ML       : w=1 (lam_ml alone, keeps ln lam_sim feature) AND w=1 without ln lam_sim
  T1-8 rank proxy    : HistGB *classifier* (log-loss) on occurrence -> PR-AUC vs cascade
  T1-9 regime split  : hybrid vs cascade IG on ACTIVE (recent M>=4.5 within 25 km) vs QUIET cells

Validation triage: report val Poisson-LL for every variant; test IG(vs cascade) always (cheap
window bootstrap CI); PR-AUC CI only where PR-AUC materially beats the cascade. Writes nothing
canonical (no models/ or predictions_* overwrite). -> results/round3/t1_booster.json
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score

from marmara.paths import RESULTS
from marmara.grid import FEATURES
from marmara.metrics import lambda_to_p, p_to_lambda
from marmara.train import split_masks, select_w_1se, _monotonic, WEIGHTS
from marmara.bootstrap import stationary_window_indices, MEAN_BLOCK, SEED

EPS = 1e-9
B = 2000
TARGET = ("y30", "count30", "lam30_sim", 3.0)   # primary


def lam(p):
    return np.clip(p_to_lambda(np.clip(np.asarray(p, float), 0.0, 1.0)), EPS, None)


def ig_ci(win, y, la, lb):
    wins = np.sort(np.unique(win)); idx = {w: np.where(win == w)[0] for w in wins}
    ca = np.array([np.sum(y[idx[w]] * np.log(la[idx[w]]) - la[idx[w]]) for w in wins])
    cb = np.array([np.sum(y[idx[w]] * np.log(lb[idx[w]]) - lb[idx[w]]) for w in wins])
    pos = np.array([y[idx[w]].sum() for w in wins])
    pt = (ca.sum() - cb.sum()) / max(pos.sum(), 1)
    rng = np.random.default_rng(SEED); seqs = stationary_window_indices(len(wins), B, MEAN_BLOCK, rng)
    bs = [(ca[r].sum() - cb[r].sum()) / max(pos[r].sum(), 1) for r in seqs]
    return round(float(pt), 4), [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def pr_ci(win, y, pa, pb):
    wins = np.sort(np.unique(win)); idx = {w: np.where(win == w)[0] for w in wins}
    yb = [y[idx[w]] for w in wins]; ab = [pa[idx[w]] for w in wins]; bb = [pb[idx[w]] for w in wins]
    rng = np.random.default_rng(SEED); seqs = stationary_window_indices(len(wins), B, MEAN_BLOCK, rng)
    d = []
    for r in seqs:
        yy = np.concatenate([yb[i] for i in r])
        if 0 < yy.sum() < len(yy):
            d.append(average_precision_score(yy, np.concatenate([ab[i] for i in r]))
                     - average_precision_score(yy, np.concatenate([bb[i] for i in r])))
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


def fit_booster(X, n, tr, mono=True, classifier=False, offset=None):
    if classifier:
        m = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, max_depth=6, random_state=42)
        m.fit(X[tr], (n[tr] > 0).astype(int))
        return m.predict_proba(X)[:, 1]
    kw = dict(loss="poisson", learning_rate=0.05, max_iter=400, max_depth=6, random_state=42)
    if mono:
        kw["monotonic_cst"] = _monotonic(list(X.columns))
    m = HistGradientBoostingRegressor(**kw)
    if offset is not None:               # exposure-weight Poisson offset: target n/off, weight off
        m.fit(X[tr], (n[tr] / np.clip(offset[tr], EPS, None)), sample_weight=np.clip(offset[tr], EPS, None))
        return np.clip(m.predict(X) * np.clip(offset, EPS, None), EPS, None)
    m.fit(X[tr], n[tr])
    return np.clip(m.predict(X), EPS, None)


def poisson_ll(y, la):
    return float(np.sum(y * np.log(np.clip(la, EPS, None)) - np.clip(la, EPS, None)))


def main():
    ycol, ccol, lcol, thr = TARGET
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    m = split_masks(grid)
    tr, va, te = m["train"], m["val"], m["test"]
    y = grid[ycol].to_numpy(float); n = grid[ccol].to_numpy(float)
    win = grid["window"].to_numpy(); lam_sim = np.clip(grid[lcol].to_numpy(), EPS, None)
    Xbase = grid[FEATURES].copy(); Xbase["ln_lam_sim"] = np.log(lam_sim + EPS)
    wt = win[te]; yt = y[te]; lc_t = lam_sim[te]

    res = {"target": ycol, "n_test_pos": int(yt.sum())}

    def blend(lm, w):
        return lam_sim ** (1 - w) * lm ** w

    def report(name, lam_pred, cheap=True):
        la_t = np.clip(lam_pred[te], EPS, None)
        ig, ci = ig_ci(wt, yt, la_t, lc_t)
        pr = float(average_precision_score(yt, lambda_to_p(la_t)))
        pr_c = float(average_precision_score(yt, lambda_to_p(lc_t)))
        entry = {"val_poisson_ll": round(poisson_ll(y[va], lam_pred[va]), 2),
                 "test_ig_vs_cascade": ig, "test_ig_ci": ci,
                 "test_pr_auc": round(pr, 4), "cascade_pr_auc": round(pr_c, 4),
                 "test_dpr": round(pr - pr_c, 4)}
        if not cheap and (pr - pr_c) > 0.005:      # candidate: PR beats cascade -> full CI
            entry["test_dpr_ci"] = pr_ci(wt, yt, lambda_to_p(la_t), lambda_to_p(lc_t))
            entry["separable_two_axis"] = bool((ci[0] > 0 or ci[1] < 0)
                                               and (entry["test_dpr_ci"][0] > 0 or entry["test_dpr_ci"][1] < 0))
        return entry

    # ---- baseline: canonical hybrid ----
    lam_ml = fit_booster(Xbase, n, tr, mono=True)
    w_sel, wdiag = select_w_1se(lam_sim, lam_ml, y, va, win, WEIGHTS)
    res["canonical_hybrid"] = {"w_sel": w_sel, "w_argmax": wdiag["w_argmax"], **report("hyb", blend(lam_ml, w_sel), cheap=False)}

    # ---- T1-3 argmax-w ----
    res["T1_3_argmax_w"] = {"w": wdiag["w_argmax"], **report("argmax", blend(lam_ml, wdiag["w_argmax"]), cheap=False)}

    # ---- T1-7 pure-ML: w=1 (keeps ln lam_sim feature) ----
    res["T1_7_pureML_w1_with_lnsim"] = report("pml", lam_ml, cheap=False)  # w=1 => lam_ml
    #        pure-ML without ln lam_sim (can features alone rediscover ETAS skill?)
    Xno = grid[FEATURES].copy()
    lam_ml_no = fit_booster(Xno, n, tr, mono=True)
    res["T1_7_pureML_w1_no_lnsim"] = report("pmlno", lam_ml_no, cheap=False)

    # ---- T1-4 drop monotonicity ----
    lam_ml_free = fit_booster(Xbase, n, tr, mono=False)
    w_free, wdf = select_w_1se(lam_sim, lam_ml_free, y, va, win, WEIGHTS)
    res["T1_4_drop_monotone"] = {"w_sel": w_free, "w_argmax": wdf["w_argmax"], **report("free", blend(lam_ml_free, w_free), cheap=False)}

    # ---- T1-6 offset (Poisson exposure-weight, no ln lam_sim feature) ----
    lam_off = fit_booster(Xno, n, tr, mono=True, offset=lam_sim)
    res["T1_6_offset"] = report("off", lam_off, cheap=False)

    # ---- T1-8 rank proxy: HistGB classifier on occurrence ----
    p_rank = fit_booster(Xbase, n, tr, classifier=True)
    la_rank = lam(p_rank)
    res["T1_8_rank_proxy"] = report("rank", la_rank, cheap=False)

    # ---- T1-9 regime split: hybrid vs cascade on ACTIVE vs QUIET test cells ----
    hyb_t = np.clip(blend(lam_ml, w_sel)[te], EPS, None)
    days45 = grid["days_since_m45_25km"].to_numpy()[te]
    active = days45 < 365.0     # recent M>=4.5 within 25 km in last year
    def sub_ig(mask):
        if mask.sum() == 0 or yt[mask].sum() == 0:
            return None
        ig, ci = ig_ci(wt[mask], yt[mask], hyb_t[mask], lc_t[mask])
        pr_h = float(average_precision_score(yt[mask], lambda_to_p(hyb_t[mask]))) if 0 < yt[mask].sum() < mask.sum() else None
        pr_c = float(average_precision_score(yt[mask], lambda_to_p(lc_t[mask]))) if 0 < yt[mask].sum() < mask.sum() else None
        return {"n_cells": int(mask.sum()), "n_pos": int(yt[mask].sum()), "ig": ig, "ig_ci": ci,
                "hybrid_pr": round(pr_h, 4) if pr_h else None, "cascade_pr": round(pr_c, 4) if pr_c else None}
    res["T1_9_regime_split"] = {"active_cells": sub_ig(active), "quiet_cells": sub_ig(~active),
                                "active_def": "days_since_m45_25km < 365"}

    (RESULTS / "round3" / "t1_booster.json").write_text(json.dumps(res, indent=2))
    print(json.dumps({k: (v if not isinstance(v, dict) else {kk: v[kk] for kk in v if kk in
          ("w_sel", "w", "w_argmax", "val_poisson_ll", "test_ig_vs_cascade", "test_ig_ci",
           "test_pr_auc", "cascade_pr_auc", "test_dpr", "separable_two_axis")})
          for k, v in res.items() if isinstance(v, dict)}, indent=1))


if __name__ == "__main__":
    main()
