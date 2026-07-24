"""T1-5: was the ML given a fair shot on booster CAPACITY?

The paper tunes only w and sigma; the booster's depth/iter/lr/l2 were never searched. Validation
grid search (Poisson-LL of the 1-SE-blended hybrid), then score the best config on test vs the
cascade. Writes results/round3/t1_tune.json. Nothing canonical is overwritten.
"""
from __future__ import annotations

import itertools
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score

from marmara.paths import RESULTS
from marmara.grid import FEATURES
from marmara.metrics import lambda_to_p, p_to_lambda
from marmara.train import split_masks, select_w_1se, _monotonic, WEIGHTS
from marmara.bootstrap import stationary_window_indices, MEAN_BLOCK, SEED

EPS = 1e-9
GRID = {"max_depth": [3, 6, 10], "max_iter": [400, 800], "learning_rate": [0.05, 0.1], "l2_regularization": [0.0, 1.0]}


def lam(p):
    return np.clip(p_to_lambda(np.clip(np.asarray(p, float), 0.0, 1.0)), EPS, None)


def poisson_ll(y, la):
    la = np.clip(la, EPS, None)
    return float(np.sum(y * np.log(la) - la))


def main():
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    m = split_masks(grid); tr, va, te = m["train"], m["val"], m["test"]
    y = grid["y30"].to_numpy(float); n = grid["count30"].to_numpy(float)
    win = grid["window"].to_numpy(); lam_sim = np.clip(grid["lam30_sim"].to_numpy(), EPS, None)
    X = grid[FEATURES].copy(); X["ln_lam_sim"] = np.log(lam_sim + EPS)
    mono = _monotonic(list(X.columns))

    keys = list(GRID); best = None; trials = []
    for combo in itertools.product(*[GRID[k] for k in keys]):
        p = dict(zip(keys, combo))
        reg = HistGradientBoostingRegressor(loss="poisson", monotonic_cst=mono, random_state=42, **p)
        reg.fit(X[tr], n[tr])
        lam_ml = np.clip(reg.predict(X), EPS, None)
        w, _ = select_w_1se(lam_sim, lam_ml, y, va, win, WEIGHTS)
        lam_h = lam_sim ** (1 - w) * lam_ml ** w
        vll = poisson_ll(y[va], lam_h[va])
        trials.append({**p, "w": w, "val_ll": round(vll, 2)})
        if best is None or vll > best["val_ll"]:
            best = {**p, "w": w, "val_ll": round(vll, 2), "lam_h": lam_h, "lam_ml": lam_ml}

    # score best on test vs cascade
    wt = win[te]; yt = y[te]; lh = np.clip(best["lam_h"][te], EPS, None); lc = lam_sim[te]
    wins = np.sort(np.unique(wt)); idx = {w: np.where(wt == w)[0] for w in wins}
    ca = np.array([np.sum(yt[idx[w]] * np.log(lh[idx[w]]) - lh[idx[w]]) for w in wins])
    cb = np.array([np.sum(yt[idx[w]] * np.log(lc[idx[w]]) - lc[idx[w]]) for w in wins])
    pos = np.array([yt[idx[w]].sum() for w in wins])
    ig = (ca.sum() - cb.sum()) / max(pos.sum(), 1)
    rng = np.random.default_rng(SEED); seqs = stationary_window_indices(len(wins), 2000, MEAN_BLOCK, rng)
    igbs = [(ca[r].sum() - cb[r].sum()) / max(pos[r].sum(), 1) for r in seqs]
    pr_h = float(average_precision_score(yt, lambda_to_p(lh)))
    pr_c = float(average_precision_score(yt, lambda_to_p(lc)))
    yb = [yt[idx[w]] for w in wins]; hb = [lambda_to_p(lh)[idx[w]] for w in wins]; cbp = [lambda_to_p(lc)[idx[w]] for w in wins]
    dprbs = []
    for r in seqs:
        yy = np.concatenate([yb[i] for i in r])
        if 0 < yy.sum() < len(yy):
            dprbs.append(average_precision_score(yy, np.concatenate([hb[i] for i in r]))
                         - average_precision_score(yy, np.concatenate([cbp[i] for i in r])))
    dpr_ci = [round(float(np.percentile(dprbs, 2.5)), 4), round(float(np.percentile(dprbs, 97.5)), 4)]
    ig_ci = [round(float(np.percentile(igbs, 2.5)), 4), round(float(np.percentile(igbs, 97.5)), 4)]

    out = {"n_configs": len(trials), "canonical_val_ll": -2003.98,
           "best_config": {k: best[k] for k in list(GRID) + ["w", "val_ll"]},
           "best_on_test": {"ig_vs_cascade": round(float(ig), 4), "ig_ci": ig_ci,
                            "hybrid_pr": round(pr_h, 4), "cascade_pr": round(pr_c, 4),
                            "dpr": round(pr_h - pr_c, 4), "dpr_ci": dpr_ci,
                            "separable_two_axis": bool((ig_ci[0] > 0 or ig_ci[1] < 0) and (dpr_ci[0] > 0 or dpr_ci[1] < 0))},
           "all_trials": sorted(trials, key=lambda t: -t["val_ll"])[:8]}
    (RESULTS / "round3" / "t1_tune.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({"best_config": out["best_config"], "best_on_test": out["best_on_test"]}, indent=1))


if __name__ == "__main__":
    main()
