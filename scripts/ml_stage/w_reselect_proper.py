"""Re-select the blend weight w under a proper (Bernoulli occurrence) validation objective.

The canonical w=0.8 was chosen by a validation Poisson objective now shown to reward artifact-
harvesting (count-term ~140% on val). Here: for each w, occurrence-calibrate the blend by its own
val scalar s_w = N_pos_val/sum(lambda_w,val), p_w = 1-e^{-s_w*lambda_w}, and score the val
Bernoulli log-likelihood; apply the SAME pre-registered 1-SE parsimony rule (smallest w within one
block-bootstrap SE of the argmax). Booster frozen. If w collapses toward 0, the selection rule
abandons the ML stage once the artifact is removed. Writes results/round3/t4_w_reselect.json.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from marmara.paths import RESULTS
from marmara.grid import FEATURES
from marmara.train import split_masks, _monotonic, WEIGHTS, select_w_1se
from marmara.bootstrap import stationary_window_indices, MEAN_BLOCK, SEED

EPS = 1e-9


def bern_1se(lam_sim, lam_ml, y, va, win, weights):
    """1-SE selection on the val BERNOULLI-occurrence objective (each w occurrence-calibrated)."""
    wv = win[va]; uw = np.unique(wv); idx = np.searchsorted(uw, wv); n_win = len(uw)
    ls, lm, yv = lam_sim[va], lam_ml[va], y[va]
    llc = np.zeros((len(weights), n_win)); curve = np.zeros(len(weights))
    for i, w in enumerate(weights):
        lamw = np.clip(ls ** (1 - w) * lm ** w, EPS, None)
        s = yv.sum() / lamw.sum()
        p = np.clip(1 - np.exp(-s * lamw), EPS, 1 - EPS)
        cell = yv * np.log(p) + (1 - yv) * np.log(1 - p)
        np.add.at(llc[i], idx, cell)
        curve[i] = cell.sum()
    istar = int(np.argmax(curve))
    rng = np.random.default_rng(SEED); seqs = stationary_window_indices(n_win, 2000, MEAN_BLOCK, rng)
    boot = llc[:, seqs].sum(axis=2)
    gap_point = curve[istar] - curve
    gap_se = (boot[istar][None, :] - boot).std(axis=1)
    within = gap_point <= gap_se
    isel = int(np.where(within)[0].min())
    return float(weights[isel]), float(weights[istar]), {round(float(weights[i]), 1): round(float(curve[i]), 2) for i in range(len(weights))}


def main():
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    m = split_masks(grid); tr, va = m["train"], m["val"]
    y = grid["y30"].to_numpy(float); n = grid["count30"].to_numpy(float)
    win = grid["window"].to_numpy(); lam_sim = np.clip(grid["lam30_sim"].to_numpy(), EPS, None)
    X = grid[FEATURES].copy(); X["ln_lam_sim"] = np.log(lam_sim + EPS)
    reg = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=400, max_depth=6,
                                        monotonic_cst=_monotonic(list(X.columns)), random_state=42)
    reg.fit(X[tr], n[tr]); lam_ml = np.clip(reg.predict(X), EPS, None)

    w_pois, wdiag = select_w_1se(lam_sim, lam_ml, y, va, win, WEIGHTS)
    w_bern, w_bern_argmax, bern_curve = bern_1se(lam_sim, lam_ml, y, va, win, WEIGHTS)
    out = {
        "w_selected_POISSON_objective": w_pois, "w_argmax_poisson": wdiag["w_argmax"],
        "w_selected_BERNOULLI_occurrence_objective": w_bern, "w_argmax_bernoulli": w_bern_argmax,
        "poisson_val_ll_curve": wdiag["ll_curve"],
        "bernoulli_val_ll_curve": bern_curve,
        "interpretation": ("if w_bernoulli << w_poisson (esp. -> 0), the parsimony rule abandons the ML "
                           "stage once the occurrence artifact is removed from the objective."),
    }
    (RESULTS / "round3" / "t4_w_reselect.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("w_selected_POISSON_objective", "w_argmax_poisson",
          "w_selected_BERNOULLI_occurrence_objective", "w_argmax_bernoulli")}, indent=1))
    print("bernoulli val-LL curve:", bern_curve)


if __name__ == "__main__":
    main()
