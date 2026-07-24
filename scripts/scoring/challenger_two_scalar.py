"""Last gem: the two-parameter dynamic challenger (s_bg, s_trig) on the ETAS background/triggered
split (Tier-1 plan specified it; only the single scalar was reported). Fit two constants on
validation Bernoulli-occurrence LL, score under the construction-free proxy vs the cascade. If it
reproduces ~+0.10, the ML's entire surviving edge is a re-fit of two ETAS calibration constants
(deployable without ML; the bridge to mu(x,t)/K(x)); if not, "dynamic feature-rich" stands.
Also emits the proper-score quiet/active regime split. Writes results/round3/t8_two_scalar.json.
"""
import json
import pickle
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import average_precision_score

from marmara.paths import RESULTS
from marmara.etas_model import KM_PER_DEG
from marmara.metrics import p_to_lambda, lambda_to_p
from marmara.train import split_masks
from marmara.bootstrap import stationary_window_indices, MEAN_BLOCK, SEED

EPS = 1e-9


def lam(p):
    return np.clip(p_to_lambda(np.clip(np.asarray(p, float), 0.0, 1.0)), EPS, None)


def bern_ig_ci(win, y, pa, pb):
    pa = np.clip(pa, EPS, 1 - EPS); pb = np.clip(pb, EPS, 1 - EPS)
    c = y * (np.log(pa) - np.log(pb)) + (1 - y) * (np.log(1 - pa) - np.log(1 - pb))
    wins = np.sort(np.unique(win)); idx = {w: np.where(win == w)[0] for w in wins}
    cw = np.array([c[idx[w]].sum() for w in wins]); pos = np.array([y[idx[w]].sum() for w in wins])
    ig = cw.sum() / max(pos.sum(), 1)
    rng = np.random.default_rng(SEED); seqs = stationary_window_indices(len(wins), 2000, MEAN_BLOCK, rng)
    bs = [cw[r].sum() / max(pos[r].sum(), 1) for r in seqs]
    return round(float(ig), 4), [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def main():
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    m = split_masks(grid)
    params = pickle.load(open(RESULTS / "etas" / "etas_params.pkl", "rb"))
    bg = params.background_xy
    clat = grid["cell_lat"].to_numpy(); clon = grid["cell_lon"].to_numpy()
    area = (0.1 * KM_PER_DEG) * (0.1 * KM_PER_DEG * np.cos(np.radians(clat)))
    lam_bg = np.clip(params.mu_total * 30.0 * bg.pdf_lonlat(clon, clat) * area, 0.0, None)  # background rate/cell
    lam_sim = np.clip(grid["lam30_sim"].to_numpy(), EPS, None)
    lam_trig = np.clip(lam_sim - lam_bg, 0.0, None)                                          # triggered rate
    y = grid["y30"].to_numpy(float); win = grid["window"].to_numpy()
    va, te = m["val"], m["test"]

    # fit s_bg, s_trig on val Bernoulli-occurrence LL
    def negLL(theta):
        s_bg, s_tr = np.exp(theta)   # positivity
        lm = s_bg * lam_bg[va] + s_tr * lam_trig[va]
        p = np.clip(1 - np.exp(-lm), EPS, 1 - EPS)
        return -np.sum(y[va] * np.log(p) + (1 - y[va]) * np.log(1 - p))
    res = minimize(negLL, np.log([0.4, 0.4]), method="Nelder-Mead")
    s_bg, s_tr = np.exp(res.x)

    lm_te = s_bg * lam_bg[te] + s_tr * lam_trig[te]
    p_2s = np.clip(1 - np.exp(-lm_te), EPS, 1 - EPS)
    # cascade, globally occurrence-calibrated (proxy), for apples-to-apples
    s_c = y[va].sum() / lam_sim[va].sum()
    p_casc = np.clip(1 - np.exp(-s_c * lam_sim[te]), EPS, 1 - EPS)
    ig, ci = bern_ig_ci(win[te], y[te], p_2s, p_casc)

    out = {"s_bg": round(float(s_bg), 4), "s_trig": round(float(s_tr), 4),
           "bg_frac_of_cascade_rate": round(float(lam_bg.sum() / lam_sim.sum()), 3),
           "two_scalar_ig_vs_cascade_proxy_bernoulli": ig, "ci": ci,
           "ml_hybrid_edge_for_reference": 0.099,
           "reading": ("if ~+0.10, the ML's surviving edge = re-fit of two ETAS calibration constants "
                       "(s_bg, s_trig); deployable without ML. If << +0.10, dynamic feature-rich stands.")}

    # proper-score quiet/active regime split (hybrid vs cascade, proxy)
    pred = pd.read_parquet(RESULTS / "grid" / "predictions_y30.parquet")
    pt = pred[pred.split == "test"].reset_index(drop=True)
    ght = grid[te].reset_index(drop=True)
    lam_h = lam(pt.hybrid.to_numpy()); lam_c = lam(pt.cascade.to_numpy())
    s_h = pred[pred.split == "val"].y.sum() / lam(pred[pred.split == "val"].hybrid.to_numpy()).sum()
    s_cc = pred[pred.split == "val"].y.sum() / lam(pred[pred.split == "val"].cascade.to_numpy()).sum()
    ph = np.clip(1 - np.exp(-s_h * lam_h), EPS, 1 - EPS); pc = np.clip(1 - np.exp(-s_cc * lam_c), EPS, 1 - EPS)
    days45 = ght["days_since_m45_25km"].to_numpy(); active = days45 < 365.0
    yt = pt.y.to_numpy(); wtt = pt.window.to_numpy()
    reg = {}
    for nm, mask in (("active", active), ("quiet", ~active)):
        if yt[mask].sum() > 0:
            igx, cix = bern_ig_ci(wtt[mask], yt[mask], ph[mask], pc[mask])
            reg[nm] = {"n_cells": int(mask.sum()), "n_pos": int(yt[mask].sum()), "ig": igx, "ci": cix}
    out["proper_score_regime_split_hybrid_vs_cascade"] = reg

    (RESULTS / "round3" / "t8_two_scalar.json").write_text(json.dumps(out, indent=2))
    print(f"s_bg={s_bg:.3f} s_trig={s_tr:.3f} (bg is {out['bg_frac_of_cascade_rate']*100:.0f}% of cascade rate)")
    print(f"TWO-SCALAR IG vs cascade (proxy Bernoulli): {ig} {ci}  (ML hybrid ~+0.10)")
    print("proper-score regime split:", {k: (v['ig'], v['ci']) for k, v in reg.items()})


if __name__ == "__main__":
    main()
