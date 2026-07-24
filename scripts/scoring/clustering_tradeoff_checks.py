"""Closing gems:
 #1a clustering trade-off: scalar-proxy Bernoulli cascade-vs-first-gen (predict ~+0.07 in cascade's
     favour = the placement the cascade's clustering earns back after removing global calibration).
 #2  static map: fit a static per-cell multiplicative correction on TRAIN windows only, apply to the
     cascade, score under the construction-free proxy Bernoulli vs cascade. If ~+0.10, the ML's
     surviving edge is a learned STATIC recalibration map (not dynamics).
Writes results/round3/t7_gems.json.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from marmara.paths import RESULTS
from marmara.metrics import p_to_lambda
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


def occ_cal(lam_arr, s):
    return np.clip(1 - np.exp(-s * lam_arr), EPS, 1 - EPS)


def main():
    pred = pd.read_parquet(RESULTS / "grid" / "predictions_y30.parquet")
    va = pred[pred.split == "val"]; te = pred[pred.split == "test"]
    yv = va.y.to_numpy(); yt = te.y.to_numpy(); wt = te.window.to_numpy()
    out = {}

    # ---- Gem #1a: scalar-proxy Bernoulli cascade-vs-first-gen ----
    Lc_v, Lf_v = lam(va.cascade.to_numpy()), lam(va.firstgen_etas.to_numpy())
    Lc_t, Lf_t = lam(te.cascade.to_numpy()), lam(te.firstgen_etas.to_numpy())
    s_c = yv.sum() / Lc_v.sum(); s_f = yv.sum() / Lf_v.sum()
    ig, ci = bern_ig_ci(wt, yt, occ_cal(Lc_t, s_c), occ_cal(Lf_t, s_f))
    out["gem1a_proxy_bernoulli_cascade_vs_firstgen"] = {"ig": ig, "ci": ci,
        "prediction": "+0.07 (placement clustering earns back)",
        "reading": "native -0.056 = calibration -0.124 (cascade worse) + placement (this, cascade better)"}

    # ---- Gem #2: static per-cell map from TRAIN, applied to cascade, proxy-Bernoulli vs cascade ----
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    m = split_masks(grid)
    tr = m["train"]
    cell = grid["ir"].to_numpy() * 100000 + grid["ic"].to_numpy()
    cnt = grid["count30"].to_numpy(float); lsim = np.clip(grid["lam30_sim"].to_numpy(), EPS, None)
    # per-cell shrunk multiplicative factor: (sum_train count + a)/(sum_train lam + a)
    a = 1.0
    df_tr = pd.DataFrame({"cell": cell[tr], "cnt": cnt[tr], "lam": lsim[tr]}).groupby("cell").sum()
    factor = ((df_tr["cnt"] + a) / (df_tr["lam"] + a)).to_dict()
    fac_all = np.array([factor.get(int(c), 1.0) for c in cell])
    lam_static = fac_all * lsim
    # occurrence-calibrate on val, score on test vs cascade (both proxy)
    vmask, tmask = m["val"], m["test"]
    s_st = grid["y30"].to_numpy()[vmask].sum() / lam_static[vmask].sum()
    s_ca = grid["y30"].to_numpy()[vmask].sum() / lsim[vmask].sum()
    yts = grid["y30"].to_numpy()[tmask]; wts = grid["window"].to_numpy()[tmask]
    ig2, ci2 = bern_ig_ci(wts, yts, occ_cal(lam_static[tmask], s_st), occ_cal(lsim[tmask], s_ca))
    # ranking of the static map (intensity PR) vs cascade
    from marmara.metrics import lambda_to_p
    pr_static = float(average_precision_score(yts, lambda_to_p(lam_static[tmask])))
    pr_casc = float(average_precision_score(yts, lambda_to_p(lsim[tmask])))
    out["gem2_static_percell_map"] = {"ig_vs_cascade_proxy_bernoulli": ig2, "ci": ci2,
        "hybrid_edge_for_reference": 0.099,
        "static_map_intensity_PR": round(pr_static, 4), "cascade_intensity_PR": round(pr_casc, 4),
        "reading": "if ~+0.10, the ML's surviving edge is a learned STATIC per-cell recalibration map"}

    out["O_h_topup_measured"] = {"O_h": 748.3, "O_cascade": 906.7, "predicted_if_geometry_closed": 971,
        "reading": "measured 748 != 971 -> ML trims the total (748<907), does not merely restructure"}
    out["native_edge_bracket"] = {"capped": 0.183, "topup_conservative": 0.095, "proxy": 0.099,
        "framing": "native edge in [+0.095,+0.183]; quote conservative +0.095; proxy lands there"}

    (RESULTS / "round3" / "t7_gems.json").write_text(json.dumps(out, indent=2))
    print("GEM1a proxy-Bernoulli cascade-vs-firstgen:", ig, ci, "(predicted ~+0.07)")
    print("GEM2 static per-cell map IG vs cascade   :", ig2, ci2, "(hybrid edge ~+0.10)")
    print("     static map PR", round(pr_static, 4), "vs cascade PR", round(pr_casc, 4))


if __name__ == "__main__":
    main()
