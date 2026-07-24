"""Final licensing checks before writing:
 (1) PAIRED hybrid-vs-two-scalar dIG interval (proxy Bernoulli, same bootstrap) -> fills the [share]
     slot: excludes 0 in hybrid's favour = irreducible dynamic remainder resolved; straddles = two
     constants inseparable from the ML stage (point-share ~half).
 (2) gem: nesting lock (rate-weighted mean of s_bg,s_trig ~ single scalar) + s_trig<s_bg saturation.
 (3) recombination: active/quiet positive fractions x regime IG -> overall +0.10 (no leak).
 (4) per-window concentration of the active-cell edge: pre vs post Kumburgaz.
Writes results/round3/t9_final_checks.json.
"""
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from marmara.paths import RESULTS
from marmara.etas_model import KM_PER_DEG
from marmara.metrics import p_to_lambda
from marmara.train import split_masks
from marmara.bootstrap import stationary_window_indices, MEAN_BLOCK, SEED

EPS = 1e-9


def lam(p):
    return np.clip(p_to_lambda(np.clip(np.asarray(p, float), 0.0, 1.0)), EPS, None)


def paired_bern_ig(win, y, pa, pb):
    pa = np.clip(pa, EPS, 1 - EPS); pb = np.clip(pb, EPS, 1 - EPS)
    c = y * (np.log(pa) - np.log(pb)) + (1 - y) * (np.log(1 - pa) - np.log(1 - pb))
    wins = np.sort(np.unique(win)); idx = {w: np.where(win == w)[0] for w in wins}
    cw = np.array([c[idx[w]].sum() for w in wins]); pos = np.array([y[idx[w]].sum() for w in wins])
    ig = cw.sum() / max(pos.sum(), 1)
    rng = np.random.default_rng(SEED); seqs = stationary_window_indices(len(wins), 2000, MEAN_BLOCK, rng)
    bs = [cw[r].sum() / max(pos[r].sum(), 1) for r in seqs]
    return round(float(ig), 4), [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def main():
    t8 = json.load(open(RESULTS / "round3" / "t8_two_scalar.json"))
    s_bg, s_tr = t8["s_bg"], t8["s_trig"]; bgf = t8["bg_frac_of_cascade_rate"]
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet"); m = split_masks(grid)
    params = pickle.load(open(RESULTS / "etas" / "etas_params.pkl", "rb")); bgfield = params.background_xy
    clat = grid["cell_lat"].to_numpy(); clon = grid["cell_lon"].to_numpy()
    area = (0.1 * KM_PER_DEG) * (0.1 * KM_PER_DEG * np.cos(np.radians(clat)))
    lam_bg = np.clip(params.mu_total * 30.0 * bgfield.pdf_lonlat(clon, clat) * area, 0.0, None)
    lam_sim = np.clip(grid["lam30_sim"].to_numpy(), EPS, None)
    lam_trig = np.clip(lam_sim - lam_bg, 0.0, None)
    va, te = m["val"], m["test"]

    pred = pd.read_parquet(RESULTS / "grid" / "predictions_y30.parquet")
    pv = pred[pred.split == "val"]; pt = pred[pred.split == "test"].reset_index(drop=True)
    ght = grid[te].reset_index(drop=True)
    yv = pv.y.to_numpy(); yt = pt.y.to_numpy(); wtt = pt.window.to_numpy()
    lam_h = lam(pt.hybrid.to_numpy()); lam_c = lam(pt.cascade.to_numpy())
    s_h = yv.sum() / lam(pv.hybrid.to_numpy()).sum(); s_c = yv.sum() / lam(pv.cascade.to_numpy()).sum()
    ph = np.clip(1 - np.exp(-s_h * lam_h), EPS, 1 - EPS)
    lam_2s = s_bg * lam_bg[te] + s_tr * lam_trig[te]
    p2s = np.clip(1 - np.exp(-lam_2s), EPS, 1 - EPS)

    out = {}
    # (1) PAIRED hybrid vs two-scalar
    ig, ci = paired_bern_ig(wtt, yt, ph, p2s)
    out["paired_hybrid_vs_two_scalar"] = {"dIG": ig, "ci": ci,
        "read": ("excludes 0 (hybrid favour) -> irreducible dynamic remainder resolved; "
                 "straddles -> two constants inseparable from ML, point-share ~half")}
    # (2) gem
    nest = bgf * s_bg + (1 - bgf) * s_tr
    out["gem_nesting_lock"] = {"rate_weighted_mean_2scalar": round(float(nest), 4),
                              "single_scalar_optimum": 0.423, "s_bg": s_bg, "s_trig": s_tr,
                              "saturation_signature": bool(s_tr < s_bg)}
    # (3) recombination
    days45 = ght["days_since_m45_25km"].to_numpy(); active = days45 < 365.0
    na, nq = int(yt[active].sum()), int(yt[~active].sum())
    iga = 0.079; igq = 0.117
    recomb = (iga * na + igq * nq) / (na + nq)
    out["recombination"] = {"active_pos": na, "quiet_pos": nq, "active_pos_frac": round(na / (na + nq), 3),
                            "weighted_IG": round(float(recomb), 4), "target_full_edge": 0.099}
    # (4) per-window concentration of active-cell edge, pre/post Kumburgaz
    SPLIT = pd.Timestamp("2025-04-23")
    t0 = pd.to_datetime(pt.t0.to_numpy())
    pre = np.asarray(t0 < SPLIT)
    amask = active
    def sub_ig(mask):
        if yt[mask].sum() == 0:
            return None
        return paired_bern_ig(wtt[mask], yt[mask], ph[mask], np.clip(1 - np.exp(-s_c * lam_c[mask]), EPS, 1 - EPS))
    ig_pre = sub_ig(amask & pre); ig_post = sub_ig(amask & ~pre)
    out["active_edge_by_period"] = {
        "pre_Kumburgaz": {"n_pos": int(yt[amask & pre].sum()), "dIG": ig_pre[0] if ig_pre else None, "ci": ig_pre[1] if ig_pre else None},
        "post_Kumburgaz": {"n_pos": int(yt[amask & ~pre].sum()), "dIG": ig_post[0] if ig_post else None, "ci": ig_post[1] if ig_post else None}}

    (RESULTS / "round3" / "t9_final_checks.json").write_text(json.dumps(out, indent=2))
    print("(1) PAIRED hybrid vs two-scalar dIG:", ig, ci)
    print("(2) nesting lock: 2-scalar rate-weighted mean", round(nest, 4), "vs single scalar 0.423; s_trig<s_bg:", s_tr < s_bg)
    print("(3) recombination: active_pos", na, "quiet_pos", nq, "frac", round(na/(na+nq), 3), "weighted IG", round(recomb, 4), "(target ~0.099)")
    print("(4) active edge pre-Kumburgaz:", out["active_edge_by_period"]["pre_Kumburgaz"], "post:", out["active_edge_by_period"]["post_Kumburgaz"])


if __name__ == "__main__":
    main()
