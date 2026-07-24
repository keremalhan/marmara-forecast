"""Occupancy re-scoring (global-s approximation) + paired scalar-vs-hybrid + isotonic clarity.

Remove the count-handicap: occurrence-calibrate EVERY model by its own val-fit scalar
s_m = N_pos_val / sum(lambda_m, val), i.e. score lambda_occ = s_m * lambda_m (equivalently
P_occ = 1 - exp(-s_m*lambda)). A global rescale is monotone -> PR-AUC unchanged; only the IG
count-term is removed. Re-check the primary verdicts under occurrence scoring vs the original
count scoring. Predictions to test: (a) hybrid's IG edge over cascade vanishes; (b) whether the
cascade separates from first-gen once its 2.2x handicap is removed.

Also: paired scalar-vs-hybrid ΔIG interval (same bootstrap), and a clean isotonic (val in-sample
vs test, to show whether the monotone map ~ the single scalar).
Writes results/round3/t2_occupancy.json.
"""
import json
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score

from marmara.paths import RESULTS
from marmara.metrics import p_to_lambda, lambda_to_p
from marmara.bootstrap import stationary_window_indices, MEAN_BLOCK, SEED

EPS = 1e-9
B = 2000
CLUSTERED = {"hybrid", "cascade", "sv_etas"}
PAIRS = [("hybrid", "cascade"), ("cascade", "firstgen_etas"), ("cascade", "sv_etas"),
         ("firstgen_etas", "modern_etas"), ("hybrid", "firstgen_etas")]


def lam(p):
    return np.clip(p_to_lambda(np.clip(np.asarray(p, float), 0.0, 1.0)), EPS, None)


def two_axis(win, y, la, lb, pa, pb):
    wins = np.sort(np.unique(win)); idx = {w: np.where(win == w)[0] for w in wins}
    ca = np.array([np.sum(y[idx[w]] * np.log(la[idx[w]]) - la[idx[w]]) for w in wins])
    cb = np.array([np.sum(y[idx[w]] * np.log(lb[idx[w]]) - lb[idx[w]]) for w in wins])
    pos = np.array([y[idx[w]].sum() for w in wins])
    ig = (ca.sum() - cb.sum()) / max(pos.sum(), 1)
    pr = average_precision_score(y, pa) - average_precision_score(y, pb)
    rng = np.random.default_rng(SEED); seqs = stationary_window_indices(len(wins), B, MEAN_BLOCK, rng)
    yb = [y[idx[w]] for w in wins]; ab = [pa[idx[w]] for w in wins]; bb = [pb[idx[w]] for w in wins]
    igbs, prbs = [], []
    for r in seqs:
        igbs.append((ca[r].sum() - cb[r].sum()) / max(pos[r].sum(), 1))
        yy = np.concatenate([yb[i] for i in r])
        if 0 < yy.sum() < len(yy):
            prbs.append(average_precision_score(yy, np.concatenate([ab[i] for i in r]))
                        - average_precision_score(yy, np.concatenate([bb[i] for i in r])))
    ig_ci = [round(float(np.percentile(igbs, 2.5)), 4), round(float(np.percentile(igbs, 97.5)), 4)]
    pr_ci = [round(float(np.percentile(prbs, 2.5)), 4), round(float(np.percentile(prbs, 97.5)), 4)]
    sep = (ig_ci[0] > 0 or ig_ci[1] < 0) and (pr_ci[0] > 0 or pr_ci[1] < 0)
    return {"ig": round(float(ig), 4), "ig_ci": ig_ci, "dpr": round(float(pr), 4), "dpr_ci": pr_ci,
            "verdict": "separable" if sep else "inseparable"}


def main():
    df = pd.read_parquet(RESULTS / "grid" / "predictions_y30.parquet")
    va = df[df.split == "val"]; te = df[df.split == "test"]
    yv = va.y.to_numpy(); yt = te.y.to_numpy(); wt = te.window.to_numpy()
    npos_v = float(yv.sum())
    models = ["hybrid", "cascade", "sv_etas", "firstgen_etas", "modern_etas", "poisson", "smoothed"]
    s = {m: npos_v / lam(va[m].to_numpy()).sum() for m in models}   # val-fit occurrence scalar (ALL models)
    Lt = {m: lam(te[m].to_numpy()) for m in models}
    Locc = {m: s[m] * Lt[m] for m in models}                        # occurrence-calibrated rate
    Pocc = {m: lambda_to_p(Locc[m]) for m in models}
    Pcnt = {m: te[m].to_numpy() for m in models}                    # original count-scored prob

    out = {"s_val": {m: round(s[m], 4) for m in models}, "count_scored": {}, "occurrence_scored": {}}
    for a, b in PAIRS:
        out["count_scored"][f"{a}_vs_{b}"] = two_axis(wt, yt, Lt[a], Lt[b], Pcnt[a], Pcnt[b])
        out["occurrence_scored"][f"{a}_vs_{b}"] = two_axis(wt, yt, Locc[a], Locc[b], Pocc[a], Pocc[b])

    # paired scalar(cascade)-vs-hybrid: scalar-rescaled cascade vs the ML hybrid (count-scored world)
    L_scal = s["cascade"] * Lt["cascade"]; P_scal = lambda_to_p(L_scal)
    out["paired_scalar_vs_hybrid"] = two_axis(wt, yt, L_scal, Lt["hybrid"], P_scal, Pcnt["hybrid"])

    # clean isotonic: monotone g(lambda_sim) on val; in-sample (val) vs out-of-sample (test)
    lc_v = lam(va["cascade"].to_numpy()); lc_t = lam(te["cascade"].to_numpy())
    iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(lc_v, yv)
    p_iso_v = np.clip(iso.predict(lc_v), 0, 1 - EPS); p_iso_t = np.clip(iso.predict(lc_t), 0, 1 - EPS)
    def ig_simple(y, la, lb):
        return float((np.sum(y * np.log(la) - la) - np.sum(y * np.log(lb) - lb)) / max(y.sum(), 1))
    out["isotonic"] = {
        "val_in_sample_ig_vs_cascade": round(ig_simple(yv, lam(p_iso_v), lc_v), 4),
        "test_ig_vs_cascade": round(ig_simple(yt, lam(p_iso_t), lc_t), 4),
        "test_vs_hybrid": two_axis(wt, yt, lam(p_iso_t), Lt["hybrid"], p_iso_t, Pcnt["hybrid"]),
        "note": "if val>>test, the monotone map overfits and the single scalar (robust) is the supported framing"}

    (RESULTS / "round3" / "t2_occupancy.json").write_text(json.dumps(out, indent=2))
    print("=== verdicts: COUNT-scored (current paper) -> OCCURRENCE-scored (handicap removed) ===")
    for a, b in PAIRS:
        c = out["count_scored"][f"{a}_vs_{b}"]; o = out["occurrence_scored"][f"{a}_vs_{b}"]
        print(f"  {a} vs {b:14s}: IG {c['ig']:+.3f}{c['ig_ci']} [{c['verdict']}]  ->  "
              f"IG {o['ig']:+.3f}{o['ig_ci']} [{o['verdict']}]")
    p = out["paired_scalar_vs_hybrid"]
    print(f"\nPAIRED scalar(cascade) vs hybrid: ΔIG {p['ig']:+.3f} {p['ig_ci']}  dPR {p['dpr']:+.4f} {p['dpr_ci']}  [{p['verdict']}]")
    i = out["isotonic"]
    print(f"ISOTONIC: val in-sample IG {i['val_in_sample_ig_vs_cascade']:+.3f}  test IG {i['test_ig_vs_cascade']:+.3f}  "
          f"(scalar test IG was +0.415)")


if __name__ == "__main__":
    main()
