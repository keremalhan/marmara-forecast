"""Closing lock: (1) exact identity numbers with provenance; (2) occurrence-scored verdicts vs the
inversion (the conclusion-moving rows: cascade/sv/hybrid vs modern_etas); (3) Bernoulli-on-p vs
Poisson-on-indicator agreement. b_op = 1.15 from canonical predictions. Writes t3_lock.json.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from marmara.paths import RESULTS
from marmara.metrics import p_to_lambda, lambda_to_p
from marmara.bootstrap import stationary_window_indices, MEAN_BLOCK, SEED

EPS = 1e-9
B = 2000


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


def events_in(cat, t0s):
    lo, hi = t0s.min(), t0s.max() + pd.Timedelta(days=30)
    return int(len(cat[(cat.datetime_utc >= lo) & (cat.datetime_utc < hi) & (cat.mag_w >= 3.0 - 1e-9)]))


def main():
    df = pd.read_parquet(RESULTS / "grid" / "predictions_y30.parquet")
    va = df[df.split == "val"]; te = df[df.split == "test"]
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])

    # ---------- (1) identity, with provenance ----------
    npos_t = float(te.y.sum()); npos_v = float(va.y.sum())
    Lc_t = lam(te.cascade.to_numpy()); Lc_v = lam(va.cascade.to_numpy())
    sumL_t = float(Lc_t.sum()); sumL_v = float(Lc_v.sum())
    nev_t = events_in(cat, pd.to_datetime(te.t0.unique())); nev_v = events_in(cat, pd.to_datetime(va.t0.unique()))
    s_test = npos_t / sumL_t; s_val = npos_v / sumL_v
    inv_mult_t = npos_t / nev_t; ccal_t = nev_t / sumL_t

    def closed(s):
        return float(np.log(s) + (1 - s) * sumL_t / npos_t)
    identity = {
        "N_pos_test": int(npos_t), "N_events_test": nev_t, "sumL_cascade_test": round(sumL_t, 1),
        "s_star_test = Npos/sumL": round(s_test, 4),
        "inverse_multiplicity_test = Npos/Nev": round(inv_mult_t, 4),
        "count_calibration_ratio_test = Nev/sumL": round(ccal_t, 4),
        "decomposition s* = inv_mult * count_calib": round(inv_mult_t * ccal_t, 4),
        "handicap_col = sumL/Npos": round(sumL_t / npos_t, 4),
        "PROVENANCE_s_is_validation_fit": round(s_val, 4),
        "validation_multiplicity = Nev_val/Npos_val": round(nev_v / npos_v, 4),
        "N_pos_val": int(npos_v), "N_events_val": nev_v, "sumL_cascade_val": round(sumL_v, 1),
        "closed_form_IG": {"at_test_optimal_0.453": round(closed(s_test), 4),
                           "at_val_fit_0.4227": round(closed(s_val), 4),
                           "measured_val_scalar_on_test": 0.4154},
    }

    # ---------- (2) occurrence-scored verdicts vs the inversion (conclusion-moving) ----------
    models = ["hybrid", "cascade", "sv_etas", "firstgen_etas", "modern_etas"]
    s = {m: npos_v / lam(va[m].to_numpy()).sum() for m in models}
    Lt = {m: lam(te[m].to_numpy()) for m in models}
    Locc = {m: s[m] * Lt[m] for m in models}
    Pocc = {m: lambda_to_p(Locc[m]) for m in models}
    Pcnt = {m: te[m].to_numpy() for m in models}
    wt = te.window.to_numpy(); yt = te.y.to_numpy()
    vs_inv = {}
    for a in ("cascade", "sv_etas", "hybrid", "firstgen_etas"):
        vs_inv[f"{a}_vs_inversion"] = {
            "count_scored": two_axis(wt, yt, Lt[a], Lt["modern_etas"], Pcnt[a], Pcnt["modern_etas"]),
            "occurrence_scored": two_axis(wt, yt, Locc[a], Locc["modern_etas"], Pocc[a], Pocc["modern_etas"])}

    # ---------- (3) Bernoulli-on-p vs Poisson-on-indicator (hybrid vs cascade, occurrence) ----------
    def ig_poisson(y, la, lb):
        return float((np.sum(y * np.log(la) - la) - np.sum(y * np.log(lb) - lb)) / max(y.sum(), 1))
    def ig_bernoulli(y, pa, pb):
        pa = np.clip(pa, EPS, 1 - EPS); pb = np.clip(pb, EPS, 1 - EPS)
        lla = np.sum(y * np.log(pa) + (1 - y) * np.log(1 - pa))
        llb = np.sum(y * np.log(pb) + (1 - y) * np.log(1 - pb))
        return float((lla - llb) / max(y.sum(), 1))
    score_def = {
        "hybrid_vs_cascade_occ_Poisson_indicator": round(ig_poisson(yt, Locc["hybrid"], Locc["cascade"]), 4),
        "hybrid_vs_cascade_occ_Bernoulli_on_p": round(ig_bernoulli(yt, Pocc["hybrid"], Pocc["cascade"]), 4),
        "mean_occupancy_test": round(float(yt.mean()), 4),
        "note": "agree to 2nd order at low occupancy; report which one is named in the paper"}

    out = {"identity": identity, "vs_inversion": vs_inv, "score_definition": score_def}
    (RESULTS / "round3" / "t3_lock.json").write_text(json.dumps(out, indent=2))
    print("IDENTITY: s*_test =", identity["s_star_test = Npos/sumL"], "= inv_mult",
          identity["inverse_multiplicity_test = Npos/Nev"], "x count_calib",
          identity["count_calibration_ratio_test = Nev/sumL"], "=",
          identity["decomposition s* = inv_mult * count_calib"])
    print("  closed-form IG:", identity["closed_form_IG"])
    print("  s provenance (val-fit):", identity["PROVENANCE_s_is_validation_fit"],
          "val multiplicity", identity["validation_multiplicity = Nev_val/Npos_val"])
    print("\nVS INVERSION (count -> occurrence):")
    for k, v in vs_inv.items():
        print(f"  {k:26s}: {v['count_scored']['ig']:+.3f} [{v['count_scored']['verdict']}]  ->  "
              f"{v['occurrence_scored']['ig']:+.3f} {v['occurrence_scored']['ig_ci']} "
              f"dPR {v['occurrence_scored']['dpr']:+.4f}{v['occurrence_scored']['dpr_ci']} [{v['occurrence_scored']['verdict']}]")
    print("\nSCORE DEF:", score_def)


if __name__ == "__main__":
    main()
