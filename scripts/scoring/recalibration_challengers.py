"""Tier-1 headline tests: is the hybrid's IG edge over the cascade a scalar / a monotone map?

T1-1 scalar challenger : global s (MLE) and a 2-scalar {low,high} split by lambda_sim median,
                         fit on VALIDATION Poisson-LL, scored on TEST.
T1-2 isotonic          : monotone g(lambda_sim) fit on validation, scored on test. A monotone
                         map cannot change ranking, so PR-AUC is unchanged by construction; if it
                         recovers the hybrid's IG, the ML edge IS a re-calibration (constructive proof).
T1-10a per-window dIG  : is the +0.29 spread across the 26 test windows or carried by a few?
T1-10b val decomposition: run the count/placement split on VALIDATION too.

All comparisons carry the paper's two-axis block-bootstrap CI (B=2000, seed 42, mean block 3).
Writes results/round3/t1_recalibration.json. Reads canonical predictions; writes nothing canonical.
"""
from __future__ import annotations

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


def lam(p):
    return np.clip(p_to_lambda(np.clip(np.asarray(p, float), 0.0, 1.0)), EPS, None)


def two_axis_ci(win, y, pa, pb):
    """point + 95% block-bootstrap CI for IG(a vs b) [per event] and dPR-AUC(a-b)."""
    la, lb = lam(pa), lam(pb)
    wins = np.sort(np.unique(win))
    idx = {w: np.where(win == w)[0] for w in wins}
    llc_a = np.array([np.sum(y[idx[w]] * np.log(la[idx[w]]) - la[idx[w]]) for w in wins])
    llc_b = np.array([np.sum(y[idx[w]] * np.log(lb[idx[w]]) - lb[idx[w]]) for w in wins])
    pos = np.array([y[idx[w]].sum() for w in wins])
    ig_pt = (llc_a.sum() - llc_b.sum()) / max(pos.sum(), 1)
    pr_a, pr_b = average_precision_score(y, pa), average_precision_score(y, pb)
    rng = np.random.default_rng(SEED)
    seqs = stationary_window_indices(len(wins), B, MEAN_BLOCK, rng)
    ig_bs, dpr_bs = [], []
    yb = [y[idx[w]] for w in wins]; pab = [pa[idx[w]] for w in wins]; pbb = [pb[idx[w]] for w in wins]
    for row in seqs:
        d = llc_a[row].sum() - llc_b[row].sum()
        n = max(pos[row].sum(), 1)
        ig_bs.append(d / n)
        yy = np.concatenate([yb[i] for i in row])
        if 0 < yy.sum() < len(yy):
            dpr_bs.append(average_precision_score(yy, np.concatenate([pab[i] for i in row]))
                          - average_precision_score(yy, np.concatenate([pbb[i] for i in row])))
    ig_ci = [float(np.percentile(ig_bs, 2.5)), float(np.percentile(ig_bs, 97.5))]
    dpr_ci = [float(np.percentile(dpr_bs, 2.5)), float(np.percentile(dpr_bs, 97.5))]
    sep = (ig_ci[0] > 0 or ig_ci[1] < 0) and (dpr_ci[0] > 0 or dpr_ci[1] < 0)
    return {"ig": round(float(ig_pt), 4), "ig_ci": [round(x, 4) for x in ig_ci],
            "pr_a": round(float(pr_a), 4), "pr_b": round(float(pr_b), 4),
            "dpr": round(float(pr_a - pr_b), 4), "dpr_ci": [round(x, 4) for x in dpr_ci],
            "separable_two_axis": bool(sep)}


def run_target(target):
    df = pd.read_parquet(RESULTS / f"predictions_{target}.parquet")
    tr = df[df.split == "val"]; te = df[df.split == "test"]
    lc_v = lam(tr.cascade.to_numpy()); yv = tr.y.to_numpy()
    lc_t = lam(te.cascade.to_numpy()); yt = te.y.to_numpy(); wt = te.window.to_numpy()

    # ---- T1-1 scalar: global s (MLE = Npos/sum-lambda on val) ----
    s_glob = yv.sum() / lc_v.sum()
    p_glob = lambda_to_p(s_glob * lc_t)
    # 2-scalar: split by val median lambda
    thr = np.median(lc_v)
    lo_v, lo_t = lc_v < thr, lc_t < thr
    s_lo = yv[lo_v].sum() / max(lc_v[lo_v].sum(), EPS)
    s_hi = yv[~lo_v].sum() / max(lc_v[~lo_v].sum(), EPS)
    lam2 = np.where(lo_t, s_lo * lc_t, s_hi * lc_t)
    p_2s = lambda_to_p(lam2)

    # ---- T1-2 isotonic g(lambda_sim) on val ----
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(lc_v, yv)                       # monotone E[y | lambda_sim]
    p_iso = np.clip(iso.predict(lc_t), 0.0, 1.0 - EPS)

    hyb = te.hybrid.to_numpy(); casc = te.cascade.to_numpy()
    res = {
        "n_test": int(len(te)), "n_pos": int(yt.sum()),
        "hybrid_vs_cascade": two_axis_ci(wt, yt, hyb, casc),
        "scalar_global_vs_cascade": {"s": round(float(s_glob), 4), **two_axis_ci(wt, yt, p_glob, casc)},
        "scalar_2_vs_cascade": {"s_low": round(float(s_lo), 4), "s_high": round(float(s_hi), 4),
                                **two_axis_ci(wt, yt, p_2s, casc)},
        "isotonic_vs_cascade": two_axis_ci(wt, yt, p_iso, casc),
        "isotonic_vs_hybrid": two_axis_ci(wt, yt, p_iso, hyb),
    }
    # fraction of the hybrid's IG edge recovered by each simpler challenger
    ig_h = res["hybrid_vs_cascade"]["ig"]
    for k in ("scalar_global_vs_cascade", "scalar_2_vs_cascade", "isotonic_vs_cascade"):
        res[k]["frac_of_hybrid_edge"] = round(res[k]["ig"] / ig_h, 3) if ig_h else None

    # ---- T1-10a per-window dIG(hybrid vs cascade) ----
    lh, lc = lam(hyb), lam(casc)
    perwin = {}
    for w in np.sort(np.unique(wt)):
        m = wt == w
        perwin[int(w)] = round(float(np.sum(yt[m] * (np.log(lh[m]) - np.log(lc[m])) - (lh[m] - lc[m]))), 2)
    res["per_window_dll_hybrid_vs_cascade"] = perwin
    vals = np.array(list(perwin.values()))
    res["per_window_summary"] = {"n_windows": len(vals), "total": round(float(vals.sum()), 1),
                                 "n_positive": int((vals > 0).sum()), "n_negative": int((vals < 0).sum()),
                                 "max_window": round(float(vals.max()), 1), "min_window": round(float(vals.min()), 1),
                                 "top_window_share_pct": round(float(vals.max() / vals.sum() * 100), 1) if vals.sum() else None}

    # ---- T1-10b val count/placement decomposition (hybrid vs cascade) ----
    lh_v, lc_v2 = lam(tr.hybrid.to_numpy()), lam(tr.cascade.to_numpy())
    place = float(np.sum(yv * (np.log(lh_v) - np.log(lc_v2))))
    count = float(np.sum(-(lh_v - lc_v2)))
    tot = place + count
    res["val_decomposition_hybrid_vs_cascade"] = {
        "total_ll": round(tot, 2), "ig_per_event": round(tot / max(yv.sum(), 1), 4),
        "placement_nats": round(place, 2), "count_nats": round(count, 2),
        "placement_share_pct": round(100 * place / tot, 1) if tot else None,
        "count_share_pct": round(100 * count / tot, 1) if tot else None}
    return res


def main():
    out = {t: run_target(t) for t in ("y30", "y35")}
    (RESULTS / "round3" / "t1_recalibration.json").write_text(json.dumps(out, indent=2))
    for t, r in out.items():
        print(f"\n=== {t} (test n_pos={r['n_pos']}) ===")
        h = r["hybrid_vs_cascade"]
        print(f"hybrid vs cascade: IG {h['ig']} {h['ig_ci']}  dPR {h['dpr']} {h['dpr_ci']}  sep2={h['separable_two_axis']}")
        for k in ("scalar_global_vs_cascade", "scalar_2_vs_cascade", "isotonic_vs_cascade"):
            x = r[k]
            print(f"  {k:28s}: IG {x['ig']} {x['ig_ci']}  dPR {x['dpr']} {x['dpr_ci']}  "
                  f"frac_edge={x['frac_of_hybrid_edge']}  sep2={x['separable_two_axis']}")
        ih = r["isotonic_vs_hybrid"]
        print(f"  isotonic vs HYBRID: IG {ih['ig']} {ih['ig_ci']} (near 0 => isotonic == hybrid)")
        pw = r["per_window_summary"]
        print(f"  per-window dIG: total {pw['total']} over {pw['n_windows']} win; "
              f"{pw['n_positive']} pos / {pw['n_negative']} neg; top-window share {pw['top_window_share_pct']}%")
        vd = r["val_decomposition_hybrid_vs_cascade"]
        print(f"  VAL decomposition: placement {vd['placement_share_pct']}%  count {vd['count_share_pct']}%")


if __name__ == "__main__":
    main()
