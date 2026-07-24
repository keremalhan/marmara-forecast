"""K=5000 Jensen debias + top-up thinning tie-break, native Bernoulli, ranking on INTENSITY PR.

Re-sim cascade at K=5000 on the 26 test windows; native occupancy at K in {500,1000,2000,5000} by
sub-sampling sim ids; extrapolate native Bernoulli IG in 1/K to K->inf (debias). Hybrid occupancy two
ways: capped thinning (t5) and thinning+Poisson-top-up (tie-break). Ranking = intensity PR-AUC
(construction-free). Writes results/round3/t6_debias.json.
"""
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from marmara.paths import RESULTS
from marmara import grid as G
from marmara.train import split_masks
from marmara.cascade import cascade_forecast
from marmara.metrics import p_to_lambda
from marmara.bootstrap import stationary_window_indices, MEAN_BLOCK, SEED

EPS = 1e-9
KMAX = 5000
KS = [500, 1000, 2000, 5000]
B_OP = 1.15


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
    m = split_masks(grid); win_t0 = grid.groupby("window")["t0"].first()
    params = pickle.load(open(RESULTS / "etas" / "etas_params.pkl", "rb"))
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    hist = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]; spec = G.MODEL_SPEC

    gt = grid[m["test"]].sort_values(["window", "ir", "ic"]).reset_index(drop=True)
    keyi = {(int(r.window), int(r.ir), int(r.ic)): i for i, r in gt.iterrows()}
    n = len(gt)
    pred = pd.read_parquet(RESULTS / "grid" / "predictions_y30.parquet")
    pte = pred[pred.split == "test"].reset_index(drop=True)
    gh_te = grid[m["test"]].reset_index(drop=True)
    tmp = gh_te.assign(hyb=pte.hybrid.to_numpy(), fg=pte.firstgen_etas.to_numpy(), inv=pte.modern_etas.to_numpy()
                       ).sort_values(["window", "ir", "ic"]).reset_index(drop=True)
    lam_hyb = lam(tmp.hyb.to_numpy()); y = tmp.y30.to_numpy(float); wt = tmp.window.to_numpy()
    P_fg = np.clip(tmp.fg.to_numpy(), EPS, 1 - EPS); P_inv = np.clip(tmp.inv.to_numpy(), EPS, 1 - EPS)
    pr_hyb_int = pte.hybrid.to_numpy(); pr_casc_int = pte.cascade.to_numpy()  # intensity ranking

    occ_c = {K: np.zeros(n) for K in KS}
    occ_hc = {K: np.zeros(n) for K in KS}   # hybrid capped-thin
    occ_ht = {K: np.zeros(n) for K in KS}   # hybrid top-up
    for w in [int(x) for x in np.sort(gt.window.unique())]:
        t0_dt = pd.Timestamp(win_t0.loc[w]); t0d = float(G._to_days(t0_dt))
        ev = cascade_forecast(params, hist[cat["datetime_utc"] < t0_dt], t0d, G.HORIZON_D,
                              spec.lon_c, spec.lat_c, K=KMAX, seed=1000 + w, b=B_OP,
                              preserve_branching=True, return_events=True)
        sid = ev["sim"]
        ic = np.floor((ev["lon"] - round(float(spec.lon_c[0]) - 0.05, 2)) / 0.1).astype(int)
        ir = np.floor((ev["lat"] - round(float(spec.lat_c[0]) - 0.05, 2)) / 0.1).astype(int)
        cflat = ir * 100000 + ic
        order = np.argsort(cflat, kind="stable")
        sid, cflat = sid[order], cflat[order]
        bounds = np.searchsorted(cflat, np.unique(cflat))
        uniq = np.unique(cflat)
        for bi, cf in enumerate(uniq):
            r_ = int(cf // 100000); c_ = int(cf % 100000)
            i = keyi.get((w, r_, c_))
            if i is None:
                continue
            lo = bounds[bi]; hi = bounds[bi + 1] if bi + 1 < len(bounds) else len(cflat)
            s_cell = sid[lo:hi]
            us, cts = np.unique(s_cell, return_counts=True)
            lam_c = len(s_cell) / KMAX
            lh = lam_hyb[i]; ratio = min(1.0, lh / lam_c) if lam_c > 0 else 0.0
            dlam = max(lh - lam_c, 0.0)
            for K in KS:
                mk = us < K
                if not mk.any():
                    continue
                cnt = cts[mk]
                occ_c[K][i] = mk.sum() / K
                # capped thin: sim keeps >=1 w.p. 1-(1-ratio)^count
                thin = (1 - (1 - ratio) ** cnt).sum() / K
                occ_hc[K][i] = thin
                # top-up: if boosted, add Poisson(dlam); occ = casc_hits + non-hit sims * (1-e^{-dlam})
                if dlam > 0:
                    hits = mk.sum()
                    occ_ht[K][i] = (hits + (K - hits) * (1 - np.exp(-dlam))) / K
                else:
                    occ_ht[K][i] = thin
        print(f"  window {w} done", flush=True)

    # native Bernoulli IG at each K, extrapolate 1/K -> 0
    def igs_over_K(occ_a, pb, is_a_dict):
        pts = []
        for K in KS:
            pa = (occ_a[K] * K + 1) / (K + 2) if is_a_dict else occ_a
            ig, ci = bern_ig_ci(wt, y, pa, pb)
            pts.append((1.0 / K, ig, ci))
        # linear extrapolation in 1/K to 0
        xs = np.array([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
        slope, inter = np.polyfit(xs, ys, 1)
        return {"per_K": {f"K{int(1/p[0])}": {"ig": p[1], "ci": p[2]} for p in pts},
                "debiased_Kinf": round(float(inter), 4)}

    Pfg = P_fg; Pinv = P_inv
    def pc(K):
        return (occ_c[K] * K + 1) / (K + 2)
    def phc(K):
        return (occ_hc[K] * K + 1) / (K + 2)
    def pht(K):
        return (occ_ht[K] * K + 1) / (K + 2)

    out = {"scoring": "native Bernoulli, K-extrapolated (debias); ranking=intensity PR", "K_levels": KS}
    # hybrid vs cascade (both MC): capped and top-up
    hc = {}
    for K in KS:
        ig, ci = bern_ig_ci(wt, y, phc(K), pc(K)); hc[f"K{K}"] = {"ig": ig, "ci": ci}
    xs = np.array([1.0 / K for K in KS]); ys = np.array([hc[f"K{K}"]["ig"] for K in KS])
    out["hybrid_vs_cascade_capped"] = {"per_K": hc, "debiased_Kinf": round(float(np.polyfit(xs, ys, 1)[1]), 4)}
    ht = {}
    for K in KS:
        ig, ci = bern_ig_ci(wt, y, pht(K), pc(K)); ht[f"K{K}"] = {"ig": ig, "ci": ci}
    ys2 = np.array([ht[f"K{K}"]["ig"] for K in KS])
    out["hybrid_vs_cascade_topup"] = {"per_K": ht, "debiased_Kinf": round(float(np.polyfit(xs, ys2, 1)[1]), 4)}
    # cascade vs first-gen / inversion (cascade MC, other analytic) -- the bias asymmetry check
    for nm, pb in (("cascade_vs_firstgen", Pfg), ("cascade_vs_inversion", Pinv)):
        rec = {}
        for K in KS:
            ig, ci = bern_ig_ci(wt, y, pc(K), pb); rec[f"K{K}"] = {"ig": ig, "ci": ci}
        ysx = np.array([rec[f"K{K}"]["ig"] for K in KS])
        out[nm] = {"per_K": rec, "debiased_Kinf": round(float(np.polyfit(xs, ysx, 1)[1]), 4)}

    # intensity-PR verdict for hybrid vs cascade (construction-free)
    dpr = float(average_precision_score(y, pr_hyb_int) - average_precision_score(y, pr_casc_int))
    wins = np.sort(np.unique(wt)); idx = {w: np.where(wt == w)[0] for w in wins}
    yb = [y[idx[w]] for w in wins]; ab = [pr_hyb_int[idx[w]] for w in wins]; bb = [pr_casc_int[idx[w]] for w in wins]
    rng = np.random.default_rng(SEED); seqs = stationary_window_indices(len(wins), 2000, MEAN_BLOCK, rng)
    dprb = []
    for r in seqs:
        yy = np.concatenate([yb[i] for i in r])
        if 0 < yy.sum() < len(yy):
            dprb.append(average_precision_score(yy, np.concatenate([ab[i] for i in r])) - average_precision_score(yy, np.concatenate([bb[i] for i in r])))
    out["hybrid_vs_cascade_intensity_PR"] = {"dpr": round(dpr, 4),
        "dpr_ci": [round(float(np.percentile(dprb, 2.5)), 4), round(float(np.percentile(dprb, 97.5)), 4)]}
    # occupancy dPR tie-break: capped vs top-up (K=5000)
    def occ_dpr(pa):
        d = float(average_precision_score(y, pa) - average_precision_score(y, pc(5000)))
        return round(d, 4)
    out["occupancy_dpr_K5000"] = {"capped": occ_dpr(phc(5000)), "topup": occ_dpr(pht(5000))}
    out["totals_test"] = {"casc": round(float((occ_c[5000]).sum()), 1), "hyb_capped": round(float(occ_hc[5000].sum()), 1),
                          "hyb_topup": round(float(occ_ht[5000].sum()), 1), "n_pos": int(y.sum())}
    (RESULTS / "round3" / "t6_debias.json").write_text(json.dumps(out, indent=2))
    print("\n=== DEBIASED (K->inf) native Bernoulli IG ===")
    print("hybrid vs cascade (capped):", out["hybrid_vs_cascade_capped"]["debiased_Kinf"],
          "per-K", {k: v["ig"] for k, v in out["hybrid_vs_cascade_capped"]["per_K"].items()})
    print("hybrid vs cascade (topup) :", out["hybrid_vs_cascade_topup"]["debiased_Kinf"])
    print("cascade vs firstgen       :", out["cascade_vs_firstgen"]["debiased_Kinf"],
          "per-K", {k: v["ig"] for k, v in out["cascade_vs_firstgen"]["per_K"].items()})
    print("cascade vs inversion      :", out["cascade_vs_inversion"]["debiased_Kinf"])
    print("intensity-PR dpr (hyb-casc):", out["hybrid_vs_cascade_intensity_PR"])
    print("occupancy dpr K5000 capped vs topup:", out["occupancy_dpr_K5000"])
    print("totals:", out["totals_test"])


if __name__ == "__main__":
    main()
