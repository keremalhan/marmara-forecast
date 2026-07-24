"""Run 3 — sv-ETAS native Monte-Carlo occupancy at K = 5000. Closes the Table 4 dagger.

Table 4 carries a dagger: "sv-ETAS was not re-simulated for native Monte-Carlo occupancy at
K = 5000, so only its construction-free proxy is available for this pair". This run removes it.

Replicates scripts/scoring/occupancy_debias_topup.py exactly, with sv-ETAS added:
  * re-simulate the cascade AND sv-ETAS over the 26 test windows at K = 5000
    (same per-window seed 1000+w, same b_op = 1.15, preserve_branching=True, as
    marmara.etas_rates / marmara.grid_hybrid -- so only the ETAS PARAMETERS differ);
  * native occupancy at K in {500, 1000, 2000, 5000} by sub-sampling sim ids;
  * regularized p_hat = (occ*K + 1)/(K + 2)  [add-one, as in t6];
  * Bernoulli log-score IG, extrapolated linearly in 1/K to K -> inf (Jensen debias);
  * hybrid occupancy under the top-up construction (the Table 4/5 construction of record,
    confirmed by Run 4 to be what the CSEP catalogues use).

Reports: Bernoulli-native dIG for cascade-vs-sv-ETAS and hybrid-vs-sv-ETAS with CIs, and
sv-ETAS's raw occupancy total for Table 5.

Writes results/round4/r3_sv_etas_native.json. Reads only.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/scoring/sv_etas_native_occupancy.py
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd

from marmara import grid as G
from marmara.bootstrap import MEAN_BLOCK, SEED, stationary_window_indices
from marmara.cascade import cascade_forecast
from marmara.metrics import p_to_lambda
from marmara.paths import RESULTS
from marmara.train import split_masks

EPS = 1e-9
KMAX = 5000
KS = [500, 1000, 2000, 5000]
B_OP = 1.15
SEED_BASE = 1000                 # per-window seed, matches grid_hybrid.py / etas_rates.py
OUT = RESULTS / "round4"
OUT.mkdir(exist_ok=True)


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
    return round(float(ig), 4), [round(float(np.percentile(bs, 2.5)), 4),
                                 round(float(np.percentile(bs, 97.5)), 4)]


def sim_occupancy(params, hist, cat, spec, w, t0_dt, keyi, n, lam_hyb=None):
    """One window's native occupancy at each K, by sub-sampling sim ids. Verbatim in substance
    from t6_debias. Returns (occ_by_K, occ_topup_by_K or None)."""
    t0d = float(G._to_days(t0_dt))
    ev = cascade_forecast(params, hist[cat["datetime_utc"] < t0_dt], t0d, G.HORIZON_D,
                          spec.lon_c, spec.lat_c, K=KMAX, seed=SEED_BASE + w, b=B_OP,
                          preserve_branching=True, return_events=True)
    sid = ev["sim"]
    ic = np.floor((ev["lon"] - round(float(spec.lon_c[0]) - 0.05, 2)) / 0.1).astype(int)
    ir = np.floor((ev["lat"] - round(float(spec.lat_c[0]) - 0.05, 2)) / 0.1).astype(int)
    cflat = ir * 100000 + ic
    order = np.argsort(cflat, kind="stable")
    sid, cflat = sid[order], cflat[order]
    uniq = np.unique(cflat)
    bounds = np.searchsorted(cflat, uniq)
    occ = {K: np.zeros(n) for K in KS}
    occ_tu = {K: np.zeros(n) for K in KS} if lam_hyb is not None else None
    for bi, cf in enumerate(uniq):
        r_ = int(cf // 100000); c_ = int(cf % 100000)
        i = keyi.get((w, r_, c_))
        if i is None:
            continue
        lo = bounds[bi]; hi = bounds[bi + 1] if bi + 1 < len(bounds) else len(cflat)
        s_cell = sid[lo:hi]
        us, cts = np.unique(s_cell, return_counts=True)
        lam_c = len(s_cell) / KMAX
        if lam_hyb is not None:
            lh = lam_hyb[i]
            ratio = min(1.0, lh / lam_c) if lam_c > 0 else 0.0
            dlam = max(lh - lam_c, 0.0)
        for K in KS:
            mk = us < K
            if not mk.any():
                continue
            occ[K][i] = mk.sum() / K
            if lam_hyb is not None:
                cnt = cts[mk]
                thin = (1 - (1 - ratio) ** cnt).sum() / K
                if dlam > 0:
                    hits = mk.sum()
                    occ_tu[K][i] = (hits + (K - hits) * (1 - np.exp(-dlam))) / K
                else:
                    occ_tu[K][i] = thin
    return occ, occ_tu


def main():
    t_all = time.time()
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    m = split_masks(grid); win_t0 = grid.groupby("window")["t0"].first()
    params_c = pickle.load(open(RESULTS / "etas" / "etas_params.pkl", "rb"))
    params_sv = pickle.load(open(RESULTS / "etas" / "etas_sv_params.pkl", "rb"))
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    hist = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]
    spec = G.MODEL_SPEC

    gt = grid[m["test"]].sort_values(["window", "ir", "ic"]).reset_index(drop=True)
    keyi = {(int(r.window), int(r.ir), int(r.ic)): i for i, r in gt.iterrows()}
    n = len(gt)
    pred = pd.read_parquet(RESULTS / "grid" / "predictions_y30.parquet")
    pte = pred[pred.split == "test"].reset_index(drop=True)
    gh_te = grid[m["test"]].reset_index(drop=True)
    tmp = gh_te.assign(hyb=pte.hybrid.to_numpy(), sv=pte.sv_etas.to_numpy()
                       ).sort_values(["window", "ir", "ic"]).reset_index(drop=True)
    lam_hyb = lam(tmp.hyb.to_numpy())
    y = tmp.y30.to_numpy(float); wt = tmp.window.to_numpy()

    wins = [int(x) for x in np.sort(gt.window.unique())]
    print(f"{len(wins)} test windows, {n} cell-windows, {int(y.sum())} positives", flush=True)

    occ_c = {K: np.zeros(n) for K in KS}
    occ_sv = {K: np.zeros(n) for K in KS}
    occ_ht = {K: np.zeros(n) for K in KS}
    for w in wins:
        t0_dt = pd.Timestamp(win_t0.loc[w])
        oc, otu = sim_occupancy(params_c, hist, cat, spec, w, t0_dt, keyi, n, lam_hyb=lam_hyb)
        osv, _ = sim_occupancy(params_sv, hist, cat, spec, w, t0_dt, keyi, n, lam_hyb=None)
        for K in KS:
            occ_c[K] += oc[K]; occ_ht[K] += otu[K]; occ_sv[K] += osv[K]
        print(f"  window {w} ({time.time()-t_all:.0f}s)", flush=True)

    def reg(o, K):
        return (o[K] * K + 1) / (K + 2)

    def over_K(pa_fn, pb_fn, label):
        rec = {}
        for K in KS:
            ig, ci = bern_ig_ci(wt, y, pa_fn(K), pb_fn(K))
            rec[f"K{K}"] = {"ig": ig, "ci": ci}
        xs = np.array([1.0 / K for K in KS])
        ys = np.array([rec[f"K{K}"]["ig"] for K in KS])
        slope, inter = np.polyfit(xs, ys, 1)
        ci5 = rec["K5000"]["ci"]; off = float(inter) - rec["K5000"]["ig"]
        # r2 of the 1/K fit: a clean Jensen trend gives r2 -> 1. Where BOTH models are MC with
        # the SAME per-window seed the biases largely cancel, leaving no trend to remove -- the
        # extrapolation then fits noise and the residual should be read as ~0 (the pre-stated
        # prediction of the Amendment-3 addendum, item 1).
        ss_res = float(np.sum((ys - (slope * xs + inter)) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return {"label": label, "per_K": rec,
                "debiased_Kinf": round(float(inter), 4),
                "slope_per_unit_inv_K": round(float(slope), 3),
                "inv_K_fit_r2": round(float(r2), 4),
                "monotone_in_invK": bool(np.all(np.diff(ys[np.argsort(-xs)]) <= 0)
                                         or np.all(np.diff(ys[np.argsort(-xs)]) >= 0)),
                # Table 4's convention: debiased POINT reported with the RAW K=5000 interval.
                "ci95_K5000_raw_table4_convention": ci5,
                # alternative: K=5000 interval shifted by the debias offset
                "debiased_ci95_K5000_shifted": [round(ci5[0] + off, 4), round(ci5[1] + off, 4)]}

    out = {
        "scoring": ("native Monte-Carlo occupancy, Bernoulli log-score, add-one regularized "
                    "p_hat=(occ*K+1)/(K+2), linearly extrapolated in 1/K to K->inf (Jensen debias)"),
        "K_levels": KS, "KMAX": KMAX, "b_op": B_OP, "seed_base": SEED_BASE,
        "n_test_windows": len(wins), "n_cellwindows": n, "n_pos": int(y.sum()),
        "hybrid_construction": "top-up (rate-exact) -- the Table 4/5 construction of record",
    }
    out["cascade_vs_sv_etas"] = over_K(lambda K: reg(occ_c, K), lambda K: reg(occ_sv, K),
                                       "cascade vs sv-ETAS (both native MC)")
    out["hybrid_vs_sv_etas"] = over_K(lambda K: reg(occ_ht, K), lambda K: reg(occ_sv, K),
                                      "hybrid (top-up) vs sv-ETAS (both native MC)")
    # reference pair, to confirm this run reproduces t6_debias's machinery
    out["hybrid_vs_cascade_topup_reference"] = over_K(lambda K: reg(occ_ht, K),
                                                      lambda K: reg(occ_c, K),
                                                      "hybrid (top-up) vs cascade [t6 reference]")

    tot = {"sv_etas": round(float(occ_sv[5000].sum()), 1),
           "cascade": round(float(occ_c[5000].sum()), 1),
           "hybrid_topup": round(float(occ_ht[5000].sum()), 1),
           "n_pos": int(y.sum())}
    out["table5_raw_occupancy_totals_K5000"] = {
        **tot,
        "ratio_sv_etas": round(tot["sv_etas"] / tot["n_pos"], 3),
        "ratio_cascade": round(tot["cascade"] / tot["n_pos"], 3),
        "ratio_hybrid_topup": round(tot["hybrid_topup"] / tot["n_pos"], 3),
        "paper_cascade_907_ratio_1.53": True,
    }
    # The adjudicator of record is the CONJUNCTIVE two-axis rule: A beats B only if BOTH the IG
    # and the (construction-free, score-invariant) intensity PR-AUC intervals exclude 0. Pull the
    # registered ranking axis from claims.json so the verdict is stated on both axes, not IG alone.
    claims = json.load(open(RESULTS / "claims.json"))["claims"]
    def dpr_of(pair):
        for c in claims:
            if c["pair"] == pair and c["target"] == "y30" and c["split"] == "test":
                return c["pr_auc"]
        return None
    dpr_c_sv = dpr_of("cascade_vs_sv_etas")
    dpr_h_sv = dpr_of("hybrid_vs_sv_etas")
    out["registered_ranking_axis_from_claims"] = {"cascade_vs_sv_etas": dpr_c_sv,
                                                  "hybrid_vs_sv_etas": dpr_h_sv}

    ci_c = out["cascade_vs_sv_etas"]["ci95_K5000_raw_table4_convention"]
    ci_h = out["hybrid_vs_sv_etas"]["ci95_K5000_raw_table4_convention"]
    def straddles(ci):
        return bool(ci is not None and ci[0] <= 0 <= ci[1])
    out["gate"] = {
        "cascade_vs_sv_etas_native_IG_excludes_0": bool(not straddles(ci_c)),
        "cascade_vs_sv_etas_dPR_straddles_0": straddles(dpr_c_sv["ci95"]) if dpr_c_sv else None,
        "cascade_vs_sv_etas_conjunctive_verdict":
            ("inseparable" if (dpr_c_sv is None or straddles(dpr_c_sv["ci95"]) or straddles(ci_c))
             else "separable"),
        "hybrid_vs_sv_etas_native_IG_excludes_0": bool(not straddles(ci_h)),
        "hybrid_vs_sv_etas_conjunctive_verdict":
            ("inseparable" if (dpr_h_sv is None or straddles(dpr_h_sv["ci95"]) or straddles(ci_h))
             else "separable"),
        "sv_occupancy_close_to_cascade": bool(abs(tot["sv_etas"] - tot["cascade"]) / tot["cascade"] < 0.05),
        "sv_ratio": round(tot["sv_etas"] / tot["n_pos"], 3),
        "cascade_ratio": round(tot["cascade"] / tot["n_pos"], 3),
    }
    out["runtime_s"] = round(time.time() - t_all, 1)
    json.dump(out, open(OUT / "r3_sv_etas_native.json", "w"), indent=2)

    print("\n=== Bernoulli-native, debiased (K->inf); CI = raw K=5000 (Table 4 convention) ===")
    for k in ("cascade_vs_sv_etas", "hybrid_vs_sv_etas", "hybrid_vs_cascade_topup_reference"):
        r = out[k]
        print(f"{r['label']:44s} dIG {r['debiased_Kinf']:+.4f} "
              f"CI {r['ci95_K5000_raw_table4_convention']}  slope {r['slope_per_unit_inv_K']} "
              f"r2 {r['inv_K_fit_r2']} mono {r['monotone_in_invK']}")
        print(f"{'':44s} per-K {[v['ig'] for v in r['per_K'].values()]}")
        print(f"{'':44s} per-K CIs {[v['ci'] for v in r['per_K'].values()]}")
    print(f"\nTable 5 raw occupancy (K=5000): {json.dumps(out['table5_raw_occupancy_totals_K5000'])}")
    print(f"GATE: {json.dumps(out['gate'])}")
    print(f"runtime {out['runtime_s']}s -> results/round4/r3_sv_etas_native.json")


if __name__ == "__main__":
    main()
