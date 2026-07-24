"""(i)+(iii): native occurrence scoring on TEST. Score each model at the forecast it states:
  cascade  -> native MC occupancy P_hat(N>=1) from K=500 clustered catalogues
  hybrid   -> native occupancy via per-cell thinning of the cascade catalogues by lam_hybrid/lam_cascade
  first-gen, inversion, poisson, smoothed -> 1 - e^{-lambda}  (their stated Poisson occurrence)
Bernoulli log-score, block-bootstrap CI, add-one shrinkage (K=500 -> p_hat=0 gives -inf). Also the
per-model-scalar DIAGNOSTIC (isolates the non-global part). b_op=1.15 (primary). Units: IG per positive.
Writes results/round3/t5_native_occupancy.json.
"""
import json
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from marmara.paths import RESULTS
from marmara import grid as G
from marmara.train import split_masks
from marmara.cascade import cascade_forecast
from marmara.metrics import p_to_lambda, lambda_to_p
from marmara.bootstrap import stationary_window_indices, MEAN_BLOCK, SEED

EPS = 1e-9
K = 500
B_OP = 1.15
B = 2000


def lam(p):
    return np.clip(p_to_lambda(np.clip(np.asarray(p, float), 0.0, 1.0)), EPS, None)


def bern_two_axis(win, y, pa, pb):
    pa = np.clip(pa, EPS, 1 - EPS); pb = np.clip(pb, EPS, 1 - EPS)
    cell = lambda p: y * np.log(p) + (1 - y) * np.log(1 - p)
    ca_cell, cb_cell = cell(pa), cell(pb)
    wins = np.sort(np.unique(win)); idx = {w: np.where(win == w)[0] for w in wins}
    ca = np.array([ca_cell[idx[w]].sum() for w in wins]); cb = np.array([cb_cell[idx[w]].sum() for w in wins])
    pos = np.array([y[idx[w]].sum() for w in wins])
    ig = (ca.sum() - cb.sum()) / max(pos.sum(), 1)
    pr = average_precision_score(y, pa) - average_precision_score(y, pb)
    rng = np.random.default_rng(SEED); seqs = stationary_window_indices(len(wins), B, MEAN_BLOCK, rng)
    yb = [y[idx[w]] for w in wins]; ab = [pa[idx[w]] for w in wins]; bb = [pb[idx[w]] for w in wins]
    igb, prb = [], []
    for r in seqs:
        igb.append((ca[r].sum() - cb[r].sum()) / max(pos[r].sum(), 1))
        yy = np.concatenate([yb[i] for i in r])
        if 0 < yy.sum() < len(yy):
            prb.append(average_precision_score(yy, np.concatenate([ab[i] for i in r]))
                       - average_precision_score(yy, np.concatenate([bb[i] for i in r])))
    ic = [round(float(np.percentile(igb, 2.5)), 4), round(float(np.percentile(igb, 97.5)), 4)]
    pc = [round(float(np.percentile(prb, 2.5)), 4), round(float(np.percentile(prb, 97.5)), 4)]
    sep = (ic[0] > 0 or ic[1] < 0) and (pc[0] > 0 or pc[1] < 0)
    return {"ig": round(float(ig), 4), "ig_ci": ic, "dpr": round(float(pr), 4), "dpr_ci": pc,
            "verdict": "separable" if sep else "inseparable"}


def main():
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    m = split_masks(grid); win_t0 = grid.groupby("window")["t0"].first()
    params = pickle.load(open(RESULTS / "etas" / "etas_params.pkl", "rb"))
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    hist = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]; spec = G.MODEL_SPEC
    NLON = G.NLON

    gt = grid[m["test"]].sort_values(["window", "ir", "ic"]).reset_index(drop=True)
    key = {(int(r.window), int(r.ir), int(r.ic)): i for i, r in gt.iterrows()}
    n = len(gt)
    # lam_hybrid per cell from canonical predictions, aligned to gt order
    pred = pd.read_parquet(RESULTS / "grid" / "predictions_y30.parquet")
    pte = pred[pred.split == "test"].reset_index(drop=True)
    gh_te = grid[m["test"]].reset_index(drop=True)
    assert np.array_equal(gh_te.y30.to_numpy(float), pte.y.to_numpy(float))
    tmp = gh_te.assign(hyb=pte.hybrid.to_numpy(), fg=pte.firstgen_etas.to_numpy(),
                       inv=pte.modern_etas.to_numpy(), sv=pte.sv_etas.to_numpy()).sort_values(
        ["window", "ir", "ic"]).reset_index(drop=True)
    lam_hyb = lam(tmp.hyb.to_numpy()); y = tmp.y30.to_numpy(float)
    P_fg = np.clip(tmp.fg.to_numpy(), EPS, 1 - EPS)          # 1-e^{-lambda} already
    P_inv = np.clip(tmp.inv.to_numpy(), EPS, 1 - EPS)

    lam_casc = np.zeros(n); hit_casc = np.zeros(n); hit_hyb = np.zeros(n)
    rng = np.random.default_rng(12345)
    for w in [int(x) for x in np.sort(gt.window.unique())]:
        t0_dt = pd.Timestamp(win_t0.loc[w]); t0d = float(G._to_days(t0_dt))
        ev = cascade_forecast(params, hist[cat["datetime_utc"] < t0_dt], t0d, G.HORIZON_D,
                              spec.lon_c, spec.lat_c, K=K, seed=1000 + w, b=B_OP,
                              preserve_branching=True, return_events=True)
        sid = ev["sim"]
        ic = np.floor((ev["lon"] - round(float(spec.lon_c[0]) - 0.05, 2)) / 0.1).astype(int)
        ir = np.floor((ev["lat"] - round(float(spec.lat_c[0]) - 0.05, 2)) / 0.1).astype(int)
        # per cell: total events (lam), distinct sims (occupancy)
        tot = defaultdict(int); sims_c = defaultdict(set)
        cell_events = defaultdict(list)
        for s_, r_, c_ in zip(sid, ir, ic):
            tot[(int(r_), int(c_))] += 1; sims_c[(int(r_), int(c_))].add(int(s_))
            cell_events[(int(r_), int(c_))].append(int(s_))
        for (r_, c_), tt in tot.items():
            i = key.get((w, r_, c_))
            if i is None:
                continue
            lam_casc[i] = tt / K; hit_casc[i] = len(sims_c[(r_, c_)])
            lc = tt / K; lh = lam_hyb[i]
            ratio = min(1.0, lh / lc) if lc > 0 else 0.0
            # thin this cell's events by `ratio`, occupancy = distinct sims with a kept event
            evs = np.array(cell_events[(r_, c_)])
            keep = rng.random(len(evs)) < ratio
            hit_hyb[i] = len(np.unique(evs[keep])) if keep.any() else 0

    # add-one (Laplace) shrinkage: p_hat = (hits+1)/(K+2)
    P_casc = (hit_casc + 1) / (K + 2)
    P_hyb = (hit_hyb + 1) / (K + 2)
    # lambda-blend sensitivity: 0.5*p_hat + 0.5*(1-e^{-lambda})
    P_casc_blend = 0.5 * (hit_casc / K) + 0.5 * (1 - np.exp(-lam_casc))
    P_casc_blend = np.clip(P_casc_blend, 1.0 / (K + 2), 1 - EPS)

    wt = tmp.window.to_numpy()
    out = {"scoring": "Bernoulli log-score on NATIVE occurrence (add-one shrinkage), b_op 1.15, TEST",
           "totals": {"sum_P_casc_native": round(float(P_casc.sum()), 1),
                      "sum_P_hyb_native": round(float(P_hyb.sum()), 1),
                      "sum_P_firstgen": round(float(P_fg.sum()), 1),
                      "sum_P_inversion": round(float(P_inv.sum()), 1),
                      "n_pos_test": int(y.sum()), "note": "test totals vs 592 occurrences"},
           "native_bernoulli": {
               "hybrid_vs_cascade": bern_two_axis(wt, y, P_hyb, P_casc),
               "cascade_vs_inversion": bern_two_axis(wt, y, P_casc, P_inv),
               "firstgen_vs_inversion": bern_two_axis(wt, y, P_fg, P_inv),
               "hybrid_vs_firstgen": bern_two_axis(wt, y, P_hyb, P_fg),
               "cascade_vs_firstgen": bern_two_axis(wt, y, P_casc, P_fg)},
           "shrinkage_sensitivity_hybrid_vs_cascade": {
               "add_one": bern_two_axis(wt, y, P_hyb, P_casc),
               "lambda_blend_cascade": bern_two_axis(wt, y, P_hyb, P_casc_blend)}}
    (RESULTS / "round3" / "t5_native_occupancy.json").write_text(json.dumps(out, indent=2))
    print("TOTALS:", out["totals"])
    print("\nNATIVE Bernoulli verdicts (b_op 1.15, test):")
    for k, v in out["native_bernoulli"].items():
        print(f"  {k:24s}: IG {v['ig']:+.3f} {v['ig_ci']}  dPR {v['dpr']:+.4f}{v['dpr_ci']}  [{v['verdict']}]")
    print("shrinkage (hyb vs casc): add-one",
          out["shrinkage_sensitivity_hybrid_vs_cascade"]["add_one"]["ig"],
          "lambda-blend", out["shrinkage_sensitivity_hybrid_vs_cascade"]["lambda_blend_cascade"]["ig"])


if __name__ == "__main__":
    main()
