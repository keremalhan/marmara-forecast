"""(iv) grind: native-Bernoulli b_op-robustness of the text-referenced pairs + y35 prediction test +
raw occupancy totals per b_op. Re-sims cascade native occupancy at b_op 1.10/1.20 (1.15 from t6),
hybrid via top-up thinning; scores the key pairs under Bernoulli-native (K=2000, add-one, raw totals
reported). Also y35 native hybrid-vs-cascade (tests pre-stated prediction < +0.095).
Emits results/round3/claims_bernoulli.json. Long batch; cannot move a verdict.
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
from marmara.metrics import p_to_lambda
from marmara.bootstrap import stationary_window_indices, MEAN_BLOCK, SEED

EPS = 1e-9
K = 2000


def lam(p):
    return np.clip(p_to_lambda(np.clip(np.asarray(p, float), 0.0, 1.0)), EPS, None)


def bern_ig(win, y, pa, pb):
    pa = np.clip(pa, EPS, 1 - EPS); pb = np.clip(pb, EPS, 1 - EPS)
    c = y * (np.log(pa) - np.log(pb)) + (1 - y) * (np.log(1 - pa) - np.log(1 - pb))
    wins = np.sort(np.unique(win)); idx = {w: np.where(win == w)[0] for w in wins}
    cw = np.array([c[idx[w]].sum() for w in wins]); pos = np.array([y[idx[w]].sum() for w in wins])
    ig = cw.sum() / max(pos.sum(), 1)
    rng = np.random.default_rng(SEED); seqs = stationary_window_indices(len(wins), 2000, MEAN_BLOCK, rng)
    bs = [cw[r].sum() / max(pos[r].sum(), 1) for r in seqs]
    return round(float(ig), 4), [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def cascade_native(grid, mask, params, hist, cat, spec, win_t0, b_op, lam_hyb, key):
    gt = grid[mask].sort_values(["window", "ir", "ic"]).reset_index(drop=True)
    n = len(gt); occ_c = np.zeros(n); occ_ht = np.zeros(n)
    for w in [int(x) for x in np.sort(gt.window.unique())]:
        t0_dt = pd.Timestamp(win_t0.loc[w]); t0d = float(G._to_days(t0_dt))
        ev = cascade_forecast(params, hist[cat["datetime_utc"] < t0_dt], t0d, G.HORIZON_D,
                              spec.lon_c, spec.lat_c, K=K, seed=1000 + w, b=b_op, preserve_branching=True, return_events=True)
        ic = np.floor((ev["lon"] - round(float(spec.lon_c[0]) - 0.05, 2)) / 0.1).astype(int)
        ir = np.floor((ev["lat"] - round(float(spec.lat_c[0]) - 0.05, 2)) / 0.1).astype(int)
        sims = defaultdict(set); cnts = defaultdict(list)
        for s_, r_, c_ in zip(ev["sim"], ir, ic):
            sims[(int(r_), int(c_))].add(int(s_)); cnts[(int(r_), int(c_))].append(int(s_))
        for (r_, c_), sset in sims.items():
            i = key.get((w, r_, c_))
            if i is None:
                continue
            lc = len(cnts[(r_, c_)]) / K; lh = lam_hyb[i]
            occ_c[i] = len(sset) / K
            if lh <= lc:
                us, ct = np.unique(cnts[(r_, c_)], return_counts=True)
                ratio = lh / lc if lc > 0 else 0.0
                occ_ht[i] = (1 - (1 - ratio) ** ct).sum() / K
            else:
                occ_ht[i] = (len(sset) + (K - len(sset)) * (1 - np.exp(-(lh - lc)))) / K
    return gt, (occ_c * K + 1) / (K + 2), (occ_ht * K + 1) / (K + 2)


def main():
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet"); m = split_masks(grid)
    win_t0 = grid.groupby("window")["t0"].first()
    params = pickle.load(open(RESULTS / "etas" / "etas_params.pkl", "rb"))
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    hist = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]; spec = G.MODEL_SPEC

    pred = pd.read_parquet(RESULTS / "grid" / "predictions_y30.parquet"); pt = pred[pred.split == "test"].reset_index(drop=True)
    gh_te = grid[m["test"]].reset_index(drop=True)
    tmp = gh_te.assign(hyb=pt.hybrid.to_numpy(), fg=pt.firstgen_etas.to_numpy(), inv=pt.modern_etas.to_numpy()
                       ).sort_values(["window", "ir", "ic"]).reset_index(drop=True)
    lam_hyb = lam(tmp.hyb.to_numpy()); y = tmp.y30.to_numpy(float); wt = tmp.window.to_numpy()
    P_fg = np.clip(tmp.fg.to_numpy(), EPS, 1 - EPS); P_inv = np.clip(tmp.inv.to_numpy(), EPS, 1 - EPS)
    key = {(int(r.window), int(r.ir), int(r.ic)): i for i, r in tmp.reset_index().iterrows()}
    # rebuild key from the sorted test grid used by cascade_native
    gt0 = grid[m["test"]].sort_values(["window", "ir", "ic"]).reset_index(drop=True)
    key = {(int(r.window), int(r.ir), int(r.ic)): i for i, r in gt0.iterrows()}

    claims = {"note": "native-Bernoulli, K=2000 add-one, raw totals; secondary to registered count-rule",
              "b_op": {}}
    for b_op in (1.10, 1.20):
        gt, Pc, Pht = cascade_native(grid, m["test"], params, hist, cat, spec, win_t0, b_op, lam_hyb, key)
        claims["b_op"][f"{b_op:.2f}"] = {
            "raw_total_cascade": round(float(((Pc * (K + 2) - 1) / K).sum()), 1),
            "raw_total_hybrid_topup": round(float(((Pht * (K + 2) - 1) / K).sum()), 1),
            "hybrid_vs_cascade": dict(zip(("ig", "ci"), bern_ig(wt, y, Pht, Pc))),
            "cascade_vs_inversion": dict(zip(("ig", "ci"), bern_ig(wt, y, Pc, P_inv))),
            "cascade_vs_firstgen": dict(zip(("ig", "ci"), bern_ig(wt, y, Pc, P_fg)))}
        print(f"b_op {b_op}: hyb-casc {claims['b_op'][f'{b_op:.2f}']['hybrid_vs_cascade']}, "
              f"casc-inv {claims['b_op'][f'{b_op:.2f}']['cascade_vs_inversion']}", flush=True)

    # y35 native hybrid-vs-cascade at 1.15 (test the prediction < +0.095)
    p35 = pd.read_parquet(RESULTS / "grid" / "predictions_y35.parquet"); p35t = p35[p35.split == "test"].reset_index(drop=True)
    g35 = grid[m["test"]].reset_index(drop=True)
    t35 = g35.assign(hyb=p35t.hybrid.to_numpy()).sort_values(["window", "ir", "ic"]).reset_index(drop=True)
    lam_hyb35 = lam(t35.hyb.to_numpy()); y35 = t35.y35.to_numpy(float); wt35 = t35.window.to_numpy()
    gt, Pc35, Pht35 = cascade_native(grid, m["test"], params, hist, cat, spec, win_t0, 1.15, lam_hyb35, key)
    claims["y35_hybrid_vs_cascade_native_1.15"] = dict(zip(("ig", "ci"), bern_ig(wt35, y35, Pht35, Pc35)))
    claims["y35_prediction"] = "pre-stated < +0.095 (y30 native edge)"
    (RESULTS / "round3" / "claims_bernoulli.json").write_text(json.dumps(claims, indent=2))
    print("y35 native hyb-casc:", claims["y35_hybrid_vs_cascade_native_1.15"], "(predicted < +0.095)")
    print("wrote claims_bernoulli.json")


if __name__ == "__main__":
    main()
