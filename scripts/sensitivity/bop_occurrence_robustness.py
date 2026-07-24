"""Gate: does the cascade-vs-inversion OCCURRENCE separation hold across b_op {1.10,1.20}?
(1.15 already SEPARABLE in t3_lock.) Re-sim cascade val+test at each b_op; inversion (modern_etas)
is b_op-invariant (its first-gen intensity is integrated, not offspring-b dependent), taken from
canonical predictions. Occurrence-calibrate both by their val s*, two-axis verdict on test.
Writes results/round3/t3_bop_occ.json.
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
from marmara.metrics import p_to_lambda, lambda_to_p
from marmara.bootstrap import stationary_window_indices, MEAN_BLOCK, SEED

EPS = 1e-9
K = 500
BOPS = [1.10, 1.20]


def lam(p):
    return np.clip(p_to_lambda(np.clip(np.asarray(p, float), 0.0, 1.0)), EPS, None)


def cascade_lam(grid, mask, params, hist, cat, spec, win_t0, b_op):
    tw = grid[mask].sort_values(["window", "ir", "ic"]).reset_index(drop=True)
    wins = [int(w) for w in np.sort(tw["window"].unique())]
    L = np.zeros(len(tw))
    for w in wins:
        t0_dt = pd.Timestamp(win_t0.loc[w]); t0d = float(G._to_days(t0_dt))
        casc = cascade_forecast(params, hist[cat["datetime_utc"] < t0_dt], t0d, G.HORIZON_D,
                                spec.lon_c, spec.lat_c, K=K, seed=1000 + w, b=b_op, preserve_branching=True)
        rows = (tw["window"] == w).to_numpy()
        L[rows] = casc["lam30"][tw.loc[rows, "ir"].to_numpy(), tw.loc[rows, "ic"].to_numpy()]
    return tw["window"].to_numpy(), tw["y30"].to_numpy(float), np.clip(L, EPS, None)


def two_axis(win, y, la, lb, pa, pb):
    wins = np.sort(np.unique(win)); idx = {w: np.where(win == w)[0] for w in wins}
    ca = np.array([np.sum(y[idx[w]] * np.log(la[idx[w]]) - la[idx[w]]) for w in wins])
    cb = np.array([np.sum(y[idx[w]] * np.log(lb[idx[w]]) - lb[idx[w]]) for w in wins])
    pos = np.array([y[idx[w]].sum() for w in wins])
    ig = (ca.sum() - cb.sum()) / max(pos.sum(), 1)
    pr = average_precision_score(y, pa) - average_precision_score(y, pb)
    rng = np.random.default_rng(SEED); seqs = stationary_window_indices(len(wins), 2000, MEAN_BLOCK, rng)
    yb = [y[idx[w]] for w in wins]; ab = [pa[idx[w]] for w in wins]; bb = [pb[idx[w]] for w in wins]
    ig_bs, pr_bs = [], []
    for r in seqs:
        ig_bs.append((ca[r].sum() - cb[r].sum()) / max(pos[r].sum(), 1))
        yy = np.concatenate([yb[i] for i in r])
        if 0 < yy.sum() < len(yy):
            pr_bs.append(average_precision_score(yy, np.concatenate([ab[i] for i in r]))
                         - average_precision_score(yy, np.concatenate([bb[i] for i in r])))
    ic = [round(float(np.percentile(ig_bs, 2.5)), 4), round(float(np.percentile(ig_bs, 97.5)), 4)]
    pc = [round(float(np.percentile(pr_bs, 2.5)), 4), round(float(np.percentile(pr_bs, 97.5)), 4)]
    sep = (ic[0] > 0 or ic[1] < 0) and (pc[0] > 0 or pc[1] < 0)
    return {"ig": round(float(ig), 4), "ig_ci": ic, "dpr": round(float(pr), 4), "dpr_ci": pc,
            "verdict": "separable" if sep else "inseparable"}


def main():
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    m = split_masks(grid); win_t0 = grid.groupby("window")["t0"].first()
    params = pickle.load(open(RESULTS / "etas" / "etas_params.pkl", "rb"))
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    hist = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]; spec = G.MODEL_SPEC

    # inversion (modern_etas), b_op-invariant. Attach to grid[test] positionally (predictions align
    # to grid_hybrid row order), then sort by (window,ir,ic) to match cascade_lam's ordering.
    pred = pd.read_parquet(RESULTS / "grid" / "predictions_y30.parquet")
    gh_te = grid[m["test"]].reset_index(drop=True)
    te_pred = pred[pred.split == "test"].reset_index(drop=True)
    assert np.array_equal(gh_te["y30"].to_numpy(float), te_pred["y"].to_numpy(float)), "misaligned"
    gh_te = gh_te.assign(modern_etas=te_pred["modern_etas"].to_numpy()).sort_values(
        ["window", "ir", "ic"]).reset_index(drop=True)
    inv_t = lam(gh_te["modern_etas"].to_numpy())
    # inversion val (for s_inv) via same positional trick on val
    gh_va = grid[m["val"]].reset_index(drop=True)
    va_pred = pred[pred.split == "val"].reset_index(drop=True)
    inv_v = lam(gh_va.assign(me=va_pred["modern_etas"].to_numpy())["me"].to_numpy())
    s_inv = float(gh_va["y30"].sum() / inv_v.sum())

    out = {"note": "cascade vs inversion, occurrence-scored (global-s), per b_op; 1.15=separable (t3_lock)", "b_op": {}}
    for b_op in BOPS:
        _, yv, Lcv = cascade_lam(grid, m["val"], params, hist, cat, spec, win_t0, b_op)
        wt, yt, Lct = cascade_lam(grid, m["test"], params, hist, cat, spec, win_t0, b_op)
        s_casc = float(yv.sum() / Lcv.sum())
        Lc_occ = s_casc * Lct; Li_occ = s_inv * inv_t
        v = two_axis(wt, yt, Lc_occ, Li_occ, lambda_to_p(Lc_occ), lambda_to_p(Li_occ))
        out["b_op"][f"{b_op:.2f}"] = {"s_cascade_val": round(s_casc, 4), **v}
        print(f"b_op {b_op:.2f}: cascade vs inversion (occ) IG {v['ig']:+.3f}{v['ig_ci']} "
              f"dPR {v['dpr']:+.4f}{v['dpr_ci']} [{v['verdict']}]")
    (RESULTS / "round3" / "t3_bop_occ.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
