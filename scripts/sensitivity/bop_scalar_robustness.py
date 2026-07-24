"""Hygiene: does the scalar-recovers-the-edge finding hold across b_op {1.10,1.15,1.20}?

s interacts with b_op through Sum(lambda), so the gate applies. Re-simulate the cascade at each
b_op on val+test windows, fit s*=N_pos_val/Sum(lambda_val), score the scalar-rescaled cascade on
test: IG(s*lambda vs lambda) + block-bootstrap CI. Writes results/round3/t2_bop_scalar.json.
"""
import json
import pickle
import numpy as np
import pandas as pd

from marmara.paths import RESULTS
from marmara import grid as G
from marmara.train import split_masks
from marmara.cascade import cascade_forecast
from marmara.metrics import p_to_lambda, lambda_to_p
from marmara.bootstrap import stationary_window_indices, MEAN_BLOCK, SEED

EPS = 1e-9
K = 500
BOPS = [1.10, 1.20]        # 1.15 comes from canonical predictions


def cascade_lam(grid, mask_rows, params, hist, cat, spec, win_t0, b_op):
    """per-(cell,window) lam30 for the given rows, at b_op."""
    tw = grid[mask_rows].sort_values(["window", "ir", "ic"]).reset_index(drop=True)
    wins = [int(w) for w in np.sort(tw["window"].unique())]
    lam = np.zeros(len(tw))
    for w in wins:
        t0_dt = pd.Timestamp(win_t0.loc[w]); t0d = float(G._to_days(t0_dt))
        casc = cascade_forecast(params, hist[cat["datetime_utc"] < t0_dt], t0d, G.HORIZON_D,
                                spec.lon_c, spec.lat_c, K=K, seed=1000 + w, b=b_op, preserve_branching=True)
        rows = (tw["window"] == w).to_numpy()
        lam[rows] = casc["lam30"][tw.loc[rows, "ir"].to_numpy(), tw.loc[rows, "ic"].to_numpy()]
    return tw["window"].to_numpy(), tw["y30"].to_numpy(float), np.clip(lam, EPS, None)


def scalar_ig(win, y, lc, s):
    ls = s * lc
    wins = np.sort(np.unique(win)); idx = {w: np.where(win == w)[0] for w in wins}
    ca = np.array([np.sum(y[idx[w]] * np.log(ls[idx[w]]) - ls[idx[w]]) for w in wins])
    cb = np.array([np.sum(y[idx[w]] * np.log(lc[idx[w]]) - lc[idx[w]]) for w in wins])
    pos = np.array([y[idx[w]].sum() for w in wins])
    ig = (ca.sum() - cb.sum()) / max(pos.sum(), 1)
    rng = np.random.default_rng(SEED); seqs = stationary_window_indices(len(wins), 2000, MEAN_BLOCK, rng)
    bs = [(ca[r].sum() - cb[r].sum()) / max(pos[r].sum(), 1) for r in seqs]
    return round(float(ig), 4), [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def main():
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    m = split_masks(grid)
    win_t0 = grid.groupby("window")["t0"].first()
    params = pickle.load(open(RESULTS / "etas" / "etas_params.pkl", "rb"))
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    hist = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]
    spec = G.MODEL_SPEC

    out = {"note": "s* fit on validation; scalar IG on test, per b_op", "b_op": {}}
    for b_op in BOPS:
        _, yv, lcv = cascade_lam(grid, m["val"], params, hist, cat, spec, win_t0, b_op)
        wt, yt, lct = cascade_lam(grid, m["test"], params, hist, cat, spec, win_t0, b_op)
        s = float(yv.sum() / lcv.sum())
        ig, ci = scalar_ig(wt, yt, lct, s)
        out["b_op"][f"{b_op:.2f}"] = {"s_star_val": round(s, 4), "sumL_test": round(float(lct.sum()), 1),
                                      "scalar_ig_vs_cascade": ig, "ig_ci": ci}
        print(f"b_op {b_op:.2f}: s*={s:.4f}  sumL_test={lct.sum():.0f}  scalar IG {ig:+.4f} {ci}")
    # anchor at 1.15 from canonical predictions
    df = pd.read_parquet(RESULTS / "grid" / "predictions_y30.parquet")
    va = df[df.split == "val"]; te = df[df.split == "test"]
    lcv = np.clip(p_to_lambda(np.clip(va.cascade.to_numpy(), 0, 1 - EPS)), EPS, None)
    lct = np.clip(p_to_lambda(np.clip(te.cascade.to_numpy(), 0, 1 - EPS)), EPS, None)
    s = float(va.y.sum() / lcv.sum())
    ig, ci = scalar_ig(te.window.to_numpy(), te.y.to_numpy(), lct, s)
    out["b_op"]["1.15"] = {"s_star_val": round(s, 4), "sumL_test": round(float(lct.sum()), 1),
                           "scalar_ig_vs_cascade": ig, "ig_ci": ci, "source": "canonical predictions"}
    print(f"b_op 1.15: s*={s:.4f}  scalar IG {ig:+.4f} {ci} (canonical)")
    (RESULTS / "round3" / "t2_bop_scalar.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
