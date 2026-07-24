"""Wide-box y45 (remedy for the sparse model-box y45).

The model-box y45 has only ~22 test positives, so the hybrid overfits. Training on the
WIDE box (25.0-31.5 / 39.0-42.5, 2275 cells) captures ~2-3x more M>=4.5 events; we then
EVALUATE on the Marmara model-box cells only, for a like-for-like comparison with the
model-box y45. Features: the full 19 on WIDE_SPEC (features_full_spec) + cascade lam45.

Output: results/grid/widebox_y45_grid.parquet , results/grid/widebox_y45_report.json
Run:  "<venv>/bin/python3" -m marmara.widebox_y45 build   (then) ... eval
      "<venv>/bin/python3" -m marmara.widebox_y45          (build+eval)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401
from sklearn.ensemble import HistGradientBoostingRegressor

from marmara import grid as G
from marmara.cascade import cascade_forecast
from marmara.metrics import information_gain, lambda_to_p, score_predictor
from marmara.train import TRAIN_END, VAL_END, TEST_TARGET_END

OUT = RESULTS
GRIDF = OUT / "grid" / "widebox_y45_grid.parquet"
BASE_MC = 3.0
K = 500
EPS = 1e-9
SPEC = G.WIDE_SPEC


def build():
    with open(OUT / "etas" / "etas_params.pkl", "rb") as f:
        import pickle; params = pickle.load(f)
    b_op = json.load(open(OUT / "etas" / "etas_fit_report.json"))["operational_b_for_cascade"]
    cat = pd.read_csv(OUT / "catalog" / "catalog_widebox.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    td = ((cat["datetime_utc"] - G.REF) / pd.Timedelta(days=1)).to_numpy()
    EV = G.build_bundle(td, cat["longitude"].to_numpy(), cat["latitude"].to_numpy(),
                        cat["mag_w"].to_numpy(), cat["depth_km"].to_numpy(), SPEC, BASE_MC)
    ctx = G.build_static_context_spec(SPEC)
    starts = G.window_starts(cat["datetime_utc"].max())
    hist = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]

    LO, LA = np.meshgrid(SPEC.lon_c, SPEC.lat_c)
    ir_flat = np.repeat(np.arange(SPEC.nlat), SPEC.nlon)
    ic_flat = np.tile(np.arange(SPEC.nlon), SPEC.nlat)
    in_box = ((LO.ravel() >= 25.6) & (LO.ravel() <= 30.9)
              & (LA.ravel() >= 39.6) & (LA.ravel() <= 41.9)).astype(int)

    rows = []; t_all = time.time()
    for kk, t0_dt in enumerate(starts):
        t0d = float(G._to_days(t0_dt))
        feats = G.features_full_spec(EV, t0d, t0_dt, SPEC, ctx, params)
        casc = cascade_forecast(params, hist[cat["datetime_utc"] < t0_dt], t0d, G.HORIZON_D,
                                SPEC.lon_c, SPEC.lat_c, K=K, seed=3000 + kk, b=b_op)
        # count45 / y45 targets on the wide box
        sub = EV["e45"]; lo = np.searchsorted(sub["t"], t0d, "left"); hi = np.searchsorted(sub["t"], t0d + 30.0, "left")
        c45 = np.zeros((SPEC.nlat, SPEC.nlon)); np.add.at(c45, (sub["ir"][lo:hi], sub["ic"][lo:hi]), 1.0)
        block = {"window": np.full(SPEC.ncells, kk),
                 "t0": np.full(SPEC.ncells, np.datetime64(t0_dt), dtype="datetime64[ns]"),
                 "ir": ir_flat, "ic": ic_flat, "in_box": in_box}
        for name in G.FEATURES:
            block[name] = feats[name].ravel()
        block["lam45_sim"] = casc["lam45"].ravel()
        block["count45"] = c45.ravel()
        block["y45"] = (c45.ravel() > 0).astype(float)
        rows.append(pd.DataFrame(block))
        if (kk + 1) % 60 == 0:
            print(f"  window {kk+1}/{len(starts)} ({time.time()-t_all:.0f}s)")
    grid = pd.concat(rows, ignore_index=True)
    grid.to_parquet(GRIDF, index=False)
    print(f"widebox y45 grid: {len(grid)} rows, {int(grid['y45'].sum())} positives, "
          f"{int(grid['count45'].sum())} events ({time.time()-t_all:.0f}s)")
    return grid


def masks(grid):
    t0 = pd.to_datetime(grid["t0"]); tgt = t0 + pd.Timedelta(days=30)
    return {"train": (t0 < TRAIN_END).to_numpy(),
            "val": ((t0 >= TRAIN_END) & (t0 < VAL_END)).to_numpy(),
            "test": ((t0 >= VAL_END) & (tgt <= TEST_TARGET_END)).to_numpy()}


def evaluate(grid):
    m = masks(grid)
    X = grid[G.FEATURES].copy(); X["ln_lam_sim"] = np.log(grid["lam45_sim"].to_numpy() + EPS)
    cols = G.FEATURES + ["ln_lam_sim"]
    n = grid["count45"].to_numpy(float); y = grid["y45"].to_numpy(float)
    lam_sim = np.clip(grid["lam45_sim"].to_numpy(), EPS, None)
    win = grid["window"].to_numpy()

    reg = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=400,
                                        max_depth=6, random_state=42)
    reg.fit(X[m["train"]], n[m["train"]])
    lam_ml = np.clip(reg.predict(X), EPS, None)

    # choose w by the pre-registered 1-SE parsimony gate (uniform with train.py): the
    # smallest w within one block-bootstrap SE of the val-LL argmax. On the sparse
    # wide-box y45 val this is expected to drive w->0, retroactively preventing the
    # manually-diagnosed overfit.
    from marmara.train import select_w_1se, WEIGHTS
    best, wdiag = select_w_1se(lam_sim, lam_ml, y, m["val"], win, WEIGHTS)
    lam_h = lam_sim ** (1 - best) * lam_ml ** best

    # EVALUATE on MODEL-BOX cells only, TEST windows
    ti = np.where(m["test"] & (grid["in_box"].to_numpy() == 1))[0]
    ys, ws = y[ti], win[ti]
    P_hyb = lambda_to_p(lam_h[ti]); P_casc = lambda_to_p(grid["lam45_sim"].to_numpy()[ti])
    sc_h = score_predictor(P_hyb, ys, ws); sc_c = score_predictor(P_casc, ys, ws)
    ig = information_gain(P_hyb, P_casc, ys)
    rep = {
        "chosen_w": float(best), "w_argmax_naive": float(wdiag["w_argmax"]),
        "n_widebox_rows": int(len(grid)),
        "widebox_train_positives": int(y[m["train"]].sum()),
        "modelbox_test_cells": int(len(ti)), "modelbox_test_positives": int(ys.sum()),
        "hybrid_widebox_trained": {"pr_auc": sc_h["pr_auc"], "roc_auc": sc_h["roc_auc"],
                                   "brier": sc_h["brier"], "molchan": sc_h["molchan"]["area_skill"]},
        "cascade": {"pr_auc": sc_c["pr_auc"], "roc_auc": sc_c["roc_auc"],
                    "brier": sc_c["brier"], "molchan": sc_c["molchan"]["area_skill"]},
        "ig_hybrid_vs_cascade": round(ig, 4),
    }
    # compare to the model-box-only y45 hybrid (from evaluation)
    try:
        mb = json.load(open(OUT / "scoring" / "evaluation.json"))["targets"]["y45"]["splits"]["test"]
        rep["modelbox_only_y45"] = {"hybrid_pr_auc": mb["scores"]["hybrid"]["pr_auc"],
                                    "hybrid_brier": mb["scores"]["hybrid"]["brier"],
                                    "ig_hybrid_vs_cascade": mb["ig_hybrid_vs"]["cascade"]}
    except Exception:
        pass
    json.dump(rep, open(OUT / "grid" / "widebox_y45_report.json", "w"), indent=2)
    print("\nwidebox-y45 (eval on model-box test cells):")
    print(f"  chosen w={best}; widebox train positives {rep['widebox_train_positives']} "
          f"(vs ~104 model-box); model-box test positives {rep['modelbox_test_positives']}")
    print(f"  hybrid(widebox-trained) PR-AUC {sc_h['pr_auc']:.4f} Brier {sc_h['brier']:.6f} "
          f"| cascade PR-AUC {sc_c['pr_auc']:.4f} Brier {sc_c['brier']:.6f}")
    print(f"  IG(hybrid vs cascade) = {ig:+.4f}")
    if "modelbox_only_y45" in rep:
        mo = rep["modelbox_only_y45"]
        print(f"  vs model-box-only y45: hybrid PR-AUC {mo['hybrid_pr_auc']:.4f} Brier "
              f"{mo['hybrid_brier']:.6f} IG {mo['ig_hybrid_vs_cascade']:+.4f}")
    return rep


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "all"
    grid = pd.read_parquet(GRIDF) if (arg == "eval" and GRIDF.exists()) else build()
    if arg != "build":
        evaluate(grid)


if __name__ == "__main__":
    main()
