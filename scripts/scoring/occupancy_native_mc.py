"""Score-definition cross-check (item ii): the in-principle-right occurrence forecast for a
clustered simulator is its NATIVE MC occupancy P_hat(N>=1) from the K catalogues, not
scalar x intensity. Compute both on the cascade in VALIDATION and compare. Negligible difference
-> one sentence; real difference -> MC occupancy is the headline score.

For each val window, re-simulate the cascade with return_events, and per cell compute
  lam       = (# M>=3.0 events)/K              (mean count; = current lam30)
  P_MC      = (# distinct sims with >=1 event)/K   (native occupancy)
compare to  1-exp(-lam)  (naive, current paper)  and  1-exp(-s*lam)  (scalar, s val-fit).
Writes results/round3/t3_mc_occupancy.json.
"""
import json
import pickle
import numpy as np
import pandas as pd

from marmara.paths import RESULTS
from marmara import grid as G
from marmara.train import split_masks
from marmara.cascade import cascade_forecast
from marmara.metrics import p_to_lambda

EPS = 1e-9
K = 500


def main():
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    m = split_masks(grid)
    win_t0 = grid.groupby("window")["t0"].first()
    params = pickle.load(open(RESULTS / "etas" / "etas_params.pkl", "rb"))
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    hist = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]
    spec = G.MODEL_SPEC
    NLON = G.NLON

    gv = grid[m["val"]].sort_values(["window", "ir", "ic"]).reset_index(drop=True)
    val_wins = [int(w) for w in np.sort(gv["window"].unique())]
    lam = np.zeros(len(gv)); pmc = np.zeros(len(gv))
    key = {(int(r.window), int(r.ir), int(r.ic)): i for i, r in gv.iterrows()}
    for w in val_wins:
        t0_dt = pd.Timestamp(win_t0.loc[w]); t0d = float(G._to_days(t0_dt))
        ev = cascade_forecast(params, hist[cat["datetime_utc"] < t0_dt], t0d, G.HORIZON_D,
                              spec.lon_c, spec.lat_c, K=K, seed=1000 + w, b=1.15,
                              preserve_branching=True, return_events=True)
        simid = ev["sim"]
        ic = np.floor((ev["lon"] - (round(float(spec.lon_c[0]) - 0.05, 2))) / 0.1).astype(int)
        ir = np.floor((ev["lat"] - (round(float(spec.lat_c[0]) - 0.05, 2))) / 0.1).astype(int)
        # per cell: total events (for lam) and distinct sims (for occupancy)
        from collections import defaultdict
        tot = defaultdict(int); sims = defaultdict(set)
        for s_, r_, c_ in zip(simid, ir, ic):
            tot[(int(r_), int(c_))] += 1; sims[(int(r_), int(c_))].add(int(s_))
        for (r_, c_), n in tot.items():
            idx = key.get((w, r_, c_))
            if idx is not None:
                lam[idx] = n / K; pmc[idx] = len(sims[(r_, c_)]) / K

    y = gv["y30"].to_numpy(float)
    lam = np.clip(lam, 0, None)
    s = float(y.sum() / np.clip(p_to_lambda(np.clip(1 - np.exp(-lam), 0, 1 - EPS)), EPS, None).sum())
    p_naive = 1 - np.exp(-lam)
    p_scalar = 1 - np.exp(-s * lam)

    def ig(y, la, lb):
        la = np.clip(la, EPS, None); lb = np.clip(lb, EPS, None)
        return float((np.sum(y * np.log(la) - la) - np.sum(y * np.log(lb) - lb)) / max(y.sum(), 1))
    lam_mc = np.clip(p_to_lambda(np.clip(pmc, 0, 1 - EPS)), EPS, None)
    lam_sc = np.clip(p_to_lambda(np.clip(p_scalar, 0, 1 - EPS)), EPS, None)
    lam_nv = np.clip(p_to_lambda(np.clip(p_naive, 0, 1 - EPS)), EPS, None)

    occ = lam > 0
    out = {
        "n_val_cells": int(len(gv)), "n_active_cells": int(occ.sum()), "s_val": round(s, 4),
        "mean_abs_diff_MC_vs_scalar_occ": round(float(np.mean(np.abs(pmc[occ] - p_scalar[occ]))), 5),
        "mean_abs_diff_MC_vs_naive_occ": round(float(np.mean(np.abs(pmc[occ] - p_naive[occ]))), 5),
        "corr_MC_vs_scalar": round(float(np.corrcoef(pmc[occ], p_scalar[occ])[0, 1]), 4),
        "sum_P_MC": round(float(pmc.sum()), 1), "sum_P_scalar": round(float(p_scalar.sum()), 1),
        "sum_P_naive": round(float(p_naive.sum()), 1), "n_pos_val": int(y.sum()),
        "ig_MC_occ_vs_naive": round(ig(y, lam_mc, lam_nv), 4),
        "ig_scalar_vs_naive": round(ig(y, lam_sc, lam_nv), 4),
        "ig_MC_occ_vs_scalar": round(ig(y, lam_mc, lam_sc), 4),
        "note": "MC occupancy = simulator's native P(N>=1); if ig_MC~ig_scalar and diff small, the "
                "global scalar is a validated proxy (one sentence); else MC occupancy is the headline.",
    }
    (RESULTS / "round3" / "t3_mc_occupancy.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
