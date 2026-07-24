"""Run 26 — Item 7 Part B.2: recalibrate the block length on negative controls, BOTH scorings.

Governed by docs/preregistration/v2_analysis_amendment_8.md (SHA-256 c5684700..., hashed 2026-07-16T11:25:21Z).

r25 found the COUNT-scored ΔIG p-value construction mildly anticonservative at mean_block 3
(3/28 true-null pairs p<0.01; binomial P(>=3 | 0.01, 28) ~ 0.003 -- the decisive cell). The cause is
VARIANCE underestimation (window-consistent seed realizations), so the fix is BLOCK LENGTH, not
null-centering (which shifts location, not spread). T0-4 established every registered verdict is
unchanged at mean blocks 2/3/5/8, so lengthening the block cannot destabilize the adjudication.

This re-runs the 28 same-model cascade-seed controls (true diff = 0) and calibrates the p-value
construction on BOTH scorings -- count-scored ΔIG and Bernoulli-native ΔIG -- at mean blocks 3, 5, 8.
The three EXPOSED findings (+0.053 two-scalar, active-cell, and the +0.095 hybrid-vs-cascade Bernoulli
remainder) live on the BERNOULLI axis, so its calibration is computed directly, not assumed to
transfer from the count axis. Adopt the smallest block at which each scoring calibrates (no pile below
0.05; binomial tail at 0.01 not significant). Tuning is blind to any finding -> no forking paths.

One K-2000 return-events simulation per (seed, window) yields both the M>=3.0 rate (count axis) and
the native occupancy P(N>=1) (Bernoulli axis).

Writes results/round4/r26_recalibration.json.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/sensitivity/multiplicity_block_recalibration.py
"""
from __future__ import annotations

import json
import pickle
import time
from math import comb

import numpy as np
import pandas as pd

from marmara import grid as G
from marmara.bootstrap import SEED, stationary_window_indices
from marmara.cascade import cascade_forecast
from marmara.metrics import lambda_to_p, p_to_lambda
from marmara.paths import RESULTS
from marmara.train import split_masks

EPS = 1e-9
KMAX = 500
B_OP = 1.15
SEEDS = [1000, 4000, 7000, 10000, 13000, 16000, 19000, 22000]
BLOCKS = [3.0, 5.0, 8.0]


def p_two_sided(win, cvec, posvec, mean_block):
    """p = 2*min(P*(d<=0), P*(d>=0)) on the block-bootstrap of per-window contributions."""
    rng = np.random.default_rng(SEED)
    seqs = stationary_window_indices(len(win), 2000, mean_block, rng)
    d = np.array([cvec[r].sum() / max(posvec[r].sum(), 1) for r in seqs])
    return 2.0 * min((d <= 0).mean(), (d >= 0).mean())


def binom_tail(k, n, p):
    return sum(comb(n, i) * p**i * (1 - p)**(n - i) for i in range(k, n + 1))


def calib(pvals, block, scoring):
    ps = np.array(pvals)
    k01 = int((ps < 0.01).sum()); k05 = int((ps < 0.05).sum())
    tail01 = binom_tail(k01, len(ps), 0.01)
    clean = (tail01 > 0.05) and (k05 <= 0.11 * len(ps))   # no significant pile at 0.01, <=~11% below .05
    return {"scoring": scoring, "mean_block": block, "n": len(ps),
            "k_p<0.01": k01, "binom_tail_P(>=k|0.01)": round(tail01, 4),
            "k_p<0.05": k05, "median_p": round(float(np.median(ps)), 3),
            "calibrated": bool(clean)}


def main():
    t0 = time.time()
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    m = split_masks(grid)
    tw = grid[grid["window"].isin(np.sort(grid.loc[m["test"], "window"].unique()))].copy()
    tw = tw.sort_values(["window", "ir", "ic"]).reset_index(drop=True)
    keyi = {(int(r.window), int(r.ir), int(r.ic)): i for i, r in tw.iterrows()}
    win_t0 = grid.groupby("window")["t0"].first()
    test_wins = [int(w) for w in np.sort(tw["window"].unique())]
    params = pickle.load(open(RESULTS / "etas" / "etas_params.pkl", "rb"))
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    hist = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]
    spec = G.MODEL_SPEC
    y = tw["y30"].to_numpy(float); wt = tw["window"].to_numpy(); n = len(tw)

    # one K=2000 return-events sim per (seed, window): rate (count) + native occupancy (Bernoulli)
    rate_p = {}; occ_p = {}
    for sd in SEEDS:
        lam = np.zeros(n); occ = np.zeros(n)
        for w in test_wins:
            t0_dt = pd.Timestamp(win_t0.loc[w]); t0d = float(G._to_days(t0_dt))
            e = cascade_forecast(params, hist[cat["datetime_utc"] < t0_dt], t0d, G.HORIZON_D,
                                 spec.lon_c, spec.lat_c, K=KMAX, seed=sd + w, b=B_OP,
                                 preserve_branching=True, return_events=True)
            # both axes from the one event set: rate lam30 = events/K, occupancy = unique-sims/K
            sid = e["sim"]
            ic_ = np.floor((e["lon"] - round(float(spec.lon_c[0]) - 0.05, 2)) / 0.1).astype(int)
            ir_ = np.floor((e["lat"] - round(float(spec.lat_c[0]) - 0.05, 2)) / 0.1).astype(int)
            cf = ir_ * 100000 + ic_; o = np.argsort(cf, kind="stable"); sid, cf = sid[o], cf[o]
            uq = np.unique(cf); bd = np.searchsorted(cf, uq)
            for bi, c_ in enumerate(uq):
                i = keyi.get((w, int(c_ // 100000), int(c_ % 100000)))
                if i is None:
                    continue
                lo = bd[bi]; hi = bd[bi + 1] if bi + 1 < len(bd) else len(cf)
                lam[i] = (hi - lo) / KMAX                       # expected M>=3.0 count (rate)
                occ[i] = len(np.unique(sid[lo:hi])) / KMAX      # P(N>=1) occupancy
        rate_p[sd] = lambda_to_p(lam)
        occ_p[sd] = (occ * KMAX + 1) / (KMAX + 2)
        print(f"  seed {sd}: rate + occupancy built ({time.time()-t0:.0f}s)", flush=True)

    # per-window contributions for each pair, each scoring
    def contrib(pa, pb, scoring):
        if scoring == "count":
            la, lb = np.clip(p_to_lambda(pa), EPS, None), np.clip(p_to_lambda(pb), EPS, None)
            c = (y * np.log(la) - la) - (y * np.log(lb) - lb)
        else:  # bernoulli
            a, b_ = np.clip(pa, EPS, 1 - EPS), np.clip(pb, EPS, 1 - EPS)
            c = y * (np.log(a) - np.log(b_)) + (1 - y) * (np.log(1 - a) - np.log(1 - b_))
        cw = np.array([c[wt == w].sum() for w in test_wins])
        pw = np.array([y[wt == w].sum() for w in test_wins])
        return cw, pw

    out = {"governed_by": {"amendment": "docs/preregistration/v2_analysis_amendment_8.md",
                           "sha256": "c5684700aa656949908640faa326c6b6f15b3a699052f627272bc26a1186e690"},
           "n_null_pairs": comb(len(SEEDS), 2), "seeds": SEEDS, "blocks": BLOCKS,
           "calibration": {}}
    for scoring, src in (("count", rate_p), ("bernoulli", occ_p)):
        for block in BLOCKS:
            ps = []
            for i in range(len(SEEDS)):
                for jx in range(i + 1, len(SEEDS)):
                    cw, pw = contrib(src[SEEDS[i]], src[SEEDS[jx]], scoring)
                    ps.append(p_two_sided(test_wins, cw, pw, block))
            out["calibration"][f"{scoring}_block{int(block)}"] = calib(ps, block, scoring)
            c = out["calibration"][f"{scoring}_block{int(block)}"]
            print(f"  {scoring:9s} block {int(block)}: k(p<0.01)={c['k_p<0.01']} "
                  f"(binomP {c['binom_tail_P(>=k|0.01)']}), k(p<0.05)={c['k_p<0.05']}, "
                  f"median {c['median_p']} -> calibrated={c['calibrated']}", flush=True)

    # adopt smallest calibrated block per scoring
    adopt = {}
    for scoring in ("count", "bernoulli"):
        cal = [b for b in BLOCKS if out["calibration"][f"{scoring}_block{int(b)}"]["calibrated"]]
        adopt[scoring] = min(cal) if cal else None
    out["adopted_mean_block"] = adopt
    out["note"] = ("T0-4 established every registered verdict is unchanged at blocks 2/3/5/8, so adopting "
                   "a longer block for calibration cannot destabilize the adjudication of record.")
    out["runtime_s"] = round(time.time() - t0, 1)
    json.dump(out, open(RESULTS / "round4" / "r26_recalibration.json", "w"), indent=1, default=str)
    print(f"\n=== ADOPTED (smallest calibrated block) ===  count: {adopt['count']}  bernoulli: {adopt['bernoulli']}")
    print(f"  ({out['runtime_s']}s) -> results/round4/r26_recalibration.json")


if __name__ == "__main__":
    main()
