"""Run 12, item A (Amendment 5, SHA-256 c97db8f5...): Mc = 3.5 full-pipeline arm, y35 only.

QUESTION. Do the Table 2 (M>=3.5) verdicts persist when EVERY stage -- features, labels, ETAS fits,
b_op calibration, ML training, w-selection -- is conducted above a stricter completeness threshold,
rather than only the labels being raised (Table S12)?

SCOPE (stated before running, and binding on the write-up). This arm tests the M>=3.5 conclusion
under strict completeness. It does NOT and CANNOT validate the M>=3.0 primary target, which has no
existence at this floor.

PIPELINE, all at base Mc = 3.5:
  1. refit the in-house ETAS (fit_stai, same code path, same 0.95 cap, same FIT_END)
  2. b_op count-calibration sweep on PRE-TEST windows only
  3. rebuild the grid: 19 features + cascade, at the refit params and the new b_op
  4. truncated-catalogue leakage self-test on the new grid
  5. train the booster, select w by the 1-SE rule on validation
  6. evaluate y35 over the 26 test windows, both axes, all available pairs

sv-ETAS and the independent inversion are NOT refit at Mc = 3.5 (each needs its own EM /
third-party inversion re-run). Their absence is disclosed rather than passed over: this arm
compares hybrid, cascade and first-generation ETAS only.

Reported unconditionally. No gate.

Writes results/round4/r12_item_A_mc35.json (+ grid_mc35.parquet, etas_params_mc35.pkl).
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/sensitivity/mc35_pipeline_arm.py
"""
from __future__ import annotations

import gc
import json
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import marmara.etas_fit as EF
from marmara import grid as G
from marmara.bootstrap import MEAN_BLOCK, SEED, stationary_window_indices, verdict_for
from marmara.cascade import cascade_forecast
from marmara.etas_model import branching_ratio
from marmara.grid import FEATURES
from marmara.metrics import information_gain, lambda_to_p, p_to_lambda
from marmara.paths import RESULTS
from marmara.train import WEIGHTS, _monotonic, select_w_1se, split_masks

EPS = 1e-9
MC35 = 3.5
B_CANDIDATES = [0.9, 1.0, 1.05, 1.1, 1.12, 1.15, 1.18, 1.2, 1.542]
K_CAL = 300
K_BACKTEST = 500
B_BOOT = 2000
R4 = RESULTS / "round4"
R4.mkdir(exist_ok=True)
INT_COLS = {"cnt30", "cnt90", "cnt365", "nbr3_cnt30", "nbr3_cnt365",
            "nbr5_cnt30", "nbr5_cnt365", "b_pos_is_fallback"}


def paired(P_a, P_b, y, win):
    wins = np.sort(np.unique(win))
    y_by, a_by, b_by, lla, llb, pos, nrow = [], [], [], [], [], [], []
    for w in wins:
        k = win == w
        yv = y[k]
        y_by.append(yv); a_by.append(P_a[k]); b_by.append(P_b[k])
        pos.append(float(yv.sum())); nrow.append(float(k.sum()))
        for P, acc in ((P_a[k], lla), (P_b[k], llb)):
            lam = np.clip(p_to_lambda(np.clip(P, 0.0, 1.0)), EPS, None)
            acc.append(float(np.sum(yv * np.log(lam) - lam)))
    lla, llb = np.array(lla), np.array(llb)
    pos, nrow = np.array(pos), np.array(nrow)
    rng = np.random.default_rng(SEED)
    seqs = stationary_window_indices(len(wins), B_BOOT, MEAN_BLOCK, rng)
    d_ig = np.empty(B_BOOT); d_pr = np.full(B_BOOT, np.nan)
    for i in range(B_BOOT):
        s = seqs[i]
        d_ig[i] = (lla[s].sum() - llb[s].sum()) / max(pos[s].sum(), 1.0)
        if 0.0 < pos[s].sum() < nrow[s].sum():
            yy = np.concatenate([y_by[j] for j in s])
            d_pr[i] = (average_precision_score(yy, np.concatenate([a_by[j] for j in s]))
                       - average_precision_score(yy, np.concatenate([b_by[j] for j in s])))
    def ci(v):
        v = v[np.isfinite(v)]
        return [round(float(np.percentile(v, 2.5)), 6), round(float(np.percentile(v, 97.5)), 6)]
    return {"d_ig": {"point": round(float(information_gain(P_a, P_b, y)), 6), "ci95": ci(d_ig)},
            "d_pr_auc": {"point": round(float(average_precision_score(y, P_a)
                                              - average_precision_score(y, P_b)), 6),
                         "ci95": ci(d_pr)}}


def main():
    t0_all = time.time()
    out = {"governed_by": {"amendment": "docs/preregistration/v2_analysis_amendment_5.md",
                           "sha256": "c97db8f54374ac4ff1b5fbfafc1a1e76c63d68077144b338319603170ce846c2",
                           "item": "A"},
           "scope": ("tests the M>=3.5 conclusion under strict completeness; does NOT and CANNOT "
                     "validate the M>=3.0 primary target, which has no existence at this floor"),
           "omitted_and_disclosed": ("sv-ETAS and the independent inversion are not refit at "
                                     "Mc=3.5; this arm compares hybrid, cascade and "
                                     "first-generation ETAS only"),
           "base_mc": MC35}

    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])

    # ---- 1. refit ETAS at Mc = 3.5 ----
    print("[1/6] refitting ETAS at Mc=3.5 ...", flush=True)
    EF.BASE_MC = MC35
    cat_fit = cat[cat["datetime_utc"] < EF.FIT_END].copy()
    params, n_tgt, n_dropped, b_val, b_aki = EF.fit_stai(cat_fit, EF.MODEL_BOX, EF.FALLBACK_CAP)
    with open(R4 / "etas_params_mc35.pkl", "wb") as f:
        pickle.dump(params, f)
    fit = {"base_mc": MC35, "n_events_fit": int(n_tgt), "n_dropped_stai": int(n_dropped),
           "mu_total": float(params.mu_total), "k": float(params.k), "alpha": float(params.alpha),
           "c": float(params.c), "p": float(params.p), "d": float(params.d), "q": float(params.q),
           "gamma": float(params.gamma), "b_positive": float(b_val), "b_aki": float(b_aki),
           "branching_untrunc": float(branching_ratio(params)),
           "branching_mmax7.6": float(branching_ratio(params, mmax=7.6)),
           "converged": bool(getattr(params, "_fit_result").success)}
    out["etas_fit_mc35"] = fit
    print(f"      n_fit={n_tgt} k={params.k:.4f} alpha={params.alpha:.4f} p={params.p:.4f} "
          f"b_pos={b_val:.4f} branching={fit['branching_mmax7.6']:.4f} "
          f"({time.time()-t0_all:.0f}s)", flush=True)
    del cat_fit; gc.collect()

    # ---- 2. b_op sweep on PRE-TEST windows ----
    print("[2/6] b_op sweep (pre-test windows only) ...", flush=True)
    spec = G.MODEL_SPEC
    EV = G.build_event_bundle(cat, MC35)
    hist = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]
    starts = G.window_starts(cat["datetime_utc"].max())
    cal = [t for t in starts if t + pd.Timedelta(days=30) <= pd.Timestamp("2024-01-01")][::3]
    e35 = EV["e35"]
    sweep = {}
    for b in B_CANDIDATES:
        pred, real = [], []
        for i, t0 in enumerate(cal):
            t0d = float(G._to_days(t0))
            c = cascade_forecast(params, hist[cat["datetime_utc"] < t0], t0d, 30.0,
                                 spec.lon_c, spec.lat_c, K=K_CAL, seed=5000 + i, b=b)
            lo = np.searchsorted(e35["t"], t0d, "left")
            hi = np.searchsorted(e35["t"], t0d + 30.0, "left")
            pred.append(float(c["lam35"].sum())); real.append(float(hi - lo))
        pred = np.array(pred); real = np.array(real)
        slope = float((pred * real).sum() / max((pred * pred).sum(), 1e-12))
        sweep[b] = {"slope": round(slope, 4), "pred_total": round(float(pred.sum()), 1),
                    "real_total": round(float(real.sum()), 1)}
        print(f"      b={b:.3f}: slope {slope:.3f}  pred {pred.sum():.0f} real {real.sum():.0f} "
              f"({time.time()-t0_all:.0f}s)", flush=True)
    b_op35 = min(B_CANDIDATES, key=lambda b: abs(sweep[b]["slope"] - 1.0))
    out["b_op_sweep_mc35"] = {"candidates": {str(b): sweep[b] for b in B_CANDIDATES},
                              "selected_b_op": float(b_op35),
                              "shipped_b_op_at_mc3.0": 1.15,
                              "n_calibration_windows": len(cal)}
    print(f"      -> b_op = {b_op35} (shipped at Mc=3.0: 1.15)", flush=True)

    # ---- 3. rebuild the grid at Mc = 3.5 ----
    print("[3/6] rebuilding grid at Mc=3.5 ...", flush=True)
    ctx = G.build_static_context()
    LO, LA = np.meshgrid(G.LON_C, G.LAT_C)
    flat_lon, flat_lat = LO.ravel(), LA.ravel()
    ir_flat = np.repeat(np.arange(G.NLAT), G.NLON)
    ic_flat = np.tile(np.arange(G.NLON), G.NLAT)
    rows = []
    for k, t0_dt in enumerate(starts):
        t0d = float(G._to_days(t0_dt))
        feats = G.features_at_window(EV, t0d, t0_dt, ctx, params)
        c = cascade_forecast(params, hist[cat["datetime_utc"] < t0_dt], t0d, G.HORIZON_D,
                             spec.lon_c, spec.lat_c, K=K_BACKTEST, seed=1000 + k, b=b_op35,
                             preserve_branching=True)
        blk = {"window": np.full(G.NCELLS, k),
               "t0": np.full(G.NCELLS, np.datetime64(t0_dt), dtype="datetime64[ns]"),
               "cell_lon": flat_lon, "cell_lat": flat_lat, "ir": ir_flat, "ic": ic_flat}
        for nm in FEATURES:
            blk[nm] = feats[nm].ravel()
        blk["lam35_sim"] = c["lam35"].ravel()
        sub = EV["e35"]
        lo = np.searchsorted(sub["t"], t0d, "left"); hi = np.searchsorted(sub["t"], t0d + 30.0, "left")
        cg = np.zeros((G.NLAT, G.NLON)); np.add.at(cg, (sub["ir"][lo:hi], sub["ic"][lo:hi]), 1.0)
        blk["count35"] = cg.ravel()
        yg = np.zeros((G.NLAT, G.NLON)); yg[sub["ir"][lo:hi], sub["ic"][lo:hi]] = 1.0
        blk["y35"] = yg.ravel()
        rows.append(pd.DataFrame(blk))
        if (k + 1) % 60 == 0:
            print(f"      window {k+1}/{len(starts)} ({time.time()-t0_all:.0f}s)", flush=True)
    grid = pd.concat(rows, ignore_index=True)
    grid.to_parquet(R4 / "grid_mc35.parquet", index=False)
    del rows; gc.collect()
    m = split_masks(grid)
    out["grid_mc35"] = {"n_rows": int(len(grid)), "n_windows": int(grid["window"].nunique()),
                        "count35_total": int(grid["count35"].sum()),
                        "test_positives": int(grid.loc[m["test"], "y35"].sum()),
                        "test_windows": int(pd.Series(grid["window"].to_numpy()[m["test"]]).nunique()),
                        "shipped_test_positives_at_mc3.0": 167}
    print(f"      grid {len(grid)} rows, test positives {out['grid_mc35']['test_positives']} "
          f"(shipped at Mc=3.0: 167)", flush=True)

    # ---- 4. leakage self-test on the new grid ----
    print("[4/6] truncated-catalogue self-test on the Mc=3.5 grid ...", flush=True)
    in_split = m["train"] | m["val"] | m["test"]
    gs = grid[in_split]
    wins = np.sort(gs["window"].unique())
    cat_s = cat.sort_values("datetime_utc").reset_index(drop=True)
    cat_t = cat_s["datetime_utc"].to_numpy()
    max_int, max_real, nchk, nfail = 0.0, 0.0, 0, 0
    for w in wins:
        gw = gs[gs.window == w].sort_values(["ir", "ic"]).reset_index(drop=True)
        t0_dt = pd.Timestamp(gw["t0"].iloc[0]); t0d = float((t0_dt - G.REF) / pd.Timedelta(days=1))
        cut = int(np.searchsorted(cat_t, np.datetime64(t0_dt), side="left"))
        EVt = G.build_event_bundle(cat_s.iloc[:cut], MC35)
        f = G.features_at_window(EVt, t0d, t0_dt, ctx, params)
        flat = gw["ir"].to_numpy() * G.NLON + gw["ic"].to_numpy()
        for col in FEATURES:
            dev = np.abs(f[col].ravel()[flat] - gw[col].to_numpy(float))
            d = float(dev.max()) if dev.size else 0.0
            if col in INT_COLS:
                max_int = max(max_int, d); bad = (f[col].ravel()[flat] != gw[col].to_numpy(float))
            else:
                max_real = max(max_real, d)
                bad = ~np.isclose(f[col].ravel()[flat], gw[col].to_numpy(float), rtol=1e-6, atol=1e-6)
            nchk += len(gw); nfail += int(bad.sum())
    out["leakage_self_test_mc35"] = {"n_windows": int(len(wins)), "total_checks": int(nchk),
                                     "n_failures": int(nfail), "max_integer_deviation": max_int,
                                     "max_real_deviation": max_real,
                                     "passes": bool(nfail == 0 and max_int == 0 and max_real <= 1e-6)}
    print(f"      {nchk:,} checks, {nfail} failures, max_int {max_int:g}, max_real {max_real:g}",
          flush=True)

    # ---- 5. train booster + select w ----
    print("[5/6] training booster + 1-SE w-selection ...", flush=True)
    X = grid[FEATURES].copy()
    X["ln_lam_sim"] = np.log(grid["lam35_sim"].to_numpy() + EPS)
    n = grid["count35"].to_numpy().astype(float)
    y = grid["y35"].to_numpy().astype(float)
    lam_sim = np.clip(grid["lam35_sim"].to_numpy(), EPS, None)
    reg = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=400,
                                        max_depth=6, monotonic_cst=_monotonic(FEATURES + ["ln_lam_sim"]),
                                        random_state=42)
    reg.fit(X[m["train"]], n[m["train"]])
    lam_ml = np.clip(reg.predict(X), EPS, None)
    w, wdiag = select_w_1se(lam_sim, lam_ml, y, m["val"], grid["window"].to_numpy(), WEIGHTS)
    P_hyb = lambda_to_p(lam_sim ** (1 - w) * lam_ml ** w)
    P_casc = lambda_to_p(lam_sim)
    # first-generation ETAS at thr == mc == 3.5: the GR rescale is 10^0 = 1, so it IS etas_rate
    P_fg = lambda_to_p(np.clip(grid["etas_rate"].to_numpy(), EPS, None))
    out["ml_mc35"] = {"selected_w": float(w), "w_argmax": float(wdiag["w_argmax"]),
                      "n_trees_fit": int(reg.n_iter_), "shipped_w_at_mc3.0": 0.4,
                      "shipped_trees_at_mc3.0": 10}
    print(f"      w={w} (argmax {wdiag['w_argmax']}), trees {reg.n_iter_}", flush=True)

    # ---- 6. evaluate on the test windows ----
    print("[6/6] evaluating y35 on the test windows ...", flush=True)
    te = m["test"]
    yt = y[te]; wt = grid["window"].to_numpy()[te]
    preds = {"hybrid": P_hyb[te], "cascade": P_casc[te], "firstgen_etas": P_fg[te]}
    out["table1_style_rows"] = {
        k: {"pr_auc": round(float(average_precision_score(yt, v)), 6),
            "roc_auc": round(float(roc_auc_score(yt, v)), 6),
            "brier": round(float(brier_score_loss(yt, v)), 6)} for k, v in preds.items()}
    shipped = {"hybrid_vs_cascade": "inseparable", "hybrid_vs_firstgen_etas": "inseparable",
               "cascade_vs_firstgen_etas": "inseparable"}
    pairs = {}
    for a, b in (("hybrid", "cascade"), ("hybrid", "firstgen_etas"), ("cascade", "firstgen_etas")):
        st = paired(preds[a], preds[b], yt, wt)
        v = verdict_for(st["d_ig"]["ci95"], st["d_pr_auc"]["ci95"])
        st["verdict"] = {"A_beats_B": f"{a} beats {b}", "B_beats_A": f"{b} beats {a}",
                         "inseparable": "inseparable"}[v]
        pairs[f"{a}_vs_{b}"] = st
        print(f"      {a}_vs_{b:16s} dIG {st['d_ig']['point']:+.4f} {st['d_ig']['ci95']}  "
              f"dPR {st['d_pr_auc']['point']:+.4f} {st['d_pr_auc']['ci95']}  -> {st['verdict']}",
              flush=True)
    out["verdicts_mc35"] = pairs
    diffs = [f"{k}: {shipped[k]} -> {v['verdict']}" for k, v in pairs.items()
             if v["verdict"] != shipped.get(k)]
    out["verdict_diff_vs_shipped_table2"] = diffs
    out["any_verdict_changed"] = bool(diffs)
    out["runtime_s"] = round(time.time() - t0_all, 1)
    json.dump(out, open(R4 / "r12_item_A_mc35.json", "w"), indent=2)
    print(f"\nverdict changes vs shipped Table 2: {diffs if diffs else 'NONE'}")
    print(f"runtime {out['runtime_s']}s -> results/round4/r12_item_A_mc35.json")


if __name__ == "__main__":
    main()
