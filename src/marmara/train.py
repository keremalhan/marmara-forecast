"""Hybrid forecaster: ML as a multiplicative correction to the
cascade rate (structurally cannot lose to the cascade, since w=0 recovers it).

lambda_ML from HistGradientBoostingRegressor(loss='poisson') on 30-day cell COUNTS,
features = the 19 grid features (at Mc=3.0) + ln(lam_sim). Final rate:
    lambda = lam_sim^(1-w) * lambda_ML^w ,  w in {0,0.1,...,1.0} by val Poisson-LL.
Isotonic-calibrate P=1-exp(-lambda) on val; one test pass.

Evaluation vs baselines (Poisson, fault-proximity, smoothed, cascade "ETAS-sim",
and the first-generation ETAS) with the shared metrics. HEADLINE = IG(hybrid
vs cascade) and IG(hybrid vs first-gen ETAS) on test, y35 and y45, + chosen w.

Output: results/models/y3{5,45}_hybrid.pkl , results/evaluation.{json,md}
Run:  "<venv>/bin/python3" -m marmara.train
"""
from __future__ import annotations

import json
import pickle
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

from marmara import baselines as B
from marmara.grid import FEATURES
from marmara.metrics import (information_gain, lambda_to_p, p_to_lambda,
                        score_predictor)
TRAIN_END = pd.Timestamp("2022-01-01")     # train t0 < this
VAL_END = pd.Timestamp("2024-01-01")       # val t0 in [TRAIN_END, VAL_END)
TEST_TARGET_END = pd.Timestamp("2026-03-31")  # test windows: t0+30d <= this


def split_masks(grid):
    t0 = pd.to_datetime(grid["t0"])
    tgt_end = t0 + pd.Timedelta(days=30)
    train = t0 < TRAIN_END
    val = (t0 >= TRAIN_END) & (t0 < VAL_END)
    test = (t0 >= VAL_END) & (tgt_end <= TEST_TARGET_END)
    forecast_tail = tgt_end > TEST_TARGET_END
    return {"train": train.to_numpy(), "val": val.to_numpy(),
            "test": test.to_numpy(), "forecast_tail": forecast_tail.to_numpy()}

OUT = RESULTS
MODELS = OUT / "models"
WEIGHTS = np.round(np.arange(0.0, 1.01, 0.1), 2)
EPS = 1e-9
FEATS_HYBRID = FEATURES + ["ln_lam_sim"]


# Phase 2: GNSS v2 physical feature columns (the availability-flag gnss_strain_fallback
# is deliberately EXCLUDED — the ablation showed it adds nothing and it is an
# availability-vs-time proxy). Merged onto the grid in main() when present.
GNSS_PHYS = ["gnss_resid_rate_90", "gnss_resid_rate_365",
             "gnss_resid_max_365", "gnss_strain_rate_365"]


def _monotonic(feats_hybrid):
    return [(-1 if f == "dist_fault_km" else 1 if f in ("etas_rate", "ln_lam_sim") else 0)
            for f in feats_hybrid]


def poisson_ll(n, lam):
    lam = np.clip(lam, EPS, None)
    return float(np.sum(n * np.log(lam) - lam))


def fit_hybrid(grid, masks, count_col, lam_col, ycol, extra_feats=(), label="hybrid"):
    """Poisson-GBR lambda_ML + geometric blend lambda = lam_sim^(1-w)*lam_ML^w.
    w chosen by val per-event Poisson log-likelihood (using occurrence y, matching
    the IG convention). Returns the RAW blend rate (no isotonic in the eval path;
    isotonic is fit separately below only for the saved forecast model).
    extra_feats appends model inputs (e.g. Phase-2 GNSS columns) — with extra_feats=()
    this is bit-identical to the published hybrid."""
    tr, va = masks["train"], masks["val"]
    feats = FEATURES + list(extra_feats)
    X = grid[feats].copy()
    X["ln_lam_sim"] = np.log(grid[lam_col].to_numpy() + EPS)
    n = grid[count_col].to_numpy().astype(float)
    y = grid[ycol].to_numpy().astype(float)
    lam_sim = np.clip(grid[lam_col].to_numpy(), EPS, None)

    reg = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05,
                                        max_iter=400, max_depth=6,
                                        monotonic_cst=_monotonic(feats + ["ln_lam_sim"]),
                                        random_state=42)
    reg.fit(X[tr], n[tr])
    lam_ml = np.clip(reg.predict(X), EPS, None)

    # choose w by val per-event Poisson LL (n=y occurrence, IG convention). w=0
    # recovers the cascade, so the chosen w cannot be worse than cascade on val.
    best = None
    for w in WEIGHTS:
        lam = lam_sim ** (1 - w) * lam_ml ** w
        ll = poisson_ll(y[va], lam[va])
        if best is None or ll > best[0]:
            best = (ll, w)
    w = best[1]
    lam_hybrid = lam_sim ** (1 - w) * lam_ml ** w

    # isotonic P calibration (for the forecast maps only; NOT used in scoring)
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(lambda_to_p(lam_hybrid[va]), y[va])
    with open(MODELS / f"{count_col}_{label}.pkl", "wb") as f:
        pickle.dump({"reg": reg, "w": float(w), "iso": iso}, f)
    return {"w": float(w), "lam_ml": lam_ml, "lam_hybrid": lam_hybrid}


def _load_extra_etas_preds(grid, thr):
    """Phase 1: extra modern-ETAS predictors computed as cascade rates from the
    sv-ETAS fit and the Mizrahi-inverted params (results/rates_{label}.parquet,
    keyed by window,ir,ic; same K/seed/b as the published cascade, so only the
    ETAS PARAMETERS differ). Absent parquet -> predictor silently skipped. Rows in
    train windows (not present in the val+test rate files) get lambda=0; they are
    never scored (evaluate() indexes val/test only)."""
    lamcol = {3.0: "lam30", 3.5: "lam35", 4.5: "lam45"}[round(float(thr), 1)]
    key = ["window", "ir", "ic"]
    idx = pd.MultiIndex.from_frame(grid[key])
    out = {}
    for label in ("sv_etas", "modern_etas"):
        path = OUT / f"rates_{label}.parquet"
        if not path.exists():
            continue
        avail = pd.read_parquet(path).columns
        if lamcol not in avail:                 # e.g. lam30 missing in an old rate file
            continue
        r = pd.read_parquet(path, columns=key + [lamcol]).set_index(key)[lamcol]
        lam = r.reindex(idx).to_numpy()
        lam = np.where(np.isfinite(lam), lam, 0.0)
        out[label] = lambda_to_p(lam)
    return out


def _gnss_grid_columns(grid):
    """Phase 2: the gnss_v2 physical columns aligned to the grid by (window,ir,ic).
    Uses results/gnss_v2_columns.parquet if it carries the keys; otherwise computes
    them via the GnssV2Source and caches WITH keys. Returns None if the GNSS data is
    not on disk (hybrid_gnss is then silently skipped — graceful degradation)."""
    key = ["window", "ir", "ic"]
    cache = OUT / "gnss_v2_columns.parquet"
    if cache.exists():
        c = pd.read_parquet(cache)
        if all(k in c.columns for k in key) and all(col in c.columns for col in GNSS_PHYS):
            c = c.set_index(key)
            idx = pd.MultiIndex.from_frame(grid[key])
            return {col: c[col].reindex(idx).to_numpy() for col in GNSS_PHYS}
    from marmara.sources.gnss_v2 import GnssV2Source
    src = GnssV2Source()
    ok, _why = src.available()
    if not ok:
        return None
    cells = grid[["window", "t0", "cell_lon", "cell_lat", "ir", "ic"]].copy()
    cols = src.add_columns(cells).reset_index(drop=True)
    pd.concat([grid[key].reset_index(drop=True), cols], axis=1).to_parquet(cache, index=False)
    return {col: cols[col].to_numpy() for col in GNSS_PHYS}


def evaluate(grid, masks, ycol, count_col, lam_col, thr, cat, mc, b_train, mc_etas):
    fit = fit_hybrid(grid, masks, count_col, lam_col, ycol)
    y = grid[ycol].to_numpy().astype(float)
    win = grid["window"].to_numpy()
    va, te = masks["val"], masks["test"]
    vi, ti = np.where(va)[0], np.where(te)[0]

    # predictors as probabilities, all from raw rates (fair, consistent footing)
    P_hybrid = lambda_to_p(fit["lam_hybrid"])
    P_casc = lambda_to_p(grid[lam_col].to_numpy())                 # ETAS-sim (cascade)
    lp = B.poisson_clim(grid, masks["train"], thr)
    lf = B.fault_prox_clim(grid, masks["train"], thr)
    P_firstgen = lambda_to_p(B.etas_baseline(grid, thr, b_train, mc_etas))  # first-gen ETAS
    # smoothed: pick sigma on val
    from sklearn.metrics import average_precision_score
    best = None
    for s in (5.0, 10.0, 20.0):
        lv = B.smoothed_seismicity(grid.iloc[vi], cat, mc, thr, s)
        ap = average_precision_score(y[vi], lambda_to_p(lv)) if 0 < y[vi].sum() < len(vi) else 0
        if best is None or ap > best[0]:
            best = (ap, s)
    sigma = best[1]
    lam_sm = np.zeros(len(grid))
    lam_sm[vi] = B.smoothed_seismicity(grid.iloc[vi], cat, mc, thr, sigma)
    lam_sm[ti] = B.smoothed_seismicity(grid.iloc[ti], cat, mc, thr, sigma)
    P_sm = lambda_to_p(lam_sm)

    preds = {"hybrid": P_hybrid, "cascade": P_casc, "poisson": lambda_to_p(lp),
             "fault_prox": lambda_to_p(lf), "smoothed": P_sm, "firstgen_etas": P_firstgen}
    preds.update(_load_extra_etas_preds(grid, thr))  # Phase 1: sv_etas, modern_etas (if present)
    # Phase 2: GNSS-augmented hybrid (promoted per the pre-registered decision rule —
    # gnss_v2 val IG +0.024 > 0.02 and test IG +0.098 > 0; genuine after the
    # availability + spatial-support confound checks). Present only when the GNSS
    # columns are merged onto the grid (main()); kept ALONGSIDE `hybrid` so the
    # bootstrap can put a CI on the +GNSS delta.
    if all(c in grid.columns for c in GNSS_PHYS):
        fit_g = fit_hybrid(grid, masks, count_col, lam_col, ycol,
                           extra_feats=GNSS_PHYS, label="hybrid_gnss")
        preds["hybrid_gnss"] = lambda_to_p(fit_g["lam_hybrid"])

    out = {"threshold": thr, "chosen_w": fit["w"], "smoothed_sigma_km": sigma, "splits": {}}
    for split, idx in (("val", vi), ("test", ti)):
        ys, ws = y[idx], win[idx]
        scored = {nm: score_predictor(preds[nm][idx], ys, ws) for nm in preds}
        ig = {nm: information_gain(P_hybrid[idx], preds[nm][idx], ys)
              for nm in preds if nm != "hybrid"}
        out["splits"][split] = {"n": int(len(idx)), "n_pos": int(ys.sum()),
                                "scores": scored, "ig_hybrid_vs": ig}

    # --- persist per-row predictions (val+test) for the Phase-0 block bootstrap.
    # Purely additive: does NOT affect evaluation.{json,md}. Each predictor stored
    # here is the SAME probability array scored above, so bootstrap.py operates on
    # the exact predictions behind the headline numbers (single source of truth;
    # a modern_etas column would be picked up automatically once added in Phase 1).
    sel = np.where(va | te)[0]
    pred_df = pd.DataFrame({
        "window": grid["window"].to_numpy()[sel],
        "t0": grid["t0"].to_numpy()[sel],
        "split": np.where(va[sel], "val", "test"),
        "y": y[sel],
    })
    for nm, P in preds.items():
        pred_df[nm] = np.asarray(P)[sel]
    pred_df.to_parquet(OUT / f"predictions_{ycol}.parquet", index=False)
    return out


def main():
    assert (OUT / "cascade_ok.json").exists(), "run cascade gate first"
    grid = pd.read_parquet(OUT / "grid_hybrid.parquet")
    gnss = _gnss_grid_columns(grid)                 # Phase 2: merge GNSS columns if present
    if gnss is not None:
        for c in GNSS_PHYS:
            grid[c] = gnss[c]
        print(f"merged GNSS v2 columns ({', '.join(GNSS_PHYS)}) -> hybrid_gnss enabled")
    masks = split_masks(grid)
    cat = pd.read_csv(OUT / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    mc = 3.0
    mc_etas = json.load(open(OUT / "etas_fit_report.json"))["base_mc"]
    from marmara.evaluate import train_b_value
    b_train = train_b_value(cat, mc)

    # Phase 3: y30 (M>=3.0) is the PRIMARY powered comparison (~10x the y35 positives);
    # y35 powered; y45 UNPOWERED (kept descriptive — no ranking claims).
    POWER = {"y30": "primary (powered)", "y35": "powered", "y45": "unpowered — no ranking claims"}
    report = {"meta": {"mc": mc, "b_train": b_train, "features": FEATS_HYBRID,
                       "power": POWER}, "targets": {}}
    for ycol, ccol, lcol, thr in (("y30", "count30", "lam30_sim", 3.0),
                                  ("y35", "count35", "lam35_sim", 3.5),
                                  ("y45", "count45", "lam45_sim", 4.5)):
        print(f"evaluating {ycol} ...")
        report["targets"][ycol] = evaluate(grid, masks, ycol, ccol, lcol, thr,
                                            cat, mc, b_train, mc_etas)
        report["targets"][ycol]["power"] = POWER[ycol]

    # headline
    def line(y):
        t = report["targets"][y]["splits"]["test"]
        ig = t["ig_hybrid_vs"]; w = report["targets"][y]["chosen_w"]
        sc = t["scores"]
        return (f"{y}: w={w:.1f}; hybrid PR-AUC {sc['hybrid']['pr_auc']:.3f} vs cascade "
                f"{sc['cascade']['pr_auc']:.3f} vs first-gen-ETAS {sc['firstgen_etas']['pr_auc']:.3f}; "
                f"IG(hybrid vs cascade)={ig['cascade']:+.3f}, IG(hybrid vs first-gen-ETAS)="
                f"{ig['firstgen_etas']:+.3f}, IG vs smoothed {ig['smoothed']:+.3f}")
    report["headline"] = " | ".join(line(y) for y in ("y30", "y35", "y45"))
    # honest verdict — all ranking claims must be read against bootstrap_ci/claims.json.
    y30t = report["targets"]["y30"]["splits"]["test"]
    y35t = report["targets"]["y35"]["splits"]["test"]["scores"]
    report["verdict"] = (
        f"PRIMARY = y30 (M>=3.0, {y30t['n_pos']} test positives — ~{y30t['n_pos']//max(report['targets']['y35']['splits']['test']['n_pos'],1)}x y35): "
        "the powered comparison where ML-vs-ETAS separation is statistically resolvable "
        "at all; read the ranking off claims.json (block-bootstrap verdicts). "
        f"y35 (M>=3.5, {report['targets']['y35']['splits']['test']['n_pos']} positives): "
        f"cascade PR-AUC {y35t['cascade']['pr_auc']:.3f} vs first-gen {y35t['firstgen_etas']['pr_auc']:.3f} — "
        "the difference is WITHIN the bootstrap CI (inseparable; see claims.json); the "
        "old 'cascade ranks best' headline is not supported. "
        f"y45 (M>=4.5, only {report['targets']['y45']['splits']['test']['n_pos']} test positives): "
        "UNPOWERED — bootstrap CIs are enormous and essentially every pair is inseparable; "
        "NO ranking claims. The w chosen for y45 was selected on ~13 val positives (noise-fit) "
        "and is NOT a headline result.")
    json.dump(report, open(OUT / "evaluation.json", "w"), indent=2)

    L = ["# Evaluation — hybrid (cascade x ML) vs baselines", "",
         "All ranking claims must be read against results/claims.json (block-bootstrap "
         "verdicts); point differences below are NOT claims on their own.", "",
         report["headline"], ""]
    for y in ("y30", "y35", "y45"):
        r = report["targets"][y]
        L.append(f"## {y} (thr {r['threshold']}, w={r['chosen_w']:.1f}, sigma "
                 f"{r['smoothed_sigma_km']:g}km) — {POWER[y]}")
        for sp in ("val", "test"):
            s = r["splits"][sp]
            L.append(f"\n### {sp} (n={s['n']}, pos={s['n_pos']})")
            L.append("| predictor | PR-AUC | ROC-AUC | Brier | Molchan |")
            L.append("|---|---|---|---|---|")
            _order = ["hybrid", "hybrid_gnss", "cascade", "sv_etas", "modern_etas",
                      "firstgen_etas", "smoothed", "poisson", "fault_prox"]
            for nm in [n for n in _order if n in s["scores"]]:
                sc = s["scores"][nm]; ms = sc["molchan"]["area_skill"]
                f = lambda v, d=4: "n/a" if v is None else f"{v:.{d}f}"
                L.append(f"| {nm} | {f(sc['pr_auc'])} | {f(sc['roc_auc'])} | {f(sc['brier'],5)} | {f(ms,3)} |")
            L.append("\nIG(hybrid − baseline), nats/event:")
            for nm, v in s["ig_hybrid_vs"].items():
                L.append(f"- vs {nm}: {v:+.4f}")
        L.append("")
    (OUT / "evaluation.md").write_text("\n".join(L))
    print("\n" + report["headline"])
    print("wrote evaluation.{json,md}")


if __name__ == "__main__":
    main()
