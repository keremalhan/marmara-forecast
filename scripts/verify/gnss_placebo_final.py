"""Run 1 — GNSS placebo battery re-run at the FINAL configuration.

WHY. The §4 / Figure 4 placebo numbers were produced at commit 1205e7b (v3/phase-V,
2026-07-08), when grid_hybrid_report.json recorded b_op = 1.2, 261 windows, and
cascade.py had NO branching-ratio-preserving rescale (`preserve_branching` was added
later). The final configuration is b_op = 1.15, 262 windows, preserve_branching=True.
The stale battery also selected the blend weight by naive argmax; the published hybrid
uses the pre-registered 1-SE parsimony gate (train.select_w_1se). Every placebo number
in §4 is therefore off-configuration on three axes at once.

WHAT. Re-runs the entire battery through the CURRENT machinery:
  * grid_hybrid.parquet at b_op = 1.15 with the corrected (subcritical) branching,
  * fit_hybrid replicated bit-for-bit INCLUDING select_w_1se (verified against the
    frozen predictions_*.parquet before any placebo is trusted -- see verify_identity),
  * full retraining for every surrogate draw,
  * three surrogates: time-shuffle, circular-shift (>= 2 yr), coverage-only,
  * paired block-bootstrap CIs (Politis-Romano, B=2000, seed=42, mean_block=3.0) --
    the same machinery as marmara.bootstrap, so the CIs are paper-matched.

Targets: y30 (primary operational, 592 positives) and y35 (channel-level, the axis
§4 quotes as "+0.098 nats").

Reads only; writes results/round4/r1_gnss_placebo_final.json. Does NOT overwrite any
frozen artifact (no model pickle is written -- fit_hybrid's pickle dump is omitted).

Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/verify/gnss_placebo_final.py
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from marmara.paths import RESULTS
from marmara.grid import FEATURES
from marmara.metrics import information_gain, lambda_to_p, p_to_lambda
from marmara.train import GNSS_PHYS, WEIGHTS, _monotonic, select_w_1se, split_masks
from marmara.bootstrap import stationary_window_indices

EPS = 1e-9
OUT = RESULTS / "round4"
OUT.mkdir(exist_ok=True)

B_BOOT = 2000
SEED = 42
MEAN_BLOCK = 3.0
SHIFT_MIN_WIN = 24          # >= 2 yr at 30-day windows (unchanged protocol)

# N_PERM held at the ORIGINAL protocol values so the re-run is apples-to-apples with
# the stale battery: only the configuration changes, not the test.
N_PERM = {"y30": 25, "y35": 30}

TARGETS = {
    "y30": {"count": "count30", "lam": "lam30_sim", "y": "y30"},
    "y35": {"count": "count35", "lam": "lam35_sim", "y": "y35"},
}


# --------------------------------------------------------------------------- #
# fit_hybrid replica -- bit-identical to marmara.train.fit_hybrid MINUS the pickle
# dump (which would clobber the frozen results/models/*.pkl).
# --------------------------------------------------------------------------- #
def fit_hybrid_local(grid, masks, count_col, lam_col, ycol, extra_feats=()):
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
    w, wdiag = select_w_1se(lam_sim, lam_ml, y, va, grid["window"].to_numpy(), WEIGHTS)
    lam_hybrid = lam_sim ** (1 - w) * lam_ml ** w
    return {"w": float(w), "w_argmax": wdiag["w_argmax"], "lam_hybrid": lam_hybrid,
            "lam_ml": lam_ml}


def load_grid():
    g = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    gc = pd.read_parquet(RESULTS / "channels" / "gnss_traj_columns.parquet").set_index(["window", "ir", "ic"])
    idx = pd.MultiIndex.from_frame(g[["window", "ir", "ic"]])
    for c in GNSS_PHYS:
        g[c] = gc[c].reindex(idx).to_numpy()
    return g, split_masks(g)


# --------------------------------------------------------------------------- #
# Identity gate: the replica must reproduce the FROZEN predictions bit-for-bit.
# If it does not, every downstream placebo number is meaningless -> hard abort.
# --------------------------------------------------------------------------- #
def verify_identity(g, m, target):
    spec = TARGETS[target]
    frozen = pd.read_parquet(RESULTS / f"predictions_{target}.parquet")
    sel = np.where(m["val"] | m["test"])[0]
    out = {}
    for label, extra in (("hybrid", ()), ("hybrid_gnss", tuple(GNSS_PHYS))):
        if label not in frozen.columns:
            out[label] = {"present_in_frozen": False}
            continue
        fit = fit_hybrid_local(g, m, spec["count"], spec["lam"], spec["y"], extra_feats=extra)
        P = lambda_to_p(fit["lam_hybrid"])[sel]
        Pf = frozen[label].to_numpy(float)
        dev = float(np.abs(P - Pf).max())
        out[label] = {"present_in_frozen": True, "w_replica": fit["w"],
                      "max_abs_dev_vs_frozen": dev, "identical": bool(dev < 1e-12)}
    return out


# --------------------------------------------------------------------------- #
# Paired block-bootstrap CI of (aug - base) for dIG and dPR-AUC, on the test split.
# Mirrors marmara.bootstrap.bootstrap_split/pair_stats exactly.
# --------------------------------------------------------------------------- #
def paired_boot(P_aug, P_base, y, win, te, b_boot=B_BOOT):
    ya, pa, pb, wa = y[te], P_aug[te], P_base[te], win[te]
    wins = np.sort(np.unique(wa))
    y_by, pa_by, pb_by = [], [], []
    lla, llb, pos, nrows = [], [], [], []
    for w in wins:
        k = wa == w
        yv = ya[k]
        y_by.append(yv); pa_by.append(pa[k]); pb_by.append(pb[k])
        pos.append(float(yv.sum())); nrows.append(float(k.sum()))
        for P, acc in ((pa[k], lla), (pb[k], llb)):
            lam = np.clip(p_to_lambda(np.clip(P, 0.0, 1.0)), EPS, None)
            acc.append(float(np.sum(yv * np.log(lam) - lam)))
    lla, llb = np.array(lla), np.array(llb)
    pos, nrows = np.array(pos), np.array(nrows)
    rng = np.random.default_rng(SEED)
    seqs = stationary_window_indices(len(wins), b_boot, MEAN_BLOCK, rng)
    d_ig = np.empty(b_boot); d_pr = np.full(b_boot, np.nan)
    for b in range(b_boot):
        s = seqs[b]
        nev = max(pos[s].sum(), 1.0)
        d_ig[b] = (lla[s].sum() - llb[s].sum()) / nev
        if 0.0 < pos[s].sum() < nrows[s].sum():
            yp = np.concatenate([y_by[i] for i in s])
            d_pr[b] = (average_precision_score(yp, np.concatenate([pa_by[i] for i in s]))
                       - average_precision_score(yp, np.concatenate([pb_by[i] for i in s])))
    def ci(v):
        v = v[np.isfinite(v)]
        return [round(float(np.percentile(v, 2.5)), 6), round(float(np.percentile(v, 97.5)), 6)]
    ig_pt = float(information_gain(pa, pb, ya))
    pr_pt = float(average_precision_score(ya, pa) - average_precision_score(ya, pb))
    return {
        "d_ig": {"point": round(ig_pt, 6), "ci95": ci(d_ig),
                 "excludes_0": bool(ci(d_ig)[0] > 0 or ci(d_ig)[1] < 0)},
        "d_pr_auc": {"point": round(pr_pt, 6), "ci95": ci(d_pr),
                     "excludes_0": bool(ci(d_pr)[0] > 0 or ci(d_pr)[1] < 0)},
        "n_windows": int(len(wins)), "n_pos": int(ya.sum()),
    }


def single_window_share(P_aug, P_base, y, win, te):
    """Share of the total (aug - base) test log-likelihood gain carried by the single
    largest-contributing window. The stale battery reported ~60% for y30."""
    ya, wa = y[te], win[te]
    contrib = {}
    for w in np.unique(wa):
        k = wa == w
        d = 0.0
        for P, sgn in ((P_aug[te][k], 1.0), (P_base[te][k], -1.0)):
            lam = np.clip(p_to_lambda(np.clip(P, 0.0, 1.0)), EPS, None)
            d += sgn * float(np.sum(ya[k] * np.log(lam) - lam))
        contrib[int(w)] = d
    tot = sum(contrib.values())
    top_w = max(contrib, key=lambda k: abs(contrib[k]))
    order = sorted(contrib.items(), key=lambda kv: -abs(kv[1]))
    return {
        "total_ll_gain": round(tot, 6),
        "total_ig_nats_per_pos": round(tot / max(float(ya.sum()), 1.0), 6),
        "top_window": top_w,
        "top_window_ll_gain": round(contrib[top_w], 6),
        "top_window_share_of_total": (round(contrib[top_w] / tot, 6) if abs(tot) > 1e-12 else None),
        "top3_windows": [{"window": int(w), "ll_gain": round(v, 6),
                          "share": (round(v / tot, 6) if abs(tot) > 1e-12 else None)}
                         for w, v in order[:3]],
    }


def table1_row(P, y, te):
    ya, pa = y[te], np.clip(P[te], 0.0, 1.0)
    return {"pr_auc": round(float(average_precision_score(ya, pa)), 6),
            "roc_auc": round(float(roc_auc_score(ya, pa)), 6),
            "brier": round(float(brier_score_loss(ya, pa)), 6)}


# --------------------------------------------------------------------------- #
# Surrogates
# --------------------------------------------------------------------------- #
def coverage_feature(g):
    """Per (window,cell): count of GNSS stations with >=5 epochs in [t0-365,t0) within 60 km."""
    from marmara.sources.gnss_traj import GnssTrajSource, KM_PER_DEG_LAT, R_KM, _decimal_year
    src = GnssTrajSource()
    ok, why = src.available()
    if not ok:
        return None, why
    series, pos, _ = src._load()
    names = list(series)
    slon = np.array([pos[s][0] for s in names]); slat = np.array([pos[s][1] for s in names])
    cov = np.zeros(len(g))
    for w, sub in g.groupby("window"):
        t0 = pd.Timestamp(sub["t0"].iloc[0]); t0dy = _decimal_year(t0)
        lo = _decimal_year(t0 - pd.Timedelta(days=365))
        active = np.array([((series[s]["yr"] >= lo) & (series[s]["yr"] < t0dy)).sum() >= 5
                           for s in names])
        clon = sub["cell_lon"].to_numpy(); clat = sub["cell_lat"].to_numpy()
        coslat = np.cos(np.radians(float(slat.mean())))
        d = np.sqrt(((clon[:, None] - slon[None, :]) * KM_PER_DEG_LAT * coslat) ** 2
                    + ((clat[:, None] - slat[None, :]) * KM_PER_DEG_LAT) ** 2)
        cov[sub.index.to_numpy()] = ((d <= R_KM) & active[None, :]).sum(axis=1)
    return cov, ""


def run_target(g, m, target, log):
    spec = TARGETS[target]
    te = m["test"]
    y = g[spec["y"]].to_numpy().astype(float)
    win = g["window"].to_numpy()
    res = {"target": target, "n_test_pos": int(y[te].sum()),
           "n_test_rows": int(te.sum()), "n_perm": N_PERM[target]}

    log(f"[{target}] base hybrid ...")
    t = time.time()
    fit_b = fit_hybrid_local(g, m, spec["count"], spec["lam"], spec["y"])
    P_base = lambda_to_p(fit_b["lam_hybrid"])
    log(f"[{target}] base hybrid w={fit_b['w']} ({time.time()-t:.1f}s)")

    log(f"[{target}] augmented hybrid ...")
    fit_a = fit_hybrid_local(g, m, spec["count"], spec["lam"], spec["y"], extra_feats=GNSS_PHYS)
    P_aug = lambda_to_p(fit_a["lam_hybrid"])
    log(f"[{target}] aug hybrid w={fit_a['w']}")

    res["w_base"] = fit_b["w"]; res["w_aug"] = fit_a["w"]
    res["w_base_argmax"] = fit_b["w_argmax"]; res["w_aug_argmax"] = fit_a["w_argmax"]
    res["real"] = paired_boot(P_aug, P_base, y, win, te)
    res["single_window"] = single_window_share(P_aug, P_base, y, win, te)
    res["table1_row_hybrid_gnss"] = table1_row(P_aug, y, te)
    res["table1_row_hybrid"] = table1_row(P_base, y, te)
    log(f"[{target}] REAL dIG={res['real']['d_ig']['point']:+.4f} "
        f"{res['real']['d_ig']['ci95']}  dPR={res['real']['d_pr_auc']['point']:+.4f} "
        f"{res['real']['d_pr_auc']['ci95']}")

    # ---- surrogates ----
    ck = g["ir"].to_numpy() * 1000 + g["ic"].to_numpy()
    order = {k: np.where(ck == k)[0][np.argsort(win[ck == k])] for k in np.unique(ck)}
    prng = np.random.default_rng(SEED)

    def surrogate(kind):
        digs, dprs = [], []
        t0 = time.time()
        for i in range(N_PERM[target]):
            gp = g.copy()
            arr = {c: g[c].to_numpy().copy() for c in GNSS_PHYS}
            for k, idx in order.items():
                nrow = len(idx)
                if kind == "time_shuffle":
                    perm = prng.permutation(nrow)
                else:
                    off = int(prng.integers(SHIFT_MIN_WIN, max(SHIFT_MIN_WIN + 1, nrow)))
                    perm = (np.arange(nrow) + off) % nrow
                for c in GNSS_PHYS:
                    arr[c][idx] = arr[c][idx][perm]
            for c in GNSS_PHYS:
                gp[c] = arr[c]
            f = fit_hybrid_local(gp, m, spec["count"], spec["lam"], spec["y"],
                                 extra_feats=GNSS_PHYS)
            Pp = lambda_to_p(f["lam_hybrid"])
            digs.append(float(information_gain(Pp[te], P_base[te], y[te])))
            dprs.append(float(average_precision_score(y[te], Pp[te])
                              - average_precision_score(y[te], P_base[te])))
            if (i + 1) % 5 == 0:
                log(f"[{target}/{kind}] {i+1}/{N_PERM[target]} ({time.time()-t0:.0f}s)")
        digs, dprs = np.array(digs), np.array(dprs)
        def band(v):
            return {"mean": round(float(v.mean()), 6),
                    "null_band_95": [round(float(np.percentile(v, 2.5)), 6),
                                     round(float(np.percentile(v, 97.5)), 6)],
                    "min": round(float(v.min()), 6), "max": round(float(v.max()), 6)}
        return {"d_ig": band(digs), "d_pr_auc": band(dprs),
                "real_dig_exceeds_null_p95": bool(res["real"]["d_ig"]["point"]
                                                  > np.percentile(digs, 97.5)),
                "real_dpr_exceeds_null_p95": bool(res["real"]["d_pr_auc"]["point"]
                                                  > np.percentile(dprs, 97.5)),
                "real_dig_exceeds_all": bool(res["real"]["d_ig"]["point"] > digs.max()),
                "runtime_s": round(time.time() - t0, 1)}

    for kind in ("time_shuffle", "circular_shift"):
        log(f"[{target}] surrogate {kind} x{N_PERM[target]} ...")
        res[kind] = surrogate(kind)
        log(f"[{target}/{kind}] dIG null {res[kind]['d_ig']['null_band_95']} "
            f"mean {res[kind]['d_ig']['mean']:+.4f}")

    # coverage-only: physical GNSS features replaced by station-availability count
    log(f"[{target}] surrogate coverage_only ...")
    cov, why = coverage_feature(g)
    if cov is None:
        res["coverage_only"] = {"skipped": why}
    else:
        gc = g.copy(); gc["gnss_coverage"] = cov
        f = fit_hybrid_local(gc, m, spec["count"], spec["lam"], spec["y"],
                             extra_feats=["gnss_coverage"])
        Pc = lambda_to_p(f["lam_hybrid"])
        c_ig = float(information_gain(Pc[te], P_base[te], y[te]))
        c_pr = float(average_precision_score(y[te], Pc[te])
                     - average_precision_score(y[te], P_base[te]))
        real_ig = res["real"]["d_ig"]["point"]
        res["coverage_only"] = {
            "d_ig": round(c_ig, 6), "d_pr_auc": round(c_pr, 6), "w": f["w"],
            "n_stations_min": int(cov.min()), "n_stations_max": int(cov.max()),
            "reproduces_gain": bool(abs(real_ig) > 1e-12 and c_ig > 0.5 * real_ig),
        }
        log(f"[{target}/coverage_only] dIG={c_ig:+.4f} dPR={c_pr:+.4f}")

    # ---- verdict under the placebo rule ----
    surv = []
    for kind in ("time_shuffle", "circular_shift"):
        surv.append(res[kind]["real_dig_exceeds_null_p95"] and res[kind]["real_dpr_exceeds_null_p95"])
    cov_ok = not res.get("coverage_only", {}).get("reproduces_gain", False)
    res["verdict"] = ("GENUINE: real exceeds every surrogate null on both axes and coverage "
                      "does not reproduce it") if (all(surv) and cov_ok) else \
                     ("NULL: the real effect does not clear the surrogate nulls on both axes "
                      "-> GNSS channel remains VOID")
    return res


def main():
    t_all = time.time()
    lines = []
    def log(s):
        print(s, flush=True); lines.append(s)

    g, m = load_grid()
    log(f"grid_hybrid: {len(g)} rows, {g['window'].nunique()} windows")
    cfg = json.load(open(RESULTS / "grid" / "grid_hybrid_report.json"))
    log(f"config: b_op={cfg['b_op']} n_rows={cfg['n_rows']} n_windows={cfg['n_windows']}")

    out = {"config": {"b_op": cfg["b_op"], "n_rows": cfg["n_rows"],
                      "n_windows": cfg["n_windows"], "K_backtest": cfg["K_backtest"]},
           "bootstrap": {"B": B_BOOT, "seed": SEED, "mean_block": MEAN_BLOCK,
                         "block": "Politis-Romano stationary bootstrap over window-ids"},
           "protocol_note": ("N_PERM held at the original per-target values (y30=25, y35=30) "
                             "so the re-run is apples-to-apples with the stale battery; only "
                             "the CONFIGURATION changes."),
           "targets": {}}

    # identity gate FIRST -- abort if the replica is not bit-identical to the frozen preds
    log("=== identity gate: replica vs frozen predictions ===")
    ident = {}
    for tgt in TARGETS:
        ident[tgt] = verify_identity(g, m, tgt)
        for lab, d in ident[tgt].items():
            log(f"  {tgt}/{lab}: {d}")
    out["identity_gate"] = ident
    bad = [f"{t}/{l}" for t, dd in ident.items() for l, d in dd.items()
           if d.get("present_in_frozen") and not d.get("identical")]
    if bad:
        out["identity_gate_status"] = f"FAIL: {bad}"
        json.dump(out, open(OUT / "r1_gnss_placebo_final.json", "w"), indent=2)
        log(f"ABORT: replica does not reproduce frozen predictions for {bad}")
        sys.exit(1)
    out["identity_gate_status"] = "PASS: replica reproduces frozen predictions bit-for-bit"
    log("identity gate PASS")

    for tgt in ("y30", "y35"):
        out["targets"][tgt] = run_target(g, m, tgt, log)
        json.dump(out, open(OUT / "r1_gnss_placebo_final.json", "w"), indent=2)

    out["runtime_s"] = round(time.time() - t_all, 1)
    json.dump(out, open(OUT / "r1_gnss_placebo_final.json", "w"), indent=2)
    (OUT / "r1_log.txt").write_text("\n".join(lines))
    log(f"\nwrote results/round4/r1_gnss_placebo_final.json ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
