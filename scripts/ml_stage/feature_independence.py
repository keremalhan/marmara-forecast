"""Round 5 — reviewer-requested ablation + grouped-PCA effective-dimensionality study.

QUESTION (single, fixed): how many genuinely independent information axes do the
~20 model inputs contain, and does ANY axis carry out-of-sample predictive
information beyond the ETAS forecasts?  This directly tests the manuscript thesis
(ML = ETAS recalibration, no new structure) with causal feature ablations instead
of post-hoc interpretation.

Design (y30 / M>=3.0, the primary powered target; test = the same 26 windows):
  A. ABLATION LADDER — pure Poisson-GBR rate models (no geometric blend, so each
     model's score measures its feature set's information content directly) on
     nested feature sets: ETAS-only (first-gen / cascade / both), counts-only,
     catalogue-without-ETAS, catalogue+geophysics-without-ETAS, ETAS+recent,
     ETAS+geophysics, full.  Identical config to the canonical hybrid
     (lr 0.05, depth 6, Poisson loss, monotonic constraints, seed 42) EXCEPT
     early stopping: the tree count is selected by val-period (2022-2024) Poisson
     LL (grouped temporal selection, per the review), not sklearn's random 10%.
  B. SCORING — count Poisson IG (occupancy convention, matches metrics.py),
     Bernoulli proper binary IG, PR-AUC/ROC/Brier/reliability via the shared
     score_predictor; paired stationary-block-bootstrap CIs (B=2000, seed 42,
     mean block 3 windows — identical to bootstrap.py) for the key pairs; the
     pre-registered verdict rule (both IG and PR-AUC CIs exclude 0).
  C. STABILITY — per-window signs, leave-one-window-out jackknife, and the
     Amendment-8 calendar rule leave-Kumburgaz-out split.
  D. CONDITIONAL (grouped) PERMUTATION IMPORTANCE — permute each feature block
     jointly on test under the FULL model (ETAS intact => information conditional
     on ETAS); global and within-window variants, R=20.
  E. GROUPED PCA — train-only standardization (log1p for heavy-tailed inputs),
     PCA within physically coherent blocks, retain >=90% block variance;
     booster retrained on top-k components, k = 1..K, to locate where predictive
     performance saturates (the predictive effective dimensionality).

Writes: results/round5/independence.json  (canonical artifacts untouched).
Run:    PYTHONPATH=src MARMARA_ROOT=. .venv/bin/python scripts/ml_stage/feature_independence.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import average_precision_score

from marmara import baselines as B
from marmara.bootstrap import (MEAN_BLOCK, SEED, bootstrap_split, full_metrics,
                               pair_stats, stationary_window_indices, verdict_for)
from marmara.evaluate import train_b_value
from marmara.grid import FEATURES
from marmara.metrics import information_gain, lambda_to_p, score_predictor
from marmara.paths import RESULTS
from marmara.train import split_masks

EPS = 1e-9
R5 = RESULTS / "round5"
R5.mkdir(exist_ok=True)
T0 = time.time()
K_LO = pd.Timestamp("2025-04-23"); K_HI = pd.Timestamp("2025-07-23")  # Amendment-8 rule

BLOCKS = {
    "recent_counts": ["cnt30", "cnt90", "cnt365", "nbr3_cnt30", "nbr3_cnt365",
                      "nbr5_cnt30", "nbr5_cnt365", "rate_ratio"],
    "recurrence": ["days_since_m35_25km", "days_since_m45_25km"],
    "catalogue_stats": ["maxmag365_25km", "mean_depth90", "b_pos_50km_730d",
                        "b_pos_is_fallback"],
    "etas": ["etas_rate", "ln_lam_sim"],
    "geophysics": ["dist_fault_km", "dcfs_perm", "dcfs_decay25", "strain_inv"],
}
CATALOGUE = BLOCKS["recent_counts"] + BLOCKS["recurrence"] + BLOCKS["catalogue_stats"]
ALL20 = FEATURES + ["ln_lam_sim"]
MODELS = {
    "full": ALL20,
    "etas_only": BLOCKS["etas"],
    "etas_firstgen_only": ["etas_rate"],
    "etas_cascade_only": ["ln_lam_sim"],
    "counts_only": BLOCKS["recent_counts"],
    "catalogue_no_etas": CATALOGUE,
    "no_etas": CATALOGUE + BLOCKS["geophysics"],
    "etas_plus_recent": BLOCKS["etas"] + BLOCKS["recent_counts"],
    "etas_plus_phys": BLOCKS["etas"] + BLOCKS["geophysics"],
}
KEY_PAIRS = [
    ("full", "etas_only"), ("full", "cascade"), ("etas_only", "cascade"),
    ("etas_only", "etas_cascade_only"), ("etas_cascade_only", "etas_firstgen_only"),
    ("counts_only", "etas_only"), ("catalogue_no_etas", "etas_only"),
    ("no_etas", "etas_only"), ("etas_plus_recent", "etas_only"),
    ("etas_plus_phys", "etas_only"), ("full", "etas_plus_recent"),
    ("catalogue_no_etas", "cascade"),
]
LOG1P = ["cnt30", "cnt90", "cnt365", "nbr3_cnt30", "nbr3_cnt365", "nbr5_cnt30",
         "nbr5_cnt365", "rate_ratio", "etas_rate", "strain_inv"]


def say(msg):
    print(f"[{time.time()-T0:6.0f}s] {msg}", flush=True)


def mono_for(feats):
    return [(-1 if f == "dist_fault_km" else 1 if f in ("etas_rate", "ln_lam_sim") else 0)
            for f in feats]


def fit_rate_model(X, n, tr, va, mono=None, tag=""):
    """Poisson GBR with grouped temporal iteration selection: fit 400 iters on
    train, pick argmax val-period Poisson LL, refit at that count (deterministic:
    identical leading trees). Returns (model, lam_all, diag)."""
    kw = dict(loss="poisson", learning_rate=0.05, max_depth=6, random_state=42,
              early_stopping=False)
    if mono is not None:
        kw["monotonic_cst"] = mono
    reg = HistGradientBoostingRegressor(max_iter=400, **kw)
    reg.fit(X[tr], n[tr])
    nv, lls = n[va], []
    for pred in reg.staged_predict(X[va]):
        lam = np.clip(pred, EPS, None)
        lls.append(float(np.sum(nv * np.log(lam) - lam)))
    it = int(np.argmax(lls)) + 1
    reg2 = HistGradientBoostingRegressor(max_iter=it, **kw)
    reg2.fit(X[tr], n[tr])
    lam_all = np.clip(reg2.predict(X), EPS, None)
    diag = {"n_iter": it, "val_ll_at_sel": round(lls[it - 1], 2),
            "val_ll_at_400": round(lls[-1], 2)}
    say(f"  {tag}: n_iter={it} valLL {lls[it-1]:.1f} (400-iter {lls[-1]:.1f})")
    return reg2, lam_all, diag


def bern_contrib(lam, y):
    p = np.clip(lambda_to_p(lam), EPS, 1 - EPS)
    return y * np.log(p) + (1 - y) * np.log(1 - p)


def occ_contrib(lam, y):
    lam = np.clip(lam, EPS, None)
    return y * np.log(lam) - lam


def per_window(c, win, wins):
    return np.array([c[win == w].sum() for w in wins])


def boot_delta(cw_a, cw_b, pos_w, rng_seed=SEED, B_=2000):
    """Paired stationary-block-bootstrap CI of a per-positive row-additive delta."""
    d = cw_a - cw_b
    npos = max(pos_w.sum(), 1.0)
    point = d.sum() / npos
    rng = np.random.default_rng(rng_seed)
    seqs = stationary_window_indices(len(d), B_, MEAN_BLOCK, rng)
    bs = d[seqs].sum(axis=1) / np.maximum(pos_w[seqs].sum(axis=1), 1.0)
    return (round(float(point), 4),
            [round(float(np.percentile(bs, 2.5)), 4),
             round(float(np.percentile(bs, 97.5)), 4)])


def main():
    say("loading grid + baselines")
    g = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    masks = split_masks(g)
    tr, va, te = masks["train"], masks["val"], masks["test"]
    n = g["count30"].to_numpy(float)
    y = g["y30"].to_numpy(float)
    win = g["window"].to_numpy()
    lam_casc = np.clip(g["lam30_sim"].to_numpy(float), EPS, None)

    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    b_train = train_b_value(cat, 3.0)
    mc_etas = json.load(open(RESULTS / "etas" / "etas_fit_report.json"))["base_mc"]
    lam_fg = np.clip(B.etas_baseline(g, 3.0, b_train, mc_etas), EPS, None)

    Xall = g[FEATURES].copy()
    Xall["ln_lam_sim"] = np.log(g["lam30_sim"].to_numpy() + EPS)

    out = {"design": {"target": "y30 (primary powered)", "note": __doc__.split("Design")[0].strip(),
                      "blocks": BLOCKS, "models": {k: v for k, v in MODELS.items()},
                      "early_stopping": "argmax val-period (2022-2024) Poisson LL over "
                                        "staged 400-iter fit; refit at selected count",
                      "bootstrap": {"B": 2000, "seed": SEED, "mean_block": MEAN_BLOCK}}}

    # ---------------- A. ablation ladder ----------------
    say("A. fitting ablation ladder")
    lam = {"cascade": lam_casc, "firstgen_etas": lam_fg}
    fitted = {}
    diags = {}
    for name, feats in MODELS.items():
        reg, lam_m, d = fit_rate_model(Xall[feats], n, tr, va, mono_for(feats), name)
        lam[name] = lam_m
        fitted[name] = (reg, feats)
        diags[name] = d
    out["fit_diagnostics"] = diags

    # ---------------- B. scoring + bootstrap ----------------
    say("B. scoring")
    preds = {m: lambda_to_p(v) for m, v in lam.items()}
    scores = {}
    for split, mask in (("val", va), ("test", te)):
        idx = np.where(mask)[0]
        ys, ws = y[idx], win[idx]
        sc = {}
        for m, P in preds.items():
            s = score_predictor(P[idx], ys, ws)
            s.pop("alert_budget", None)
            cw = occ_contrib(lam[m][idx], ys)
            bw = bern_contrib(lam[m][idx], ys)
            s["occ_poisson_ll_per_pos"] = round(float(cw.sum() / max(ys.sum(), 1)), 4)
            s["bernoulli_ll_per_pos"] = round(float(bw.sum() / max(ys.sum(), 1)), 4)
            cnt = occ_contrib(lam[m][idx], n[idx])  # true-count Poisson LL
            s["count_ll_per_event"] = round(float(cnt.sum() / max(n[idx].sum(), 1)), 4)
            sc[m] = s
        scores[split] = {"n": len(idx), "n_pos": int(ys.sum()), "models": sc}
    out["scores"] = scores

    say("B. block bootstrap (PR/ROC pooled, B=2000)")
    ti = np.where(te)[0]
    df = pd.DataFrame({"window": win[ti], "y": y[ti]})
    mnames = list(preds)
    for m in mnames:
        df[m] = preds[m][ti]
    rng = np.random.default_rng(SEED)
    full = full_metrics(df, mnames)
    bs = bootstrap_split(df, mnames, 2000, rng)

    wins = np.sort(np.unique(win[ti]))
    wt = win[ti]; yt = y[ti]
    pos_w = per_window(yt, wt, wins)
    occ_w = {m: per_window(occ_contrib(lam[m][ti], yt), wt, wins) for m in mnames}
    ber_w = {m: per_window(bern_contrib(lam[m][ti], yt), wt, wins) for m in mnames}

    pairs = {}
    for a, b in KEY_PAIRS:
        st = pair_stats(bs, full, a, b)
        st["d_ig"]["point"] = round(float(information_gain(df[a], df[b], yt)), 6)
        ig_pt, ig_ci = boot_delta(occ_w[a], occ_w[b], pos_w)
        bn_pt, bn_ci = boot_delta(ber_w[a], ber_w[b], pos_w)
        st["d_ig"]["ci95"] = ig_ci
        st["d_bernoulli_ig"] = {"point": bn_pt, "ci95": bn_ci,
                                "excludes_0": bool(bn_ci[0] > 0 or bn_ci[1] < 0)}
        st["verdict"] = verdict_for(ig_ci, st["d_pr_auc"]["ci95"])
        pairs[f"{a}_vs_{b}"] = st
    out["test_pairs"] = pairs

    # ---------------- C. stability ----------------
    say("C. stability")
    win_t0 = g.groupby("window")["t0"].first()
    excl = [int(w) for w in wins if (pd.Timestamp(win_t0.loc[w]) + pd.Timedelta(days=30) >= K_LO)
            and (pd.Timestamp(win_t0.loc[w]) <= K_HI)]
    keep = [int(w) for w in wins if int(w) not in excl]
    stab = {"kumburgaz_excluded_t0": [str(pd.Timestamp(win_t0.loc[w]).date()) for w in excl]}
    for a, b in [("full", "etas_only"), ("full", "cascade"), ("etas_only", "cascade"),
                 ("etas_plus_phys", "etas_only"), ("etas_plus_recent", "etas_only")]:
        d_ig = occ_w[a] - occ_w[b]
        d_bn = ber_w[a] - ber_w[b]
        npos = max(pos_w.sum(), 1.0)
        # leave-one-window-out jackknife of the per-positive delta
        tot, ptot = d_ig.sum(), pos_w.sum()
        lowo = [(tot - d_ig[i]) / max(ptot - pos_w[i], 1.0) for i in range(len(wins))]
        km = np.isin(wins, excl)
        stab[f"{a}_vs_{b}"] = {
            "per_window_sign_ig": f"+{int((d_ig>0).sum())}/-{int((d_ig<0).sum())}/0:{int((d_ig==0).sum())}",
            "per_window_sign_bern": f"+{int((d_bn>0).sum())}/-{int((d_bn<0).sum())}/0:{int((d_bn==0).sum())}",
            "ig_full": round(float(tot / npos), 4),
            "ig_lowo_range": [round(float(min(lowo)), 4), round(float(max(lowo)), 4)],
            "ig_leave_kumburgaz_out": round(float(d_ig[~km].sum() / max(pos_w[~km].sum(), 1.0)), 4),
            "ig_kumburgaz_only": round(float(d_ig[km].sum() / max(pos_w[km].sum(), 1.0)), 4),
            "bern_leave_kumburgaz_out": round(float(d_bn[~km].sum() / max(pos_w[~km].sum(), 1.0)), 4),
        }
    out["stability"] = stab

    # ---------------- D. conditional grouped permutation importance ----------------
    say("D. grouped permutation importance (full model, test, R=20)")
    reg_f, feats_f = fitted["full"]
    Xte = Xall[feats_f].iloc[ti].reset_index(drop=True)
    base_lam = np.clip(reg_f.predict(Xte), EPS, None)
    base_ig = occ_contrib(base_lam, yt).sum() / max(yt.sum(), 1)
    base_bn = bern_contrib(base_lam, yt).sum() / max(yt.sum(), 1)
    base_pr = average_precision_score(yt, lambda_to_p(base_lam))
    perm = {"baseline": {"occ_ll_per_pos": round(float(base_ig), 4),
                         "bern_ll_per_pos": round(float(base_bn), 4),
                         "pr_auc": round(float(base_pr), 4)}}
    rngp = np.random.default_rng(SEED)
    win_idx = {w: np.where(wt == w)[0] for w in wins}
    for bname, cols in BLOCKS.items():
        drops = {"global": {"ig": [], "bern": [], "pr": []},
                 "within_window": {"ig": [], "bern": [], "pr": []}}
        for _ in range(20):
            for mode in ("global", "within_window"):
                Xp = Xte.copy()
                if mode == "global":
                    p_ = rngp.permutation(len(Xp))
                    Xp[cols] = Xte[cols].to_numpy()[p_]
                else:
                    arr = Xte[cols].to_numpy().copy()
                    for w in wins:
                        ii = win_idx[w]
                        arr[ii] = arr[ii[rngp.permutation(len(ii))]]
                    Xp[cols] = arr
                lp = np.clip(reg_f.predict(Xp), EPS, None)
                drops[mode]["ig"].append(base_ig - occ_contrib(lp, yt).sum() / max(yt.sum(), 1))
                drops[mode]["bern"].append(base_bn - bern_contrib(lp, yt).sum() / max(yt.sum(), 1))
                drops[mode]["pr"].append(base_pr - average_precision_score(yt, lambda_to_p(lp)))
        perm[bname] = {mode: {k: [round(float(np.mean(v)), 4), round(float(np.std(v)), 4)]
                              for k, v in dd.items()} for mode, dd in drops.items()}
        say(f"  {bname}: global ΔLL/pos {perm[bname]['global']['ig'][0]:+.4f}")
    out["conditional_permutation_importance"] = {
        "note": "score DROP when the block is permuted on test under the FULL model "
                "(ETAS features intact for non-etas blocks => importance conditional "
                "on ETAS); mean+-sd over 20 repeats; within_window permutes only "
                "across cells inside each window (spatial information only).",
        "blocks": perm}

    # ---------------- E. grouped PCA ----------------
    say("E. grouped PCA")
    Xp = Xall.copy()
    for c in LOG1P:
        Xp[c] = np.log1p(np.clip(Xp[c], 0, None))
    med = Xp[tr].median()
    Xp = Xp.fillna(med)
    mu, sd = Xp[tr].mean(), Xp[tr].std().replace(0, 1.0)
    Xs = (Xp - mu) / sd

    comps, load, blk_info = [], {}, {}
    for bname, cols in BLOCKS.items():
        pca = PCA(random_state=42).fit(Xs.loc[tr, cols])
        evr = pca.explained_variance_ratio_
        k90 = int(np.searchsorted(np.cumsum(evr), 0.90) + 1)
        k95 = int(np.searchsorted(np.cumsum(evr), 0.95) + 1)
        Z = pca.transform(Xs[cols])
        for j in range(k90):
            cname = f"{bname}_pc{j+1}"
            comps.append({"name": cname, "block": bname,
                          "evr": float(evr[j]), "abs_var": float(evr[j] * len(cols)),
                          "z": Z[:, j]})
            load[cname] = {c: round(float(v), 3) for c, v in zip(cols, pca.components_[j])
                           if abs(v) >= 0.25}
        blk_info[bname] = {"n_features": len(cols), "k_at_90pct": k90, "k_at_95pct": k95,
                           "evr": [round(float(v), 3) for v in evr]}
    comps.sort(key=lambda c: -c["abs_var"])
    order = [c["name"] for c in comps]
    Zall = np.column_stack([c["z"] for c in comps])
    say(f"  retained {len(comps)} components at 90% block variance "
        f"(blocks: { {b: i['k_at_90pct'] for b, i in blk_info.items()} })")

    sweep = []
    for k in range(1, len(comps) + 1):
        _, lam_k, d = fit_rate_model(pd.DataFrame(Zall[:, :k], columns=order[:k]),
                                     n, tr, va, None, f"pca_k{k}")
        idx = ti
        sweep.append({
            "k": k, "added": order[k - 1], "n_iter": d["n_iter"],
            "test_occ_ig_vs_cascade": round(float(
                (occ_contrib(lam_k[idx], yt) - occ_contrib(lam_casc[idx], yt)).sum()
                / max(yt.sum(), 1)), 4),
            "test_bern_ig_vs_cascade": round(float(
                (bern_contrib(lam_k[idx], yt) - bern_contrib(lam_casc[idx], yt)).sum()
                / max(yt.sum(), 1)), 4),
            "test_pr_auc": round(float(average_precision_score(yt, lambda_to_p(lam_k[idx]))), 4),
            "val_pr_auc": round(float(average_precision_score(
                y[va], lambda_to_p(lam_k[np.where(va)[0]]))), 4),
        })
    out["grouped_pca"] = {
        "transform": "log1p on heavy-tailed cols, train-only standardization, "
                     "train-only PCA per block, >=90% block variance retained",
        "log1p_cols": LOG1P, "block_info": blk_info,
        "component_order_by_abs_variance": order,
        "loadings_abs_ge_0.25": load, "dimensionality_sweep": sweep}

    out["runtime_s"] = round(time.time() - T0, 1)
    json.dump(out, open(R5 / "independence.json", "w"), indent=1, default=str)
    say("wrote results/round5/independence.json")

    print("\n=== KEY VERDICTS (y30 test, pre-registered rule) ===")
    for k, st in pairs.items():
        print(f"  {k:40s} {st['verdict']:12s} IG {st['d_ig']['point']:+.4f} {st['d_ig']['ci95']}"
              f"  Bern {st['d_bernoulli_ig']['point']:+.4f} {st['d_bernoulli_ig']['ci95']}"
              f"  ΔPR {st['d_pr_auc']['point']:+.4f}")


if __name__ == "__main__":
    main()
