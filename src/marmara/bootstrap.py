"""Phase 0 — stationary block bootstrap over test windows + machine-derived claims.

WHY BLOCK, NOT IID.  Windows tile the timeline at 30-day steps (grid.STEP_D=30,
HORIZON_D=30): the *target* periods are non-overlapping, but consecutive 30-day
windows are NOT independent. They share trailing feature history (cnt365,
days-since, b-value, ETAS integral) and, more importantly, aftershock / triggered
sequences produce correlated positives across several consecutive windows. Naive
iid resampling of rows — or even of window-ids — understates sampling variability
and yields fake-tight CIs a reviewer would (correctly) destroy.

WHAT WE DO.  Two-level resampling:
  * the resampling UNIT is a whole window-id (all 1219 cells kept together) — this
    respects the strong within-window spatial correlation;
  * window-ids are drawn by the Politis-Romano STATIONARY bootstrap over the
    ordered window sequence (geometric block length, expected 3 consecutive window
    starts ~= 90 days ~ aftershock-cluster decorrelation) — this respects the
    temporal correlation between neighbouring windows.

For every model pair we bootstrap the PAIRED differences of PR-AUC, ROC-AUC,
IG (nats/event) and Brier (B=2000, seed=42; the SAME resample is scored for every
model, so differences are genuinely paired). We report the point estimate (observed
delta on the full split), the 95% percentile CI and the win-rate.

claims.json then applies the pre-registered claim rule (Phase 0.2):
  A "beats" B  iff  the 95% CI of the paired delta excludes 0 in A's favour for
  BOTH information gain AND PR-AUC.  Otherwise the pair is "inseparable".
All later reporting must quote claims.json, never a raw point difference.

Reads:  results/predictions_{target}.parquet   (written by marmara.train)
Writes: results/bootstrap_ci.{json,md}, results/claims.json
Run:    "<venv>/bin/python" -m marmara.bootstrap
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from marmara.metrics import information_gain, p_to_lambda

OUT = RESULTS
B_DEFAULT = 2000
SEED = 42
MEAN_BLOCK = 3.0          # expected block length in consecutive window starts
EPS = 1e-9

# Canonical model ordering for the comparison. fault_prox (a deliberately naive
# baseline, not a claim comparator) is omitted from the pair matrix per the
# blueprint. sv_etas / modern_etas are picked up automatically once Phase 1 adds
# their columns to predictions_*.parquet.
CANON = ["hybrid", "hybrid_gnss", "cascade", "sv_etas", "modern_etas", "firstgen_etas",
         "smoothed", "poisson"]
TARGETS = ["y30", "y35", "y45"]   # y30 = primary powered comparison (Phase 3)
SPLITS = ["test", "val"]      # test is primary; val reported for completeness


# --------------------------------------------------------------------------- #
# Stationary (Politis-Romano) bootstrap of the ordered window sequence.
# --------------------------------------------------------------------------- #
def stationary_window_indices(n_win: int, n_boot: int, mean_block: float,
                              rng: np.random.Generator) -> np.ndarray:
    """Return (n_boot, n_win) array of positions in [0, n_win): each row is one
    resampled sequence of window positions with geometric block length (mean
    `mean_block`), wrapping circularly. Length n_win reproduces the original
    number of window-blocks per resample."""
    p = 1.0 / mean_block
    out = np.empty((n_boot, n_win), dtype=np.intp)
    for b in range(n_boot):
        i = int(rng.integers(n_win))
        for k in range(n_win):
            out[b, k] = i
            if rng.random() < p:
                i = int(rng.integers(n_win))     # start a fresh block
            else:
                i = (i + 1) % n_win               # continue the current block
    return out


# --------------------------------------------------------------------------- #
# Per-window precomputation. IG and Brier are row-additive, so we accumulate
# per-window partial sums (fast + exact); PR/ROC are rank-based and need the
# pooled resample, so we keep the per-window y / p arrays for those two only.
# --------------------------------------------------------------------------- #
def precompute(df: pd.DataFrame, models: list[str]) -> dict:
    wins = np.sort(df["window"].unique())
    y_by, p_by = [], {m: [] for m in models}
    pos, sse, llc, n_rows = [], {m: [] for m in models}, {m: [] for m in models}, []
    for w in wins:
        sub = df[df["window"] == w]
        yv = sub["y"].to_numpy(float)
        y_by.append(yv)
        pos.append(float(yv.sum()))
        n_rows.append(len(yv))
        for m in models:
            pv = np.clip(sub[m].to_numpy(float), 0.0, 1.0)
            p_by[m].append(pv)
            sse[m].append(float(np.sum((pv - yv) ** 2)))          # Brier numerator
            lam = np.clip(p_to_lambda(pv), EPS, None)             # matches metrics.IG
            llc[m].append(float(np.sum(yv * np.log(lam) - lam)))  # Poisson LL contrib
    return {
        "wins": wins, "y_by": y_by, "p_by": p_by,
        "pos": np.array(pos), "n_rows": np.array(n_rows, float),
        "sse": {m: np.array(sse[m]) for m in models},
        "llc": {m: np.array(llc[m]) for m in models},
    }


def full_metrics(df: pd.DataFrame, models: list[str]) -> dict:
    """Observed (point-estimate) metrics on the full split."""
    y = df["y"].to_numpy(float)
    out = {}
    for m in models:
        p = np.clip(df[m].to_numpy(float), 0.0, 1.0)
        npos = float(y.sum())
        pr = float(average_precision_score(y, p)) if 0 < npos < len(y) else np.nan
        ro = float(roc_auc_score(y, p)) if 0 < npos < len(y) else np.nan
        out[m] = {"pr_auc": pr, "roc_auc": ro, "brier": float(brier_score_loss(y, p))}
    return out


def bootstrap_split(df: pd.DataFrame, models: list[str], n_boot: int,
                    rng: np.random.Generator) -> dict:
    pc = precompute(df, models)
    n_win = len(pc["wins"])
    seqs = stationary_window_indices(n_win, n_boot, MEAN_BLOCK, rng)

    M = len(models)
    PR = np.full((n_boot, M), np.nan)
    RO = np.full((n_boot, M), np.nan)
    BR = np.full((n_boot, M), np.nan)
    LL = np.zeros((n_boot, M))
    NEV = np.empty(n_boot)                     # events per resample (>=1, IG convention)

    for b in range(n_boot):
        seq = seqs[b]
        npos = pc["pos"][seq].sum()
        nrow = pc["n_rows"][seq].sum()
        NEV[b] = max(npos, 1.0)
        y_pool = np.concatenate([pc["y_by"][i] for i in seq])
        auc_ok = 0.0 < npos < nrow
        for j, m in enumerate(models):
            LL[b, j] = pc["llc"][m][seq].sum()
            BR[b, j] = pc["sse"][m][seq].sum() / nrow
            if auc_ok:
                p_pool = np.concatenate([pc["p_by"][m][i] for i in seq])
                PR[b, j] = average_precision_score(y_pool, p_pool)
                RO[b, j] = roc_auc_score(y_pool, p_pool)

    return {"models": models, "PR": PR, "RO": RO, "BR": BR, "LL": LL, "NEV": NEV,
            "n_win": n_win, "n_valid_auc": int(np.isfinite(PR[:, 0]).sum())}


# --------------------------------------------------------------------------- #
# Pairwise deltas + CIs + win-rates.
# --------------------------------------------------------------------------- #
def _ci(dist: np.ndarray) -> tuple[float, float, int]:
    v = dist[np.isfinite(dist)]
    if len(v) == 0:
        return float("nan"), float("nan"), 0
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)), len(v)


def _winrate(dist: np.ndarray, higher_better: bool) -> float:
    v = dist[np.isfinite(dist)]
    if len(v) == 0:
        return float("nan")
    return float(np.mean(v > 0) if higher_better else np.mean(v < 0))


def pair_stats(bs: dict, full: dict, a: str, b: str) -> dict:
    j = {m: k for k, m in enumerate(bs["models"])}
    ia, ib = j[a], j[b]
    d_pr = bs["PR"][:, ia] - bs["PR"][:, ib]
    d_ro = bs["RO"][:, ia] - bs["RO"][:, ib]
    d_br = bs["BR"][:, ia] - bs["BR"][:, ib]
    d_ig = (bs["LL"][:, ia] - bs["LL"][:, ib]) / bs["NEV"]

    def blk(dist, point, higher_better):
        lo, hi, nv = _ci(dist)
        pt = None if (point is None or not np.isfinite(point)) else round(float(point), 6)
        return {"point": pt,
                "ci95": [None if not np.isfinite(lo) else round(lo, 6),
                         None if not np.isfinite(hi) else round(hi, 6)],
                "win_rate": round(_winrate(dist, higher_better), 4),
                "excludes_0": bool(np.isfinite(lo) and np.isfinite(hi) and (lo > 0 or hi < 0)),
                "n_resamples": nv}

    pr_point = full[a]["pr_auc"] - full[b]["pr_auc"] if np.isfinite(full[a]["pr_auc"]) and np.isfinite(full[b]["pr_auc"]) else np.nan
    ro_point = full[a]["roc_auc"] - full[b]["roc_auc"] if np.isfinite(full[a]["roc_auc"]) and np.isfinite(full[b]["roc_auc"]) else np.nan
    br_point = full[a]["brier"] - full[b]["brier"]
    # IG point estimate straight from the shared metrics fn (exact match)
    return {
        "d_pr_auc": blk(d_pr, pr_point, higher_better=True),
        "d_roc_auc": blk(d_ro, ro_point, higher_better=True),
        "d_ig": blk(d_ig, None, higher_better=True),   # point filled by caller (needs y,p)
        "d_brier": blk(d_br, br_point, higher_better=False),
    }


def verdict_for(ig_ci: list, pr_ci: list) -> str:
    """Pre-registered rule: A beats B iff BOTH IG and PR-AUC 95% CIs exclude 0 in
    the same (A-favouring) direction; symmetric for B; else inseparable."""
    if None in ig_ci or None in pr_ci:
        return "inseparable"
    ig_lo, ig_hi = ig_ci
    pr_lo, pr_hi = pr_ci
    if ig_lo > 0 and pr_lo > 0:
        return "A_beats_B"
    if ig_hi < 0 and pr_hi < 0:
        return "B_beats_A"
    return "inseparable"


# --------------------------------------------------------------------------- #
def run(n_boot: int = B_DEFAULT) -> dict:
    report = {"meta": {"B": n_boot, "seed": SEED, "mean_block": MEAN_BLOCK,
                       "block": "Politis-Romano stationary bootstrap over ordered "
                       "window-ids (all 1219 cells per window kept together)",
                       "note": "windows tile at 30-day steps; targets non-overlapping "
                       "but temporally dependent via shared feature history + aftershock "
                       "clustering -> block (not iid) resampling required."},
              "results": {}}
    claims = []

    for target in TARGETS:
        path = OUT / f"predictions_{target}.parquet"
        if not path.exists():
            print(f"SKIP {target}: {path.name} missing (run marmara.train first)")
            continue
        df_all = pd.read_parquet(path)
        models = [m for m in CANON if m in df_all.columns]
        report["results"][target] = {}
        for split in SPLITS:
            df = df_all[df_all["split"] == split].reset_index(drop=True)
            if len(df) == 0:
                continue
            rng = np.random.default_rng(SEED)     # same seed per split -> reproducible
            full = full_metrics(df, models)
            bs = bootstrap_split(df, models, n_boot, rng)
            # exact IG point estimates from the shared metrics fn
            y = df["y"].to_numpy(float)
            ig_point = {}
            for m in models:
                ig_point[m] = {mm: information_gain(df[m].to_numpy(float),
                                                    df[mm].to_numpy(float), y)
                               for mm in models}
            pairs = {}
            for i in range(len(models)):
                for k in range(i + 1, len(models)):
                    a, bb = models[i], models[k]
                    st = pair_stats(bs, full, a, bb)
                    st["d_ig"]["point"] = round(float(ig_point[a][bb]), 6)
                    v = verdict_for(st["d_ig"]["ci95"], st["d_pr_auc"]["ci95"])
                    st["verdict"] = v
                    key = f"{a}_vs_{bb}"
                    pairs[key] = st
                    claims.append({
                        "pair": key, "A": a, "B": bb, "target": target, "split": split,
                        "verdict": v,
                        "ig": {"point": st["d_ig"]["point"], "ci95": st["d_ig"]["ci95"]},
                        "pr_auc": {"point": st["d_pr_auc"]["point"], "ci95": st["d_pr_auc"]["ci95"]},
                        "primary": bool(target == "y30" and split == "test"),
                    })
            report["results"][target][split] = {
                "n": int(len(df)), "n_pos": int(y.sum()), "n_windows": bs["n_win"],
                "n_valid_auc_resamples": bs["n_valid_auc"],
                "models": models,
                "full_metrics": {m: {k: (None if not np.isfinite(v) else round(v, 6))
                                     for k, v in full[m].items()} for m in models},
                "pairs": pairs,
            }
            print(f"[{target}/{split}] n={len(df)} pos={int(y.sum())} "
                  f"windows={bs['n_win']} models={models}")

    report["claims"] = claims
    return report


# --------------------------------------------------------------------------- #
def write_markdown(report: dict, path):
    L = ["# Phase 0 — block-bootstrap CIs (paired model differences)", ""]
    m = report["meta"]
    L.append(f"B={m['B']}, seed={m['seed']}, stationary block bootstrap over "
             f"window-ids (mean block {m['mean_block']} windows). {m['note']}")
    L.append("")
    L.append("Delta is defined as **(A − B)** for the listed pair; the reverse pair "
             "follows by symmetry (sign flip, win-rate → 1−w). Higher is better for "
             "PR-AUC / ROC-AUC / IG; lower is better for Brier. "
             "**Verdict** applies the pre-registered rule: A beats B iff the 95% CI "
             "excludes 0 (A-favouring) for BOTH IG and PR-AUC.")
    for target, splits in report["results"].items():
        for split, r in splits.items():
            tag = " *(PRIMARY — acceptance target)*" if (target == "y35" and split == "test") else ""
            L.append(f"\n## {target} / {split}{tag}  "
                     f"(n={r['n']}, positives={r['n_pos']}, windows={r['n_windows']})")
            L.append("\nFull-split metrics:")
            L.append("| model | PR-AUC | ROC-AUC | Brier |")
            L.append("|---|---|---|---|")
            for mm in r["models"]:
                fm = r["full_metrics"][mm]
                f = lambda v, d=4: "n/a" if v is None else f"{v:.{d}f}"
                L.append(f"| {mm} | {f(fm['pr_auc'])} | {f(fm['roc_auc'])} | {f(fm['brier'],5)} |")
            L.append("\nPaired differences (point [95% CI], win-rate):")
            L.append("| pair (A vs B) | Δ IG | Δ PR-AUC | Δ ROC-AUC | Δ Brier | verdict |")
            L.append("|---|---|---|---|---|---|")

            def cell(blk, d=4):
                pt = "n/a" if blk["point"] is None else f"{blk['point']:+.{d}f}"
                lo, hi = blk["ci95"]
                ci = "[n/a]" if lo is None else f"[{lo:+.{d}f}, {hi:+.{d}f}]"
                star = "*" if blk["excludes_0"] else ""
                return f"{pt} {ci}{star} w={blk['win_rate']}"

            for key, st in r["pairs"].items():
                a, bb = key.split("_vs_")
                vshort = {"A_beats_B": f"**{a} beats {bb}**", "B_beats_A": f"**{bb} beats {a}**",
                          "inseparable": "inseparable"}[st["verdict"]]
                L.append(f"| {a} vs {bb} | {cell(st['d_ig'])} | {cell(st['d_pr_auc'])} | "
                         f"{cell(st['d_roc_auc'])} | {cell(st['d_brier'],5)} | {vshort} |")
            L.append("\n(* = 95% CI excludes 0.)")
    path.write_text("\n".join(L))


def write_claims(report: dict, path):
    out = {
        "rule": "A beats B iff the 95% percentile bootstrap CI of the paired "
                "difference excludes 0 in A's favour for BOTH information gain "
                "(nats/event) AND PR-AUC; otherwise the pair is inseparable.",
        "bootstrap": report["meta"],
        "claims": report["claims"],
    }
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


def main():
    n_boot = int(sys.argv[1]) if len(sys.argv) > 1 else B_DEFAULT
    report = run(n_boot)
    with open(OUT / "bootstrap_ci.json", "w") as f:
        json.dump(report, f, indent=2)
    write_markdown(report, OUT / "bootstrap_ci.md")
    write_claims(report, OUT / "claims.json")

    # headline: y30 (primary powered comparison, Phase 3) verdicts.
    prim = [c for c in report["claims"] if c["primary"]]
    print("\n=== y30 / test verdicts (PRIMARY powered comparison) ===")
    for c in prim:
        print(f"  {c['pair']:34s} {c['verdict']:12s} "
              f"IG {c['ig']['ci95']}  PR {c['pr_auc']['ci95']}")
    # Phase-0 pre-registered check still reported (cascade vs firstgen, y35 test)
    key = next((c for c in report["claims"]
                if c["pair"] == "cascade_vs_firstgen_etas" and c["target"] == "y35"
                and c["split"] == "test"), None)
    if key:
        print(f"\nPRE-REGISTERED CHECK — cascade vs first-gen ETAS (y35 test): "
              f"{key['verdict']} (expected: inseparable)")
    print("\nwrote bootstrap_ci.{json,md}, claims.json")


if __name__ == "__main__":
    main()
