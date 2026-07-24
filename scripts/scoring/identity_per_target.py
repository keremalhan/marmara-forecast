"""Locks two things:
(1) Per-target identity: h_m = sum(lambda_m)/N_pos, artifact_m = h_m - 1 - ln h_m. The pairwise
    count->occurrence IG shift is PREDICTED by artifact_A - artifact_B (cascade-vs-firstgen:
    predicted 0.386 vs measured 0.388). Extended to y30/y35/y45 (h->1 as threshold rises -> artifact
    vanishes). This is the h-1-ln(h) curve with each model placed on it, predicted vs measured.
(2) Three-scoring verdicts with CIs for the key pairs: count (raw lambda, Poisson-on-indicator),
    occurrence-Poisson (s*lambda, Poisson-on-indicator), occurrence-Bernoulli (p=1-e^{-s*lambda},
    Bernoulli log-score). Decision: does the Bernoulli hybrid-vs-cascade IG interval exclude zero?
Global-scalar occurrence (s = val N_pos / sum lambda_val, per model). MC-native occupancy handled
separately. Writes results/round3/t4_identity_scoring.json.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from marmara.paths import RESULTS
from marmara.metrics import p_to_lambda, lambda_to_p
from marmara.bootstrap import stationary_window_indices, MEAN_BLOCK, SEED

EPS = 1e-9
B = 2000
MODELS = ["hybrid", "cascade", "sv_etas", "firstgen_etas", "modern_etas", "poisson", "smoothed"]
KEY = [("hybrid", "cascade"), ("cascade", "modern_etas"), ("cascade", "firstgen_etas"),
       ("cascade", "sv_etas"), ("firstgen_etas", "modern_etas"), ("hybrid", "firstgen_etas")]


def lam(p):
    return np.clip(p_to_lambda(np.clip(np.asarray(p, float), 0.0, 1.0)), EPS, None)


def ig_ci_generic(win, y, contrib_a, contrib_b, pos):
    """contrib_* are per-window LL sums; returns point IG + CI (per positive)."""
    ig = (contrib_a.sum() - contrib_b.sum()) / max(pos.sum(), 1)
    rng = np.random.default_rng(SEED); seqs = stationary_window_indices(len(pos), B, MEAN_BLOCK, rng)
    bs = [(contrib_a[r].sum() - contrib_b[r].sum()) / max(pos[r].sum(), 1) for r in seqs]
    return round(float(ig), 4), [round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def per_window(win, arr):
    wins = np.sort(np.unique(win)); idx = {w: np.where(win == w)[0] for w in wins}
    return wins, np.array([arr[idx[w]].sum() for w in wins]), idx


def dpr_ci(win, y, pa, pb):
    wins = np.sort(np.unique(win)); idx = {w: np.where(win == w)[0] for w in wins}
    yb = [y[idx[w]] for w in wins]; ab = [pa[idx[w]] for w in wins]; bb = [pb[idx[w]] for w in wins]
    rng = np.random.default_rng(SEED); seqs = stationary_window_indices(len(wins), B, MEAN_BLOCK, rng)
    d = []
    for r in seqs:
        yy = np.concatenate([yb[i] for i in r])
        if 0 < yy.sum() < len(yy):
            d.append(average_precision_score(yy, np.concatenate([ab[i] for i in r]))
                     - average_precision_score(yy, np.concatenate([bb[i] for i in r])))
    return [round(float(np.percentile(d, 2.5)), 4), round(float(np.percentile(d, 97.5)), 4)]


def load(target):
    df = pd.read_parquet(RESULTS / f"predictions_{target}.parquet")
    return df[df.split == "val"], df[df.split == "test"]


def main():
    out = {"identity_per_target": {}, "y30_three_scoring": {}}

    # ---------- (1) per-target identity ----------
    for tgt in ("y30", "y35", "y45"):
        va, te = load(tgt)
        npos = float(te.y.sum())
        rows = {}
        for m in MODELS:
            h = float(lam(te[m].to_numpy()).sum() / npos)
            rows[m] = {"h": round(h, 3), "artifact_h_minus_1_minus_lnh": round(h - 1 - np.log(h), 4)}
        # pairwise predicted vs measured shift (Poisson, global-scalar occurrence)
        yv = va.y.to_numpy(); wt = te.window.to_numpy(); yt = te.y.to_numpy()
        s = {m: float(yv.sum() / lam(va[m].to_numpy()).sum()) for m in MODELS}
        Lt = {m: lam(te[m].to_numpy()) for m in MODELS}
        Locc = {m: s[m] * Lt[m] for m in MODELS}
        def ig_p(la, lb):
            return float((np.sum(yt * np.log(la) - la) - np.sum(yt * np.log(lb) - lb)) / max(npos, 1))
        pairs = {}
        for a, b in KEY:
            if a not in MODELS or b not in MODELS:
                continue
            meas = ig_p(Lt[a], Lt[b]) - ig_p(Locc[a], Locc[b])
            pred = rows[a]["artifact_h_minus_1_minus_lnh"] - rows[b]["artifact_h_minus_1_minus_lnh"]
            pairs[f"{a}_vs_{b}"] = {"measured_shift": round(meas, 4), "predicted_shift": round(pred, 4)}
        out["identity_per_target"][tgt] = {"n_pos": int(npos), "models": rows, "pairwise_shift": pairs}

    # ---------- (2) three-scoring verdicts (y30) ----------
    va, te = load("y30")
    yv = va.y.to_numpy(); yt = te.y.to_numpy(); wt = te.window.to_numpy()
    npos = float(yt.sum())
    s = {m: float(yv.sum() / lam(va[m].to_numpy()).sum()) for m in MODELS}
    Lt = {m: lam(te[m].to_numpy()) for m in MODELS}
    Locc = {m: s[m] * Lt[m] for m in MODELS}
    Pocc = {m: np.clip(lambda_to_p(Locc[m]), EPS, 1 - EPS) for m in MODELS}
    wins, pos, idx = per_window(wt, yt)

    def contrib_poisson(L):
        return np.array([np.sum(yt[idx[w]] * np.log(L[idx[w]]) - L[idx[w]]) for w in wins])
    def contrib_bern(p):
        cell = yt * np.log(p) + (1 - yt) * np.log(1 - p)
        return np.array([cell[idx[w]].sum() for w in wins])

    for a, b in KEY:
        rec = {}
        # count (raw lambda, Poisson)
        rec["count"] = dict(zip(("ig", "ig_ci"), ig_ci_generic(wt, yt, contrib_poisson(Lt[a]), contrib_poisson(Lt[b]), pos)))
        # occurrence-Poisson (s*lambda)
        rec["occ_poisson"] = dict(zip(("ig", "ig_ci"), ig_ci_generic(wt, yt, contrib_poisson(Locc[a]), contrib_poisson(Locc[b]), pos)))
        # occurrence-Bernoulli (p)
        rec["occ_bernoulli"] = dict(zip(("ig", "ig_ci"), ig_ci_generic(wt, yt, contrib_bern(Pocc[a]), contrib_bern(Pocc[b]), pos)))
        dpr = float(average_precision_score(yt, te[a].to_numpy()) - average_precision_score(yt, te[b].to_numpy()))
        rec["dpr"] = round(dpr, 4); rec["dpr_ci"] = dpr_ci(wt, yt, te[a].to_numpy(), te[b].to_numpy())
        for sc in ("count", "occ_poisson", "occ_bernoulli"):
            ic = rec[sc]["ig_ci"]; pc = rec["dpr_ci"]
            rec[sc]["verdict"] = "separable" if ((ic[0] > 0 or ic[1] < 0) and (pc[0] > 0 or pc[1] < 0)) else "inseparable"
            rec[sc]["ig_excludes_zero"] = bool(ic[0] > 0 or ic[1] < 0)
        out["y30_three_scoring"][f"{a}_vs_{b}"] = rec

    (RESULTS / "round3" / "t4_identity_scoring.json").write_text(json.dumps(out, indent=2))
    print("=== IDENTITY: artifact = h-1-ln(h); pairwise shift predicted vs measured ===")
    for tgt in ("y30", "y35", "y45"):
        t = out["identity_per_target"][tgt]
        print(f"\n{tgt} (n_pos {t['n_pos']}): h = " + ", ".join(f"{m[:5]} {t['models'][m]['h']}" for m in ("cascade", "firstgen_etas", "modern_etas", "hybrid")))
        for p, v in t["pairwise_shift"].items():
            print(f"    {p:26s} shift measured {v['measured_shift']:+.3f}  predicted {v['predicted_shift']:+.3f}")
    print("\n=== y30 THREE-SCORING (count / occ-Poisson / occ-Bernoulli) ===")
    for p, r in out["y30_three_scoring"].items():
        print(f"  {p:26s}: count {r['count']['ig']:+.3f}[{r['count']['verdict'][:5]}]  "
              f"occP {r['occ_poisson']['ig']:+.3f}{r['occ_poisson']['ig_ci']}[{r['occ_poisson']['verdict'][:5]}]  "
              f"occB {r['occ_bernoulli']['ig']:+.3f}{r['occ_bernoulli']['ig_ci']}[{r['occ_bernoulli']['verdict'][:5]}]")


if __name__ == "__main__":
    main()
