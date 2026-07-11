"""does the OPERATIONAL y30 GNSS benefit (hybrid_gnss beats hybrid)
survive a time-shuffle placebo? Rebuild hybrid_gnss with the GNSS features time-shuffled
per cell (destroying time-alignment) and recompute hybrid_gnss-vs-hybrid ΔPR-AUC / ΔIG
on the y30 test split. If the REAL benefit lies inside the placebo null, the y30
operational benefit is spurious too -> C3 VOID. Writes results/verify/gnss_y30_placebo.json.
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.ensemble import HistGradientBoostingRegressor
from marmara.paths import RESULTS
from marmara.grid import FEATURES
from marmara.metrics import information_gain, lambda_to_p
from marmara.train import split_masks, GNSS_PHYS, _monotonic, WEIGHTS, poisson_ll

EPS = 1e-9; OUT = RESULTS; VER = OUT / "verify"; N_PERM = 25; SEED = 42


def load():
    g = pd.read_parquet(OUT / "grid_hybrid.parquet")
    gc = pd.read_parquet(OUT / "gnss_traj_columns.parquet").set_index(["window", "ir", "ic"])
    idx = pd.MultiIndex.from_frame(g[["window", "ir", "ic"]])
    for c in GNSS_PHYS:
        g[c] = gc[c].reindex(idx).to_numpy()
    return g, split_masks(g)


def hybrid_P(g, m, extra):
    """Replicate train.fit_hybrid for y30 (count30/lam30_sim) with optional extra feats."""
    tr, va = m["train"], m["val"]
    feats = FEATURES + list(extra)
    X = g[feats].copy(); X["ln_lam_sim"] = np.log(g["lam30_sim"].to_numpy() + EPS)
    n = g["count30"].to_numpy().astype(float); y = g["y30"].to_numpy().astype(float)
    lam_sim = np.clip(g["lam30_sim"].to_numpy(), EPS, None)
    reg = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=400,
                                        max_depth=6, monotonic_cst=_monotonic(feats + ["ln_lam_sim"]),
                                        random_state=42)
    reg.fit(X[tr], n[tr]); lam_ml = np.clip(reg.predict(X), EPS, None)
    best = None
    for w in WEIGHTS:
        lam = lam_sim ** (1 - w) * lam_ml ** w
        ll = poisson_ll(y[va], lam[va])
        if best is None or ll > best[0]:
            best = (ll, w)
    w = best[1]
    return lambda_to_p(lam_sim ** (1 - w) * lam_ml ** w)


def main():
    g, m = load()
    te = m["test"]; y = g["y30"].to_numpy().astype(float); yte = y[te]
    P_h = hybrid_P(g, m, [])                       # hybrid (no GNSS), fixed
    P_hg = hybrid_P(g, m, GNSS_PHYS)               # hybrid_gnss (real)
    def dpr(Pa): return average_precision_score(yte, Pa[te]) - average_precision_score(yte, P_h[te])
    def dig(Pa): return information_gain(Pa[te], P_h[te], yte)
    real_dpr = float(dpr(P_hg)); real_dig = float(dig(P_hg))
    print(f"real y30 hybrid_gnss vs hybrid: dPR-AUC {real_dpr:+.4f}  dIG {real_dig:+.4f}")

    ck = g["ir"].to_numpy() * 1000 + g["ic"].to_numpy(); wins = g["window"].to_numpy()
    order = {k: np.where(ck == k)[0][np.argsort(wins[ck == k])] for k in np.unique(ck)}
    prng = np.random.default_rng(SEED)
    null_dpr, null_dig = [], []
    for _ in range(N_PERM):
        gp = g.copy(); arr = {c: g[c].to_numpy().copy() for c in GNSS_PHYS}
        for k, idx in order.items():
            perm = prng.permutation(len(idx))
            for c in GNSS_PHYS:
                arr[c][idx] = arr[c][idx][perm]
        for c in GNSS_PHYS:
            gp[c] = arr[c]
        Pp = hybrid_P(gp, m, GNSS_PHYS)
        null_dpr.append(float(dpr(Pp))); null_dig.append(float(dig(Pp)))
    null_dpr = np.array(null_dpr); null_dig = np.array(null_dig)
    out = {
        "real_dPR_auc": round(real_dpr, 5), "real_dIG": round(real_dig, 5), "n_perm": N_PERM,
        "placebo_dPR_mean": round(float(null_dpr.mean()), 5),
        "placebo_dPR_p95": [round(float(np.percentile(null_dpr, 2.5)), 5), round(float(np.percentile(null_dpr, 97.5)), 5)],
        "placebo_dIG_mean": round(float(null_dig.mean()), 5),
        "placebo_dIG_p95": [round(float(np.percentile(null_dig, 2.5)), 5), round(float(np.percentile(null_dig, 97.5)), 5)],
        "real_dPR_exceeds_placebo_p95": bool(real_dpr > np.percentile(null_dpr, 97.5)),
        "real_dIG_exceeds_placebo_p95": bool(real_dig > np.percentile(null_dig, 97.5)),
    }
    out["verdict"] = ("GENUINE at y30: the real benefit exceeds the time-shuffle placebo null"
                      if (out["real_dPR_exceeds_placebo_p95"] and out["real_dIG_exceeds_placebo_p95"])
                      else "SPURIOUS: the y30 benefit lies inside the placebo null; GNSS gain is not "
                      "time-aligned deformation -> C3 VOID")
    json.dump(out, open(VER / "gnss_y30_placebo.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
