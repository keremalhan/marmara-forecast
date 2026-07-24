"""GNSS forensic audit: bootstrap CI on the aug-vs-base IG, and
permutation importance of the gnss_traj columns.

(c) Block-bootstrap (stationary, over test window-ids, B=2000) the count-model
    IG(base+gnss vs base) on the y35 test split. If the 95% CI includes 0 -> C3
    downgrades to "suggestive".
(d) Permutation importance of each gnss_traj column within the trained aug model: if
    the gain loads on gnss_strain_fallback (availability proxy) rather than the
    residual-rate / strain features -> artifact -> VOID.

Writes results/verify/gnss_ig.json.
"""
import json
import numpy as np
import pandas as pd
from marmara.paths import RESULTS
from marmara.grid import FEATURES
from marmara.metrics import information_gain, lambda_to_p, p_to_lambda
from marmara.train import split_masks
from marmara.bootstrap import stationary_window_indices, MEAN_BLOCK
from sklearn.ensemble import HistGradientBoostingRegressor

EPS = 1e-9
OUT = RESULTS
VER = OUT / "verify"; VER.mkdir(exist_ok=True)
GNSS5 = ["gnss_resid_rate_90", "gnss_resid_rate_365", "gnss_resid_max_365",
         "gnss_strain_rate_365", "gnss_strain_fallback"]
GNSS4 = GNSS5[:4]
BASE = FEATURES + ["ln_lam_sim"]
B = 2000; SEED = 42


def load():
    g = pd.read_parquet(OUT / "grid" / "grid_hybrid.parquet")
    gc = pd.read_parquet(OUT / "channels" / "gnss_traj_columns.parquet").set_index(["window", "ir", "ic"])
    idx = pd.MultiIndex.from_frame(g[["window", "ir", "ic"]])
    for c in GNSS5:
        g[c] = gc[c].reindex(idx).to_numpy()
    g["ln_lam_sim"] = np.log(g["lam35_sim"].to_numpy() + EPS)
    return g, split_masks(g)


def fit(g, tr, cols):
    r = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=400,
                                      max_depth=6, random_state=42)
    r.fit(g.loc[tr, cols], g.loc[tr, "count35"].to_numpy())
    return r


def main():
    g, m = load()
    tr, te = m["train"], m["test"]
    y = g["y35"].to_numpy().astype(float); win = g["window"].to_numpy()
    reg_b = fit(g, tr, BASE)
    reg_a = fit(g, tr, BASE + GNSS4)               # physical-4 (promoted set)
    reg_a5 = fit(g, tr, BASE + GNSS5)              # incl. fallback (for importance)
    Pb = lambda_to_p(np.clip(reg_b.predict(g[BASE]), EPS, None))
    Pa = lambda_to_p(np.clip(reg_a.predict(g[BASE + GNSS4]), EPS, None))
    ig_pt = information_gain(Pa[te], Pb[te], y[te])

    # --- V2c: block bootstrap of IG over test window-ids ---
    ti = np.where(te)[0]
    wte = win[ti]; yte = y[ti]
    lam_a = np.clip(p_to_lambda(Pa[ti]), EPS, None); lam_b = np.clip(p_to_lambda(Pb[ti]), EPS, None)
    lla = yte * np.log(lam_a) - lam_a; llb = yte * np.log(lam_b) - lam_b
    wins = np.sort(np.unique(wte))
    pos = np.array([yte[wte == w].sum() for w in wins])
    dll = np.array([(lla[wte == w] - llb[wte == w]).sum() for w in wins])
    rng = np.random.default_rng(SEED)
    seqs = stationary_window_indices(len(wins), B, MEAN_BLOCK, rng)
    ig_boot = np.array([dll[s].sum() / max(pos[s].sum(), 1.0) for s in seqs])
    lo, hi = float(np.percentile(ig_boot, 2.5)), float(np.percentile(ig_boot, 97.5))
    excludes0 = bool(lo > 0 or hi < 0)

    # --- V2d: permutation importance of each gnss col within the aug(5) model ---
    ig_a5 = information_gain(lambda_to_p(np.clip(reg_a5.predict(g[BASE + GNSS5]), EPS, None))[te], Pb[te], y[te])
    Xte = g.loc[te, BASE + GNSS5].copy().reset_index(drop=True)
    prng = np.random.default_rng(SEED)
    imp = {}
    for c in GNSS5:
        drops = []
        for _ in range(20):
            Xp = Xte.copy()
            Xp[c] = prng.permutation(Xp[c].to_numpy())
            Pp = lambda_to_p(np.clip(reg_a5.predict(Xp), EPS, None))
            drops.append(ig_a5 - information_gain(Pp, Pb[te], y[te]))
        imp[c] = {"mean_ig_drop": round(float(np.mean(drops)), 5),
                  "std": round(float(np.std(drops)), 5)}

    out = {
        "target": "y35", "split": "test", "aug_features": "base + gnss physical-4",
        "ig_point_estimate": round(float(ig_pt), 5),
        "V2c_bootstrap": {"B": B, "seed": SEED, "block": "stationary over window-ids",
                          "ci95": [round(lo, 5), round(hi, 5)],
                          "excludes_0": excludes0,
                          "verdict": "IG>0 supported (CI excludes 0)" if (lo > 0) else
                          ("IG<0" if hi < 0 else "CI includes 0 -> DOWNGRADE C3 to suggestive")},
        "V2d_permutation_importance_aug5": imp,
        "V2d_verdict": ("gain loads on residual-rate/strain (OK)"
                        if max(imp[c]["mean_ig_drop"] for c in GNSS4) >
                        imp["gnss_strain_fallback"]["mean_ig_drop"]
                        else "gain loads on availability flag -> ARTIFACT -> VOID C3"),
    }
    json.dump(out, open(VER / "gnss_ig.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
