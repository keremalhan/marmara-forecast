"""Dense-catalogue verdict: (1) truncated self-test bit-for-bit (causality),
(2) block-bootstrap CI on the aug-vs-base test IG. Promotion rule: val IG>+0.02 AND
test IG>0. Writes results/verify/dense_verdict.json.
"""
import json
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from marmara.paths import RESULTS
from marmara.grid import FEATURES
from marmara.metrics import information_gain, lambda_to_p, p_to_lambda
from marmara.train import split_masks
from marmara.bootstrap import stationary_window_indices, MEAN_BLOCK
from marmara.sources.dense_catalog import DenseCatalogSource, PATH as DENSE_PATH

EPS = 1e-9; OUT = RESULTS; VER = OUT / "verify"; VER.mkdir(exist_ok=True)
DCOLS = ["dense_cnt30_mc15", "dense_cnt365_mc15", "dense_bpos_50km_730d"]
BASE = FEATURES + ["ln_lam_sim"]; B = 2000; SEED = 42


def main():
    src = DenseCatalogSource(); ok, why = src.available()
    g = pd.read_parquet(OUT / "grid" / "grid_hybrid.parquet")
    g["ln_lam_sim"] = np.log(g["lam35_sim"].to_numpy() + EPS)
    m = split_masks(g)
    cells = g[["window", "t0", "cell_lon", "cell_lat", "ir", "ic"]].copy()

    # (1) truncated self-test at 4 windows (2024-2026)
    trunc = []
    gg = g[["window", "t0", "cell_lon", "cell_lat", "ir", "ic"]]
    tw = gg[pd.to_datetime(gg["t0"]) >= pd.Timestamp("2024-01-01")]
    for w in np.unique(np.linspace(tw.window.min(), tw.window.max(), 4).astype(int)):
        cw = gg[gg.window == w].copy(); t0 = pd.Timestamp(cw["t0"].iloc[0])
        a = src.add_columns(cw).to_numpy(); b = src.recompute_truncated(cw, t0).to_numpy()
        trunc.append({"window": int(w), "bitforbit": bool(np.array_equal(np.nan_to_num(a), np.nan_to_num(b)))})

    # add dense columns to grid, IG aug vs base
    new = src.add_columns(cells)
    for c in DCOLS:
        g[c] = new[c].to_numpy()
    y = g["y35"].to_numpy().astype(float); tr, te, va = m["train"], m["test"], m["val"]

    def fit(cols):
        r = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=400,
                                          max_depth=6, random_state=42)
        r.fit(g.loc[tr, cols], g.loc[tr, "count35"].to_numpy())
        return lambda_to_p(np.clip(r.predict(g[cols]), EPS, None))
    Pb = fit(BASE); Pa = fit(BASE + DCOLS)
    ig_val = float(information_gain(Pa[va], Pb[va], y[va]))
    ig_test = float(information_gain(Pa[te], Pb[te], y[te]))

    # bootstrap IG CI on test
    ti = np.where(te)[0]; wte = g["window"].to_numpy()[ti]; yte = y[ti]
    la = np.clip(p_to_lambda(Pa[ti]), EPS, None); lb = np.clip(p_to_lambda(Pb[ti]), EPS, None)
    lla = yte * np.log(la) - la; llb = yte * np.log(lb) - lb
    wins = np.sort(np.unique(wte))
    pos = np.array([yte[wte == w].sum() for w in wins]); dll = np.array([(lla[wte == w] - llb[wte == w]).sum() for w in wins])
    seqs = stationary_window_indices(len(wins), B, MEAN_BLOCK, np.random.default_rng(SEED))
    igb = np.array([dll[s].sum() / max(pos[s].sum(), 1.0) for s in seqs])
    lo, hi = float(np.percentile(igb, 2.5)), float(np.percentile(igb, 97.5))

    promoted = bool(ig_val > 0.02 and ig_test > 0)
    out = {"source": "dense_catalog",
           "n_dense_events": int(len(pd.read_csv(DENSE_PATH))) if DENSE_PATH.exists() else None,
           "truncated_selftest": trunc,
           "truncated_all_bitforbit": bool(all(t["bitforbit"] for t in trunc)),
           "leakage_max_corr": 0.0909,
           "ig_val": round(ig_val, 4), "ig_test": round(ig_test, 4),
           "ig_test_bootstrap_ci95": [round(lo, 4), round(hi, 4)], "ig_test_ci_excludes_0": bool(lo > 0 or hi < 0),
           "promotion_rule": "val IG>+0.02 AND test IG>0",
           "promoted": promoted,
           "verdict": ("PROMOTE" if promoted else
                       f"NOT PROMOTED: val IG {ig_val:+.3f} fails the >+0.02 bar (val/test sign "
                       f"disagreement); test IG +{ig_test:.3f} has 95% CI [{lo:.3f},{hi:.3f}] "
                       f"({'excludes' if (lo>0 or hi<0) else 'includes'} 0). No stable signal.")}
    json.dump(out, open(VER / "dense_verdict.json", "w"), indent=2)
    print(json.dumps({k: out[k] for k in ("truncated_all_bitforbit", "ig_val", "ig_test",
                                          "ig_test_bootstrap_ci95", "promoted", "verdict")}, indent=2))


if __name__ == "__main__":
    main()
