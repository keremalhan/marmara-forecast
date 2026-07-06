"""Honest information-gain test for a multi-source feature extension.

For a FeatureSource, it: (1) adds the source's causal columns to the hybrid grid, (2) runs
the leakage gate on the new columns (no column may correlate >0.999 with a target),
(3) retrains the Poisson HistGB (the hybrid's ML stage) WITH vs WITHOUT the columns
and reports IG(augmented - base) per event on val AND test. Positive test IG = the
source adds real predictive information; ~0 = no gain (reported plainly, no theater).

`--validate` runs three CONTROL columns to prove the harness works:
  noise        -> leakage PASS, IG ~ 0        (does not hallucinate gain)
  leaky (y35)  -> leakage FAIL (corr ~ 1)      (catches target leakage)
  informative  -> leakage PASS, IG > 0         (detects real information;
                  held-out etas_rate re-added to a base that excludes it)

Run:  "<venv>/bin/python3" -m marmara.source_ig_test validate
      "<venv>/bin/python3" -m marmara.source_ig_test <source_name>
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401
from sklearn.ensemble import HistGradientBoostingRegressor

from marmara.grid import FEATURES
from marmara.metrics import information_gain, lambda_to_p
from marmara.sources.base import causal_corr_check
from marmara.sources.dense_catalog import DenseCatalogSource
from marmara.sources.gnss_coupling import GnssCouplingSource
from marmara.sources.repeating_eq import RepeatingEqSource
from marmara.train import split_masks

OUT = RESULTS
EPS = 1e-9
BASE = FEATURES + ["ln_lam_sim"]
SOURCES = {s.name: s for s in (GnssCouplingSource(), RepeatingEqSource(), DenseCatalogSource())}


def load_grid():
    g = pd.read_parquet(OUT / "grid_hybrid.parquet")
    g["ln_lam_sim"] = np.log(g["lam35_sim"].to_numpy() + EPS)
    return g, split_masks(g)


def _lam(Xcols, g, masks):
    tr = masks["train"]
    reg = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=400,
                                        max_depth=6, random_state=42)
    reg.fit(g.loc[tr, Xcols], g.loc[tr, "count35"].to_numpy())
    return np.clip(reg.predict(g[Xcols]), EPS, None)


def ig_test(new_cols: pd.DataFrame, g, masks, base_cols, label):
    y = g["y35"].to_numpy().astype(float)
    leak = causal_corr_check(new_cols, g[["y35", "y45"]])
    aug = g.copy()
    for c in new_cols.columns:
        aug[c] = new_cols[c].to_numpy()
    lam_base = _lam(base_cols, aug, masks)
    lam_aug = _lam(base_cols + list(new_cols.columns), aug, masks)
    P_base, P_aug = lambda_to_p(lam_base), lambda_to_p(lam_aug)
    res = {"label": label, "columns": list(new_cols.columns),
           "leakage": {"max_abs_corr": round(leak["max_abs_corr"], 4), "passes": leak["passes"]},
           "ig_aug_vs_base": {}}
    for split in ("val", "test"):
        idx = masks[split]
        res["ig_aug_vs_base"][split] = round(
            information_gain(P_aug[idx], P_base[idx], y[idx]), 4)
    res["n_new_cols_all_nan"] = int(new_cols.isna().all().sum())
    return res


def evaluate_source(name):
    g, masks = load_grid()
    src = SOURCES[name]
    ok, why = src.available()
    spec = {"source": name, "data_spec": src.data_spec, "available": ok, "reason": why}
    if not ok:
        spec["status"] = "SKIPPED — data not present; drop it in and re-run (no fabrication)"
        return spec
    cells = g[["window", "t0", "cell_lon", "cell_lat", "ir", "ic"]].copy()
    new = src.add_columns(cells)
    spec.update(ig_test(new, g, masks, BASE, name))
    spec["verdict"] = ("adds information" if spec["ig_aug_vs_base"]["test"] > 0.01
                       and spec["leakage"]["passes"] else
                       "leakage!" if not spec["leakage"]["passes"] else "no measurable gain")
    return spec


def validate():
    g, masks = load_grid()
    rng = np.random.default_rng(42)
    results = {}
    # noise: deterministic per (window,ir,ic) so it is causal & reproducible, no signal
    key = (g["window"].to_numpy() * 100003 + g["ir"].to_numpy() * 53 + g["ic"].to_numpy())
    noise = ((key * 2654435761) % 1_000_000) / 1_000_000.0
    results["noise"] = ig_test(pd.DataFrame({"ctrl_noise": noise}, index=g.index), g, masks, BASE, "noise")
    # leaky: the target itself -> must fail the leakage gate
    results["leaky"] = ig_test(pd.DataFrame({"ctrl_leaky": g["y35"].to_numpy(float)}, index=g.index),
                               g, masks, BASE, "leaky")
    # informative: hold out etas_rate from the base, re-add it as a "source" -> IG>0
    base_ho = [c for c in BASE if c != "etas_rate"]
    results["informative"] = ig_test(pd.DataFrame({"ctrl_info": g["etas_rate"].to_numpy()}, index=g.index),
                                     g, masks, base_ho, "informative(etas_rate held-out)")

    checks = {
        "noise_no_gain": abs(results["noise"]["ig_aug_vs_base"]["test"]) < 0.05 and results["noise"]["leakage"]["passes"],
        "leaky_caught": not results["leaky"]["leakage"]["passes"],
        "informative_gain": results["informative"]["ig_aug_vs_base"]["test"] > 0.01 and results["informative"]["leakage"]["passes"],
    }
    results["harness_validation"] = {**checks, "all_pass": all(checks.values())}
    return results


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "validate"
    if arg == "validate":
        out = validate()
        json.dump(out, open(OUT / "source_ig_validation.json", "w"), indent=2)
        v = out["harness_validation"]
        print("HARNESS VALIDATION (controls):")
        print(f"  noise  -> test IG {out['noise']['ig_aug_vs_base']['test']:+.4f}, leak-pass {out['noise']['leakage']['passes']}  [{ 'OK' if v['noise_no_gain'] else 'FAIL'}]")
        print(f"  leaky  -> max corr {out['leaky']['leakage']['max_abs_corr']}, leak-pass {out['leaky']['leakage']['passes']}  [{'OK' if v['leaky_caught'] else 'FAIL'}]")
        print(f"  info   -> test IG {out['informative']['ig_aug_vs_base']['test']:+.4f}  [{'OK' if v['informative_gain'] else 'FAIL'}]")
        print(f"  ALL PASS: {v['all_pass']}")
    else:
        out = evaluate_source(arg)
        json.dump(out, open(OUT / f"source_ig_{arg}.json", "w"), indent=2)
        if not out["available"]:
            print(f"{arg}: SKIPPED — {out['reason']}\n  need: {out['data_spec'].get('fetch','')}")
        else:
            print(f"{arg}: leakage-pass={out['leakage']['passes']} (max corr {out['leakage']['max_abs_corr']}); "
                  f"test IG {out['ig_aug_vs_base']['test']:+.4f} val IG {out['ig_aug_vs_base']['val']:+.4f} "
                  f"-> {out['verdict']}")


if __name__ == "__main__":
    main()
