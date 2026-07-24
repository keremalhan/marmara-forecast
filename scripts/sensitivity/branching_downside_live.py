"""Run 12, item C (final piece): propagate the n = 0.90 constrained fit through the live products.

The amendment specifies: "Propagate the n = 0.90 fit through the 30/90/365-d regional P(M>=6) as
ONE DOWNSIDE SCENARIO, labeled as such" -- not as an uncertainty band. The profile workers recorded
the scalar parameters but not the fitted background field, which cascade_forecast needs, so this
refits once at cap = 0.90 and keeps the params object.

Labelling, fixed by the amendment before the answer was known: this is a single downside scenario at
a branching ratio the likelihood REJECTS relative to the operational 0.95 (2[l(0.95) - l(0.90)] =
58.5). It is not a confidence bound, and the profile earned no interval (the criterion is crossed
between coarse grid points).

Writes results/round4/r12_item_C_downside.json.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/sensitivity/branching_downside_live.py
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd

import marmara.etas_fit as EF
from marmara import grid as G
from marmara.cascade import cascade_forecast
from marmara.etas_model import branching_ratio
from marmara.paths import RESULTS

R4 = RESULTS / "round4"
T0 = pd.Timestamp("2026-07-05")
CAP = 50_000
B_OP = 1.15
LEVELS = (5.5, 6.0)
HORIZONS = [("30d", 30.0, 10000), ("90d", 90.0, 10000), ("365d", 365.0, 3000)]
N_DOWN = 0.90


def main():
    t0 = time.time()
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])

    print(f"refitting at cap = n = {N_DOWN} (keeping the params object) ...", flush=True)
    cat_fit = cat[cat["datetime_utc"] < EF.FIT_END].copy()
    params, n_tgt, n_dropped, b_val, b_aki = EF.fit_stai(cat_fit, EF.MODEL_BOX, N_DOWN)
    with open(R4 / "etas_params_n090.pkl", "wb") as f:
        pickle.dump(params, f)
    n_real = float(branching_ratio(params))
    print(f"  realized n = {n_real:.6f} (pinned={abs(n_real-N_DOWN)<1e-4}), "
          f"k={params.k:.4f} alpha={params.alpha:.4f} p={params.p:.4f} ({time.time()-t0:.0f}s)",
          flush=True)
    del cat_fit

    hist = cat[cat["datetime_utc"] < T0][["datetime_utc", "longitude", "latitude", "mag_w"]]
    t0d = float(G._to_days(T0))
    base = pickle.load(open(RESULTS / "etas" / "etas_params.pkl", "rb"))

    out = {"governed_by": {"amendment": "docs/preregistration/v2_analysis_amendment_5.md",
                           "sha256": "c97db8f54374ac4ff1b5fbfafc1a1e76c63d68077144b338319603170ce846c2",
                           "item": "C (downside propagation)"},
           "label": ("ONE DOWNSIDE SCENARIO at a branching ratio the likelihood rejects relative to "
                     "the operational 0.95 (2[l(0.95)-l(0.90)] = 58.5). NOT a confidence bound; the "
                     "profile earned no interval."),
           "n_downside": N_DOWN, "n_realized": n_real, "b_op": B_OP, "t0": str(T0.date()),
           "fit_n090": {"k": params.k, "alpha": params.alpha, "c": params.c, "p": params.p,
                        "mu_total": params.mu_total,
                        "branching_mmax7.6": float(branching_ratio(params, mmax=7.6))},
           "fit_operational_n095": {"k": base.k, "alpha": base.alpha, "c": base.c, "p": base.p,
                                    "mu_total": base.mu_total,
                                    "branching_mmax7.6": float(branching_ratio(base, mmax=7.6))},
           "horizons": {}}

    for name, H, K in HORIZONS:
        rec = {}
        for lbl, prm in (("operational_n0.95", base), ("downside_n0.90", params)):
            c = cascade_forecast(prm, hist, t0d, H, G.LON_C, G.LAT_C, K=K, seed=42, b=B_OP,
                                 per_sim_cap=CAP)
            rec[lbl] = {f"P{l}": float(c[f"Preg{l}"]) for l in LEVELS}
            print(f"  {name}/{lbl}: {rec[lbl]} ({time.time()-t0:.0f}s)", flush=True)
        for l in LEVELS:
            k = f"P{l}"
            b_, d_ = rec["operational_n0.95"][k], rec["downside_n0.90"][k]
            rec[f"effect_{k}"] = {"operational": round(b_, 6), "downside": round(d_, 6),
                                 "frac_change": (round(d_ / b_ - 1.0, 5) if b_ > 0 else None)}
        out["horizons"][name] = rec

    out["runtime_s"] = round(time.time() - t0, 1)
    json.dump(out, open(R4 / "r12_item_C_downside.json", "w"), indent=2)
    print("\n=== n=0.90 downside scenario vs operational n=0.95 ===")
    for name, _, _ in HORIZONS:
        for l in LEVELS:
            e = out["horizons"][name][f"effect_P{l}"]
            print(f"  {name:>5} P(M>={l}): operational {e['operational']*100:7.3f}%  "
                  f"downside {e['downside']*100:7.3f}%  ({e['frac_change']:+.1%})")


if __name__ == "__main__":
    main()
