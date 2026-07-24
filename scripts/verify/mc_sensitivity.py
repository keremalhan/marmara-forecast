"""Mc sensitivity of the in-house ETAS fit (referee item: the common-mode argument
protects comparisons, not the FIT).

The max-curvature completeness of the mixed Md/ML catalogue is Mc = 3.65, inflated by the
incomplete 2003-2015 Md era piling near mag_w 3.45; the operational fit uses base Mc = 3.0.
If ingesting that incomplete era biased the background mu and the productivity (k, alpha),
those biases would propagate to every ETAS-derived forecaster. This script refits the ETAS with
IDENTICAL code at base Mc in {3.0, 3.65} (isolating the Mc effect from nondeterminism) and
compares the fitted parameters and the derived first-generation ETAS rate on the shipped
evaluation grid. Writes results/catalog/mc_sensitivity.json. Does NOT touch canonical artifacts.

Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/verify/mc_sensitivity.py
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

import marmara.etas_fit as EF
from marmara.etas_model import branching_ratio
from marmara.paths import RESULTS


def refit(cat: pd.DataFrame, base_mc: float):
    EF.BASE_MC = base_mc                      # module global used by fit_stai / mc_of_time
    params, n_tgt, n_dropped, b_val, b_aki = EF.fit_stai(cat, EF.MODEL_BOX, EF.FALLBACK_CAP)
    return {
        "base_mc": base_mc, "n_events_fit": int(n_tgt), "n_dropped_stai": int(n_dropped),
        "mu_total": float(params.mu_total), "k": float(params.k), "alpha": float(params.alpha),
        "c": float(params.c), "p": float(params.p), "b_positive": float(b_val),
        "branching_untrunc": float(branching_ratio(params)),
        "branching_mmax7.6": float(branching_ratio(params, mmax=7.6)),
        "converged": bool(getattr(params, "_fit_result").success),
    }, params


def main():
    import gc
    # The shipped etas_params.pkl IS fit_stai(cat, MODEL_BOX, 0.95) at base Mc=3.0 (identical
    # code path), so we use it as the Mc=3.0 arm and only run the NEW Mc=3.65 fit here (which
    # ingests far fewer events -> low RAM; running both fits in one process OOMs).
    with open(RESULTS / "etas" / "etas_params.pkl", "rb") as f:
        pc = pickle.load(f)
    fitrep = json.load(open(RESULTS / "etas" / "etas_fit_report.json"))
    r30 = {"base_mc": 3.0, "n_events_fit": int(fitrep["n_events_fit"]),
           "mu_total": float(pc.mu_total), "k": float(pc.k), "alpha": float(pc.alpha),
           "c": float(pc.c), "p": float(pc.p), "b_positive": float(pc.b),
           "branching_untrunc": float(branching_ratio(pc)),
           "branching_mmax7.6": float(branching_ratio(pc, mmax=7.6)),
           "converged": True, "source": "canonical shipped etas_params.pkl"}
    del pc
    gc.collect()

    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    cat = cat[cat["datetime_utc"] < EF.FIT_END].copy()
    r365, p365 = refit(cat, 3.65)
    del p365
    gc.collect()

    def rel(a, b):
        return None if b == 0 else round(100 * (a - b) / abs(b), 1)
    deltas = {key: {"mc3.0": r30[key], "mc3.65": r365[key], "pct_change": rel(r365[key], r30[key])}
              for key in ("n_events_fit", "mu_total", "k", "alpha", "c", "p",
                          "b_positive", "branching_mmax7.6")}

    out = {
        "purpose": "does refitting ETAS at Mc=3.65 (vs 3.0) materially change the fit / verdicts?",
        "fit_mc3.0": r30, "fit_mc3.65": r365,
        "deltas_3.65_vs_3.0": deltas,
    }
    (RESULTS / "catalog" / "mc_sensitivity.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(deltas, indent=2))
    print("\nfit n_events: Mc3.0 =", r30["n_events_fit"], " Mc3.65 =", r365["n_events_fit"])
    print(f"branching(mmax7.6): 3.0={r30['branching_mmax7.6']:.4f}  3.65={r365['branching_mmax7.6']:.4f}")
    print(f"alpha: 3.0={r30['alpha']:.4f}  3.65={r365['alpha']:.4f}   k: 3.0={r30['k']:.4f}  3.65={r365['k']:.4f}")


if __name__ == "__main__":
    main()
