"""Run 12 — In-house CSEP cross-check at the FINAL configuration (blocker triage).

THE CONTRADICTION. §3 states: "The N and M statistics are also implemented independently as a
cross-check, and the two agree for every model." But results/csep/csep_results.json (in-house)
stores cascade N-test delta1 = 0.013 (REJECT) where results/csep/pycsep_results.json stores
delta1 = 0.094 (ACCEPT) -- opposite verdicts on a headline consistency claim.

TWO CANDIDATE CAUSES, and they have very different consequences:
  (A) STALE b. marmara/csep_eval.py hardcodes MODELS_B = {"cascade": 1.2, "sv_etas": 1.2, ...},
      i.e. the OLD b_op; the file records "b_gr": 1.2. Final config is b_op = 1.15. If this is
      the cause, re-running at 1.15 fixes it and the §3 sentence stands.
  (B) STRUCTURAL. csep_eval's own docstring says it builds catalogues by POISSON-sampling each
      cell's count from Lambda(cell): "Poisson cell-sampling assumes within-period
      cell-independence, so it tests the RATE forecast; the cascade's clustering is carried in
      Lambda but not in the count over-dispersion (documented approximation vs native clustered
      catalogues)." pyCSEP instead scores the simulator's NATIVE clustered catalogues. §4 already
      says the cascade's N-test pass "rests on the clustered N-test's dispersion -- aftershock
      sequences widen the count distribution well beyond" Poisson. If this is the cause, the two
      CANNOT agree on N at any b, and the §3 sentence is simply false.

THE DECISIVE TEST. Re-run the in-house N/M at b_op = 1.15, changing nothing else. Under (A) the
cascade's delta1 moves to ~0.094. Under (B) it stays ~0.01 while the MEAN stays ~1,305, and the
Poisson-vs-clustered dispersion gap is exposed directly by comparing each test's simulated count
standard deviation against sqrt(mean) (the Poisson expectation).

Writes results/round4/r12_csep_inhouse_recheck.json. Does NOT touch results/csep/*.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/csep/inhouse_recheck.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from marmara import csep_eval as CE
from marmara.paths import RESULTS

OUT = RESULTS / "round4"
OUT.mkdir(exist_ok=True)

B_FINAL = {"cascade": 1.15, "sv_etas": 1.15, "modern_etas": 1.762}   # inversion keeps its own fit b
B_STALE = dict(CE.MODELS_B)


def native_count_dispersion():
    """Count distribution of the NATIVE clustered catalogues actually fed to pyCSEP, so the
    Poisson-vs-clustered dispersion gap is measured, not asserted."""
    out = {}
    for model in ("cascade", "sv_etas", "modern_etas"):
        p = RESULTS / "csep" / "inputs" / f"{model}_catalogs.npz"
        if not p.exists():
            continue
        d = np.load(p)
        cid = d["catalog_id"]; n_sim = int(d["n_sim"])
        counts = np.bincount(cid, minlength=n_sim).astype(float)
        out[model] = {"n_sim": n_sim, "mean": round(float(counts.mean()), 2),
                      "std": round(float(counts.std(ddof=1)), 2),
                      "poisson_std_sqrt_mean": round(float(np.sqrt(counts.mean())), 2),
                      "overdispersion_ratio": round(float(counts.std(ddof=1) / np.sqrt(counts.mean())), 2),
                      "delta1_P_X_ge_1383": round(float((counts >= 1383).mean()), 4)}
    return out


def main():
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    test_wins = CE._test_windows(grid)
    lams = CE.model_lambda(grid, test_wins)
    obs_counts, obs_mags, period = CE.observed(grid, test_wins)
    n_obs = int(obs_counts.sum())
    print(f"test period {period[0]}..{period[1]}, {n_obs} observed M>=3.0, "
          f"{len(test_wins)} windows, models {list(lams)}", flush=True)

    res = {"period": period, "n_observed": n_obs, "n_windows": int(len(test_wins)),
           "N_sim_inhouse": CE.N_SIM, "seed": CE.SEED,
           "b_stale_hardcoded": B_STALE, "b_final": B_FINAL,
           "inhouse_method": ("Poisson-sample each cell's count from Lambda(cell) = sum of lam30 "
                              "over the 26 test windows; magnitudes from GR(b, Mc=3.0, Mmax=7.6). "
                              "Per its own docstring this tests the RATE forecast and does NOT "
                              "carry the clustered count over-dispersion."),
           "pycsep_method": "score the simulator's NATIVE clustered catalogues (500 per model)",
           "runs": {}}

    for tag, B in (("stale_b_1.2", B_STALE), ("final_b_1.15", B_FINAL)):
        res["runs"][tag] = {}
        for model, lam in lams.items():
            rng = np.random.default_rng(CE.SEED)
            r = CE.run_model(lam, B[model], obs_counts, obs_mags, rng)
            res["runs"][tag][model] = r
            print(f"  [{tag:13s}] {model:12s} b={B[model]:5} mean {r['N_forecast_mean']:7} "
                  f"N d1 {r['N_test']['delta1']:6} d2 {r['N_test']['delta2']:6} "
                  f"pass {r['N_test']['pass']} | M gamma {r['M_test']['gamma']:6} "
                  f"pass {r['M_test']['pass']}", flush=True)

    # published pyCSEP values, for the side-by-side
    py = json.load(open(RESULTS / "csep" / "pycsep_results.json"))
    res["pycsep_published"] = py

    res["native_clustered_count_dispersion"] = native_count_dispersion()

    # ---- verdict ----
    ih_stale = res["runs"]["stale_b_1.2"]["cascade"]
    ih_final = res["runs"]["final_b_1.15"]["cascade"]
    disp = res["native_clustered_count_dispersion"].get("cascade", {})
    d1_final = ih_final["N_test"]["delta1"]
    res["verdict"] = {
        "inhouse_cascade_delta1_stale_b": ih_stale["N_test"]["delta1"],
        "inhouse_cascade_delta1_final_b": d1_final,
        "pycsep_cascade_delta1": 0.094,
        "b_refresh_reconciles_N": bool(abs(d1_final - 0.094) < 0.02),
        "cause": None, "sec3_sentence_status": None,
    }
    if abs(d1_final - 0.094) < 0.02:
        res["verdict"]["cause"] = "STALE b -- refreshing to b_op=1.15 reconciles the N-test"
        res["verdict"]["sec3_sentence_status"] = "STANDS once csep_results.json is regenerated"
    else:
        res["verdict"]["cause"] = (
            "STRUCTURAL -- the in-house test Poisson-samples cell counts and cannot reproduce the "
            "clustered count over-dispersion that the native catalogues carry. The MEAN agrees "
            "(the rate forecast is the same); only the count DISPERSION differs, which is exactly "
            "what the N-test integrates over. b is also stale (1.2 vs 1.15), but refreshing it "
            "does not reconcile the N-test.")
        res["verdict"]["sec3_sentence_status"] = (
            "FALSE as written -- '…and the two agree for every model' must be corrected. §4 "
            "already states the opposite ('the pass rests on the clustered N-test's dispersion'), "
            "so §3 contradicts §4, not just the artifact.")
    res["verdict"]["mean_agreement"] = {
        "inhouse_final_b_mean": ih_final["N_forecast_mean"],
        "pycsep_native_mean": disp.get("mean"),
        "means_agree_within_1pct": (bool(abs(ih_final["N_forecast_mean"] - disp["mean"])
                                         / disp["mean"] < 0.01) if disp else None),
    }
    res["verdict"]["dispersion_gap"] = disp

    json.dump(res, open(OUT / "r12_csep_inhouse_recheck.json", "w"), indent=2)
    print("\n=== NATIVE CLUSTERED COUNT DISPERSION (what pyCSEP scores) ===")
    for m, v in res["native_clustered_count_dispersion"].items():
        print(f"  {m:12s} mean {v['mean']:8.2f}  std {v['std']:7.2f}  "
              f"Poisson std sqrt(mean) {v['poisson_std_sqrt_mean']:6.2f}  "
              f"overdispersion {v['overdispersion_ratio']:5.2f}x  "
              f"delta1(P(X>=1383)) {v['delta1_P_X_ge_1383']}")
    print(f"\nVERDICT: {json.dumps(res['verdict'], indent=1)}")


if __name__ == "__main__":
    main()
