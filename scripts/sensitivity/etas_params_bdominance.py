"""Run 8 — Full ETAS parameter table + the "b dominates" number.

PART A. Extract the complete fit into one table: every EtasParams field, the background
normalization/floor, the branching cap and constraints, the optimizer and its bounds, the
convergence diagnostics, and parameter uncertainties IF retrievable.

PART B. The limitations paragraph asserts that b (the operational magnitude law) dominates the
triggering parameters in the live products, without a number. We give it one: re-run the regional
P(M>=5.5) and P(M>=6) at 30/90/365 d under
  (i)   baseline params at b_op = 1.15                    [reproduces the published live product]
  (ii)  params perturbed by the Mc-3.65 refit deltas      [k -8%, alpha -3%, p +0.5%]
  (iii) baseline params at b_aki = 1.0186 and b_pos = 1.5424   [the published b-ensemble]
and compare the fractional change from (ii) against the span of (iii).

Uncertainty source. The fit is L-BFGS-B on a reparameterized objective (ln mu, ln n, alpha,
ln c, ln(p-1), ln d, ln(q-1), gamma) with the branching ratio PINNED at the 0.95 cap. A pinned
(active-bound) parameter has no interior Hessian, so a curvature-based +/-1 sigma is not defined
for the productivity scale; scipy's L-BFGS-B `hess_inv` is a limited-memory approximation and is
not persisted in etas_params.pkl in any case. We therefore use the Mc-3.65 refit deltas -- an
empirical, end-to-end perturbation of exactly the triggering parameters at issue -- as pre-specified
in the task, and say so rather than manufacturing a sigma.

Writes results/round4/r8_etas_params_bdominance.json.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/sensitivity/etas_params_bdominance.py
"""
from __future__ import annotations

import inspect
import json
import pickle
import time
from dataclasses import replace

import numpy as np
import pandas as pd

from marmara import grid as G
from marmara.cascade import cascade_forecast
from marmara.etas_model import LN10, branching_ratio
from marmara.paths import RESULTS

MMAX = 7.6
T0 = pd.Timestamp("2026-07-05")           # the published live epoch
CAP = 50_000
LEVELS = (5.5, 6.0)
HORIZONS = [("30d", 30.0, 10000), ("90d", 90.0, 10000), ("365d", 365.0, 3000)]
OUT = RESULTS / "round4"
OUT.mkdir(exist_ok=True)

# Mc-3.65 refit deltas (docs/round2_adversarial_changelog.md finding 5; results/catalog/mc_sensitivity.json)
DELTAS = {"k": -0.08, "alpha": -0.03, "p": +0.005}


def main():
    t_all = time.time()
    p = pickle.load(open(RESULTS / "etas" / "etas_params.pkl", "rb"))
    rep = json.load(open(RESULTS / "etas" / "etas_fit_report.json"))
    mcs = json.load(open(RESULTS / "catalog" / "mc_sensitivity.json"))
    b_op = float(rep["operational_b_for_cascade"])
    b_aki = float(rep["b_aki"]); b_pos = float(rep["b_positive"])

    bg = p.background_xy
    bg_desc = {}
    for a in ("kind", "bandwidth_km", "h_km", "floor", "uniform_floor", "n_points", "region"):
        if hasattr(bg, a):
            v = getattr(bg, a)
            bg_desc[a] = (v if isinstance(v, (int, float, str, dict, type(None)))
                          else f"<{type(v).__name__}>")
    bg_desc["class"] = type(bg).__name__
    bg_desc["attrs_present"] = [a for a in vars(bg)] if hasattr(bg, "__dict__") else []

    # exact deltas implied by the Mc-3.65 refit, recomputed (not trusted from prose)
    f30, f365 = mcs["fit_mc3.0"], mcs["fit_mc3.65"]
    measured_deltas = {k: round(f365[k] / f30[k] - 1.0, 6) for k in ("k", "alpha", "p", "c", "mu_total")}

    table = {
        "productivity_k": p.k, "alpha": p.alpha, "omori_c_days": p.c, "omori_p": p.p,
        "spatial_kernel_d_km2_at_mc": p.d, "spatial_power_q": p.q,
        "spatial_magnitude_scaling_gamma_log10": p.gamma,
        "mc_fit": p.mc, "b_fit_used_in_params": p.b,
        "mu_total_events_per_day_ge_mc": p.mu_total,
        "region": p.region,
        "background_field": bg_desc,
        "branching_untruncated": rep["operational_branching_untrunc"],
        "branching_mmax7.6": rep["operational_branching_mmax7.6"],
        "branching_cap_operational": rep["operational_cap"],
        "diagnostic_fit_cap_0.999_branching": rep["diagnostic_fit_cap_0.999_branching"],
        "re_pinned_at_raised_cap": rep["re_pinned_at_raised_cap"],
        "n_events_fit": rep["n_events_fit"], "n_dropped_by_stai": rep["n_dropped_by_stai"],
        "converged": rep["converged"],
        "b_positive": b_pos, "b_aki": b_aki, "b_operational_for_cascade": b_op,
        "b_ensemble_for_cascade": rep["b_ensemble_for_cascade"],
        "optimizer": {
            "method": "L-BFGS-B", "jac": "analytic (_nll_grad, jac=True)",
            "maxiter": 200, "maxfun": 800, "stages": "2 (refit from best.x, maxiter 150)",
            "background_em_iterations": 2,
            "parameterization": ["ln mu", "ln n (branching ratio)", "alpha", "ln c",
                                 "ln (p-1)", "ln d", "ln (q-1)", "gamma"],
            "bounds": {
                "ln_mu": ["ln 1e-4", "ln 200"], "ln_n": ["ln 1e-3", "ln cap (0.95)"],
                "alpha": [0.05, "min(3.2, beta - 0.02)"], "ln_c": ["ln 5e-4", "ln 3.0"],
                "ln_p_minus_1": ["ln 0.02", "ln 1.5  -> p in [1.02, 2.5]"],
                "ln_d": ["ln 1e-3", "ln 400"], "ln_q_minus_1": ["ln 0.08", "ln 2.5 -> q in (1.08, 3.5]"],
            },
            "key_constraint": ("productivity parameterized through the untruncated branching ratio "
                               "n = k*beta/(beta-alpha), bounded n <= 0.95 -> subcritical, "
                               "stationary; blocks the degenerate p->1 / large-k MLE mode"),
        },
        "uncertainties": {
            "hessian_available": False,
            "why": ("the branching ratio is PINNED at the 0.95 active bound (n = 0.94999...), so the "
                    "productivity scale has no interior curvature and a curvature-based sigma is "
                    "undefined there; scipy L-BFGS-B's hess_inv is a limited-memory approximation "
                    "and is not persisted in etas_params.pkl. Evidence of the pin: raising the cap "
                    "to 0.999 re-pinned at the new cap (n = 0.9990), the documented degenerate "
                    "near-critical mode."),
            "substitute": ("Mc-3.65 refit deltas -- an end-to-end empirical perturbation of the "
                           "triggering parameters (k, alpha, p)"),
            "measured_refit_deltas_mc3.0_to_mc3.65": measured_deltas,
            "applied_deltas": DELTAS,
        },
    }

    # ---- Part B: perturbation vs b-ensemble ----
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    hist = cat[cat["datetime_utc"] < T0][["datetime_utc", "longitude", "latitude", "mag_w"]]
    t0d = float(G._to_days(T0))

    p_pert = replace(p, k=p.k * (1 + DELTAS["k"]), alpha=p.alpha * (1 + DELTAS["alpha"]),
                     p=p.p * (1 + DELTAS["p"]))
    table["perturbed_params"] = {"k": p_pert.k, "alpha": p_pert.alpha, "p": p_pert.p,
                                 "branching_mmax7.6_perturbed": float(branching_ratio(p_pert, mmax=MMAX)),
                                 "branching_mmax7.6_baseline": float(branching_ratio(p, mmax=MMAX))}

    def run(params, b, H, K):
        c = cascade_forecast(params, hist, t0d, H, G.LON_C, G.LAT_C, K=K, seed=42, b=b,
                             per_sim_cap=CAP)
        return {f"P{lvl}": float(c[f"Preg{lvl}"]) for lvl in LEVELS}

    conds = [("baseline_b_op", p, b_op), ("perturbed_b_op", p_pert, b_op),
             ("baseline_b_aki", p, b_aki), ("baseline_b_pos", p, b_pos)]
    res = {}
    for name, H, K in HORIZONS:
        res[name] = {"horizon_days": H, "K": K, "conditions": {}}
        for cname, prm, b in conds:
            res[name]["conditions"][cname] = run(prm, b, H, K)
            print(f"  {name}/{cname}: {res[name]['conditions'][cname]} ({time.time()-t_all:.0f}s)",
                  flush=True)
        c = res[name]["conditions"]
        for lvl in LEVELS:
            k = f"P{lvl}"
            base = c["baseline_b_op"][k]
            pert = c["perturbed_b_op"][k]
            blo, bhi = c["baseline_b_pos"][k], c["baseline_b_aki"][k]   # high b -> low P
            b_span = abs(bhi - blo)
            par_eff = abs(pert - base)
            res[name][f"effect_{k}"] = {
                "baseline": round(base, 6), "perturbed": round(pert, 6),
                "param_abs_change": round(par_eff, 6),
                "param_frac_change": (round(pert / base - 1.0, 5) if base > 0 else None),
                "b_ensemble_lo_at_b_pos": round(blo, 6), "b_ensemble_hi_at_b_aki": round(bhi, 6),
                "b_span_abs": round(b_span, 6),
                "b_span_ratio_hi_over_lo": (round(bhi / blo, 2) if blo > 0 else None),
                "param_effect_as_frac_of_b_span": (round(par_eff / b_span, 5) if b_span > 0 else None),
                "b_effect_over_param_effect": (round(b_span / par_eff, 1) if par_eff > 0 else None),
            }
    out = {"t0": str(T0.date()), "b_op": b_op, "b_ensemble": [b_aki, b_pos],
           "per_sim_cap": CAP, "parameter_table": table, "horizons": res,
           "published_live_30d": {"P_M6_central_b_op": 0.015381748828067976,
                                  "P_M6_range": [0.0012992202859282154, 0.0373873550360424],
                                  "P_M5.5_range": [0.006578734804763564, 0.11717986714016237],
                                  "source": "results/forecast/forecast_2026-07-05/forecast_summary.json"},
           "runtime_s": round(time.time() - t_all, 1)}
    e30 = res["30d"]["effect_P6.0"]
    out["gate"] = {
        "b_effect_dominates_30d_P6": bool(e30["b_effect_over_param_effect"] is not None
                                          and e30["b_effect_over_param_effect"] > 5),
        "30d_P6_param_frac_change": e30["param_frac_change"],
        "30d_P6_b_span_ratio": e30["b_span_ratio_hi_over_lo"],
        "30d_P6_param_effect_as_frac_of_b_span": e30["param_effect_as_frac_of_b_span"],
        "baseline_reproduces_published_1.54pct": bool(abs(e30["baseline"] - 0.01538) < 0.004),
    }
    json.dump(out, open(OUT / "r8_etas_params_bdominance.json", "w"), indent=2)

    print("\n=== P(M>=6) / P(M>=5.5): parameter perturbation vs b-ensemble ===")
    for name, _, _ in HORIZONS:
        for lvl in LEVELS:
            e = res[name][f"effect_P{lvl}"]
            print(f"{name:>5} P(M>={lvl}): base {e['baseline']*100:7.3f}%  pert {e['perturbed']*100:7.3f}% "
                  f"({e['param_frac_change']:+.2%})  | b-span [{e['b_ensemble_lo_at_b_pos']*100:.3f}%, "
                  f"{e['b_ensemble_hi_at_b_aki']*100:.3f}%] = {e['b_span_ratio_hi_over_lo']}x  "
                  f"| b/param = {e['b_effect_over_param_effect']}x")
    print(f"\nGATE: {json.dumps(out['gate'])}")


if __name__ == "__main__":
    main()
