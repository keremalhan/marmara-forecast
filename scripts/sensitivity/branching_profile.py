"""Run 12, item C (Amendment 5, SHA-256 c97db8f5...): branching-ratio profile — aggregation.

Collects the per-n constrained refits (one process each, .tmp/r12C/n_*.json) and applies the
NAMING RULE fixed in the amendment BEFORE the answers were known:

  * if the grid supports it, state a ONE-SIDED lower bound with the boundary-adjusted LR criterion
    (optimum at an active bound -> the LR statistic follows a 50:50 chi2_0/chi2_1 mixture, so the
    one-sided 95% critical value is chi2_1(0.90) = 2.706, not 3.841);
  * if ANY constrained refit fails to converge, or the criterion is crossed BETWEEN two coarse grid
    points rather than at one, the item is named a CONSTRAINED BRANCHING-RATIO SENSITIVITY and no
    interval is stated. We do not interpolate a bound out of a six-point grid.

Reference. There is no interior MLE: the likelihood is monotone in n and the fit pins at whatever
cap it is given (0.95 -> 0.95; 0.999 -> 0.9990). We therefore report the profile against BOTH the
operational n = 0.95 and the degenerate n = 0.999, and let the naming rule decide what may be said.

Writes results/round4/r12_item_C.json.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/sensitivity/branching_profile.py
"""
from __future__ import annotations

import glob
import json

from marmara.paths import RESULTS

R4 = RESULTS / "round4"
CRIT_ONE_SIDED_95 = 2.706          # chi2_1(0.90): the 50:50 boundary mixture
GRID = [0.80, 0.85, 0.88, 0.90, 0.92, 0.94, 0.95, 0.999]


def main():
    recs = {}
    for p in sorted(glob.glob(str(RESULTS.parent / ".tmp" / "r12C" / "n_*.json"))):
        r = json.load(open(p))
        recs[round(r["n_fixed_cap"], 4)] = r
    ns = sorted(recs)
    assert ns, "no worker outputs found"

    ll = {n: recs[n]["loglik"] for n in ns}
    all_conv = all(recs[n]["converged"] for n in ns)
    all_pinned = all(recs[n]["pinned_at_cap"] for n in ns)
    n_hat_op = 0.95 if 0.95 in ll else max(ns)
    n_hat_deg = 0.999 if 0.999 in ll else max(ns)

    rows = []
    for n in ns:
        rows.append({
            "n": n,
            "loglik": round(ll[n], 4),
            "converged": recs[n]["converged"],
            "pinned_at_cap": recs[n]["pinned_at_cap"],
            "realized_n": round(recs[n]["n_realized_untruncated"], 6),
            "LR_vs_0.95": round(2 * (ll[n_hat_op] - ll[n]), 4),
            "LR_vs_0.999": round(2 * (ll[n_hat_deg] - ll[n]), 4),
            "k": round(recs[n]["k"], 4), "alpha": round(recs[n]["alpha"], 4),
            "p": round(recs[n]["p"], 4), "mu_total": round(recs[n]["mu_total"], 4),
        })

    # where does the one-sided criterion get crossed, against the operational reference?
    prof = [(r["n"], r["LR_vs_0.95"]) for r in rows if r["n"] <= 0.95]
    crossed_at = None
    between = None
    for i in range(len(prof) - 1):
        lo, hi = prof[i], prof[i + 1]
        if lo[1] >= CRIT_ONE_SIDED_95 > hi[1]:
            between = [lo[0], hi[0]]
        if abs(lo[1] - CRIT_ONE_SIDED_95) < 1e-6:
            crossed_at = lo[0]
    monotone = all(prof[i][1] >= prof[i + 1][1] - 1e-9 for i in range(len(prof) - 1))

    # the naming rule, applied exactly as written
    if not all_conv:
        naming = "constrained branching-ratio sensitivity"
        reason = "at least one constrained refit failed to converge"
        interval = None
    elif not all_pinned:
        naming = "constrained branching-ratio sensitivity"
        reason = ("at least one refit did not pin at its cap, so that point is not a constrained "
                  "fit at the intended n")
        interval = None
    elif crossed_at is not None:
        naming = "one-sided lower bound"
        reason = "the criterion is crossed exactly at a grid point"
        interval = {"lower_bound_n": crossed_at, "criterion": CRIT_ONE_SIDED_95,
                    "caveat": "boundary-adjusted: 50:50 chi2_0/chi2_1 mixture, optimum at an active bound"}
    else:
        naming = "constrained branching-ratio sensitivity"
        reason = (f"the criterion is crossed BETWEEN coarse grid points "
                  f"({between[0]} and {between[1]}), not at one; the amendment forbids "
                  f"interpolating a bound out of a six-point grid"
                  if between else
                  "the criterion is not crossed anywhere on the grid")
        interval = None

    out = {
        "governed_by": {"amendment": "docs/preregistration/v2_analysis_amendment_5.md",
                        "sha256": "c97db8f54374ac4ff1b5fbfafc1a1e76c63d68077144b338319603170ce846c2",
                        "item": "C"},
        "design": ("constrained refits: cap = n fixes the branching ratio at n (the likelihood is "
                   "monotone in n and pins at the cap), with every nuisance parameter re-optimized"),
        "no_interior_mle": ("the fit pins at whatever cap it is given (0.95 -> 0.95; 0.999 -> "
                            "0.9990, the documented degenerate near-critical mode), so there is no "
                            "interior MLE and only a one-sided statement could ever be meaningful"),
        "profile": rows,
        "all_converged": all_conv,
        "all_pinned_at_cap": all_pinned,
        "profile_monotone_in_n": monotone,
        "criterion_one_sided_95": CRIT_ONE_SIDED_95,
        "crossed_between": between,
        "NAMING": naming,
        "naming_reason": reason,
        "interval": interval,
    }
    json.dump(out, open(R4 / "r12_item_C.json", "w"), indent=2)

    print(f"{'n':>7} {'loglik':>14} {'2[l(.95)-l(n)]':>15} {'2[l(.999)-l(n)]':>16} "
          f"{'pinned':>7} {'conv':>5}  {'k':>7} {'alpha':>7} {'p':>7}")
    print("-" * 96)
    for r in rows:
        print(f"{r['n']:>7.3f} {r['loglik']:>14.3f} {r['LR_vs_0.95']:>15.3f} "
              f"{r['LR_vs_0.999']:>16.3f} {str(r['pinned_at_cap']):>7} {str(r['converged']):>5}  "
              f"{r['k']:>7.4f} {r['alpha']:>7.4f} {r['p']:>7.4f}")
    print(f"\nall converged: {all_conv} | all pinned at cap: {all_pinned} | "
          f"profile monotone: {monotone}")
    print(f"one-sided 95% criterion (50:50 boundary mixture): {CRIT_ONE_SIDED_95}")
    print(f"crossed between: {between}")
    print(f"\nNAMING (per the amendment's rule, fixed before the answers): {naming}")
    print(f"  reason: {reason}")
    print(f"  interval: {interval}")


if __name__ == "__main__":
    main()
