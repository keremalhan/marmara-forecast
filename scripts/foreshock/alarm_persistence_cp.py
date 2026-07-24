"""Run 12, items D and E (Amendment 5, SHA-256 c97db8f5...).

D — Alarm-duration ladder over the CORRECT denominator.
   The shipped ladder computed >=40x persistence over the 621 triggers escalating at >=10x. Most of
   those never reach 40x at all, so they were recorded as "falling below 40x within the hour" when
   they were never above it. That artifact produced the "two-thirds decay within an hour" sentence.
   Here the >=40x ladder is recomputed over the 217 triggers that actually exceed 40x at t+10min.
   The 217 are a strict subset of the 621 (gain_post >= 40 > 10), so this is a re-tabulation of the
   already-computed ladder, not a new simulation.
   The 621-denominator FAR/precision rows are left untouched: the two denominators answer different
   questions and both stay in Table S10.

E — Clopper-Pearson 95% intervals for the precision 1/621 (cell) and 1/627 (25 km).

Writes results/round4/r12_items_DE.json. Reads only.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/foreshock/alarm_persistence_cp.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy.stats import beta

from marmara.paths import RESULTS

R4 = RESULTS / "round4"
LADDER = ["+1h", "+6h", "+1d", "+3d", "+7d", "+14d", "+30d"]


def clopper_pearson(k, n, alpha=0.05):
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return [lo, hi]


def main():
    d = json.load(open(R4 / "r7_foreshock_denominator.json"))
    df = pd.read_parquet(R4 / "r7_triggers.parquet")
    per = d["alarm_duration"]["per_trigger"]
    out = {"governed_by": {"amendment": "docs/preregistration/v2_analysis_amendment_5.md",
                           "sha256": "c97db8f54374ac4ff1b5fbfafc1a1e76c63d68077144b338319603170ce846c2",
                           "items": ["D", "E"]}}

    # ---------------- D ----------------
    esc10 = [p for p in per]                       # the 621 escalating at >=10x
    esc40 = [p for p in per if p["gain_post"] >= 40.0]   # the 217 that actually cross 40x
    n40 = int((df["gain_post"] >= 40.0).sum())
    assert len(esc40) == n40, f"subset mismatch {len(esc40)} vs {n40}"

    def tally(rows, key):
        t = {}
        for r in rows:
            k = r.get(key) or "<+1h"
            t[k] = t.get(k, 0) + 1
        return {k: t[k] for k in ["<+1h"] + LADDER if k in t}

    def survival(rows, key):
        """Fraction still at or above threshold at each rung."""
        n = len(rows)
        s = {}
        for i, rung in enumerate(LADDER):
            still = sum(1 for r in rows
                        if r.get(key) is not None and LADDER.index(r[key]) >= i)
            s[rung] = {"n": still, "frac": round(still / n, 4) if n else None}
        return s

    # observability: every rung <= +30d must be evaluable for every trigger (the 30-day
    # follow-on filter guarantees it). Verify rather than assert.
    missing = [(r["t"], k) for r in esc10 for k, v in r["ladder"].items() if v is None]
    out["D_alarm_duration"] = {
        "defect_corrected": ("the shipped >=40x ladder used the 621-trigger (>=10x) denominator; "
                             "triggers that never reached 40x were counted as decaying below it "
                             "within the hour"),
        "denominator_10x": {"n": len(esc10), "meaning": "triggers escalating >=10x at t+10min"},
        "denominator_40x": {"n": len(esc40), "meaning": "triggers exceeding 40x at t+10min"},
        "subset_check": "the 217 are a strict subset of the 621 (gain_post >= 40 > 10)",
        "observability": {
            "n_missing_ladder_points": len(missing),
            "all_rungs_observable": len(missing) == 0,
            "note": ("the pre-registered 30-day follow-on observability filter makes every rung "
                     "<= +30 d evaluable for every trigger; durations are interval-censored "
                     "between rungs"),
        },
        "ge40x_last_above_OLD_wrong_denominator_621": tally(esc10, "last_above_40x"),
        "ge40x_last_above_CORRECT_denominator_217": tally(esc40, "last_above_40x"),
        "ge40x_survival_CORRECT_denominator_217": survival(esc40, "last_above_40x"),
        "ge10x_last_above_denominator_621_unchanged": tally(esc10, "last_above_10x"),
        "ge10x_survival_denominator_621": survival(esc10, "last_above_10x"),
    }
    # the headline sentence, both ways
    t_old = tally(esc10, "last_above_40x")
    t_new = tally(esc40, "last_above_40x")
    below_1h_old = t_old.get("<+1h", 0)
    below_1h_new = t_new.get("<+1h", 0)
    out["D_alarm_duration"]["headline"] = {
        "old_sentence": "at >=40x, two-thirds decay within an hour",
        "old_arithmetic": f"{below_1h_old}/{len(esc10)} = {below_1h_old/len(esc10):.1%} "
                          f"(WRONG denominator: most never crossed 40x)",
        "corrected": f"{below_1h_new}/{len(esc40)} = {below_1h_new/len(esc40):.1%} of the triggers "
                     f"that DO cross 40x fall below it within the hour",
        "still_above_30d": f"{t_new.get('+30d', 0)}/{len(esc40)} = "
                           f"{t_new.get('+30d', 0)/len(esc40):.1%}",
    }

    # ---------------- E ----------------
    rows = {}
    for label, key in (("cell", "primary_gain_post_cell"), ("nbr25km", "gain_post_nbr25km")):
        a = d["accounting"][key]["ge_10x"]
        n, k = a["n_escalations"], a["n_hits"]
        ci = clopper_pearson(k, n)
        rows[label] = {"n_alarms": n, "n_hits": k,
                       "precision": round(k / n, 6),
                       "precision_ci95_clopper_pearson": [round(ci[0], 6), round(ci[1], 6)],
                       "false_alarm_rate": round(1 - k / n, 6),
                       "far_ci95": [round(1 - ci[1], 6), round(1 - ci[0], 6)]}
    for label, key in (("cell_ge40x", "primary_gain_post_cell"),):
        a = d["accounting"][key]["ge_40x"]
        n, k = a["n_escalations"], a["n_hits"]
        ci = clopper_pearson(k, n)
        rows[label] = {"n_alarms": n, "n_hits": k, "precision": round(k / n, 6),
                       "precision_ci95_clopper_pearson": [round(ci[0], 6), round(ci[1], 6)],
                       "false_alarm_rate": round(1 - k / n, 6),
                       "far_ci95": [round(1 - ci[1], 6), round(1 - ci[0], 6)]}
    out["E_precision_intervals"] = {
        "method": "Clopper-Pearson exact binomial, 95%",
        "rows": rows,
    }

    json.dump(out, open(R4 / "r12_items_DE.json", "w"), indent=2)

    D = out["D_alarm_duration"]
    print("=== D — alarm duration, correct denominator ===")
    print(f"  denominators: >=10x n={len(esc10)} | >=40x n={len(esc40)} (strict subset)")
    print(f"  all rungs observable: {D['observability']['all_rungs_observable']} "
          f"({D['observability']['n_missing_ladder_points']} missing)")
    print(f"  >=40x last-above, WRONG denom (621): {json.dumps(t_old)}")
    print(f"  >=40x last-above, CORRECT denom (217): {json.dumps(t_new)}")
    print(f"  headline OLD : {D['headline']['old_arithmetic']}")
    print(f"  headline NEW : {D['headline']['corrected']}")
    print(f"  still >=40x at +30d: {D['headline']['still_above_30d']}")
    print("\n  >=40x survival among the 217:")
    for rung, v in D["ge40x_survival_CORRECT_denominator_217"].items():
        print(f"    {rung:>5}: {v['n']:3d}/{len(esc40)} = {v['frac']:.1%}")
    print("\n  >=10x survival among the 621 (unchanged):")
    for rung, v in D["ge10x_survival_denominator_621"].items():
        print(f"    {rung:>5}: {v['n']:3d}/{len(esc10)} = {v['frac']:.1%}")
    print("\n=== E — Clopper-Pearson precision ===")
    for k, v in rows.items():
        print(f"  {k:12s} {v['n_hits']}/{v['n_alarms']}  precision {v['precision']:.5f} "
              f"CI {[round(x,5) for x in v['precision_ci95_clopper_pearson']]}  "
              f"FAR {v['false_alarm_rate']:.5f} CI {[round(x,5) for x in v['far_ci95']]}")
    print("\nwrote results/round4/r12_items_DE.json")


if __name__ == "__main__":
    main()
