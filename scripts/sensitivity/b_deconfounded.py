"""Run 19 — the deconfounded magnitude law and the test-era split, MEASURED and archived.

WHY THIS EXISTS. Amendment 7 reports three deconfounded estimates (1.365 / 1.405 / 1.418 -> "1.37-1.42")
and a test-era split (1.280 pre / 1.108 post / 1.167 full) as *established results*. Its own provenance
ledger says they were computed before the amendment was written. **No script in this repository
computes any of them and no artifact records them.** r18 has just shown what that costs: the same
class of claim (argmin 1.05/1.00) was also unarchived, and had to be measured before it could be
printed. These numbers are now load-bearing in section 3, section 6 and the future-work motivation,
so they get a run and an artifact like everything else.

THE ARITHMETIC (Amendment 7 section 2). The branching-preserving rescale pins the simulated total at
n = 0.95 regardless of b, so the cascade's M>=3.0 slope is b-independent and carries the rate error
alone. Dividing the >=M slope by it cancels the rate error and, algebraically, the cascade:

    slope_M(b) / slope_3.0 = (real_M / real_3.0) * 10^{b (M - 3.0)}

which crosses one at   b = log10(real_3.0 / real_M) / (M - 3.0).

So the deconfounded calibration is just the catalogue's own effective Gutenberg-Richter slope between
the two thresholds. That is the claim under test here, and it is a pure counting exercise -- which is
precisely why there was never an excuse for it to be unarchived.

Counting note: window_starts uses step = horizon = 30 d (grid.py STEP_D/HORIZON_D), so the windows
are contiguous non-overlapping tiles and the 231 pre-test windows tile 2005-01-01 -> 2023-12-23
exactly once, with no double counting. (An earlier draft of this note claimed a 10-day step and
threefold overlap; that was wrong. It never touched the arithmetic -- the ratio n_3.0/n_M is
invariant to any uniform multiplicity -- but a wrong explanation in a script is a future defect.)

Reported unconditionally; every claimed value is compared against what is measured, and a mismatch
is printed as a mismatch. Writes results/round4/r19_deconfounded_b.json. Touches no frozen artifact.

Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/sensitivity/b_deconfounded.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from marmara import grid as G
from marmara.paths import RESULTS

R4 = RESULTS / "round4"
TEST_START = pd.Timestamp("2024-01-01")
MODERN_START = pd.Timestamp("2013-01-01")
KUMBURGAZ = pd.Timestamp("2025-04-23")


def b_eff(n_lo, n_hi, m_lo, m_hi):
    """Effective GR b between two thresholds from two counts."""
    if n_hi <= 0 or n_lo <= 0:
        return None
    return float(np.log10(n_lo / n_hi) / (m_hi - m_lo))


def window_counts(cat, windows, thresholds):
    """Sum of events >= each threshold over the union of 30-day windows (with multiplicity)."""
    out = {}
    for m in thresholds:
        t = np.sort(np.asarray((cat.loc[cat["mag_w"] >= m, "datetime_utc"] - G.REF)
                               / pd.Timedelta(days=1), dtype=float))
        tot = 0
        for w in windows:
            d = float(G._to_days(w))
            tot += int(np.searchsorted(t, d + 30.0, "left") - np.searchsorted(t, d, "left"))
        out[m] = tot
    return out


def main():
    t0 = time.time()
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    starts = G.window_starts(cat["datetime_utc"].max())
    pre = [t for t in starts if t + pd.Timedelta(days=30) <= TEST_START]
    modern = [t for t in pre if t >= MODERN_START]

    out = {
        "why": ("Amendment 7 reports the deconfounded estimates and the test-era split as established "
                "results, but no script or artifact in the repo produced them. This measures them."),
        "formula": "b = log10(real_3.0 / real_M) / (M - 3.0)  [Amendment 7 section 2]",
        "claims_under_test": {"modern_3.0_to_3.5": 1.365, "all_pretest_3.0_to_4.0": 1.405,
                              "catalogue_3.5_to_4.5": 1.418,
                              "test_era_pre": 1.280, "test_era_post": 1.108, "test_era_full": 1.167},
    }

    # --- the three deconfounded estimates -------------------------------------------------
    est = {}
    wm = window_counts(cat, modern, [3.0, 3.5])
    est["modern_3.0_to_3.5"] = {"n_windows": len(modern), "counts": wm,
                                "b": b_eff(wm[3.0], wm[3.5], 3.0, 3.5)}
    wa = window_counts(cat, pre, [3.0, 4.0])
    est["all_pretest_3.0_to_4.0"] = {"n_windows": len(pre), "counts": wa,
                                     "b": b_eff(wa[3.0], wa[4.0], 3.0, 4.0)}
    n35 = int((cat["mag_w"] >= 3.5).sum()); n45 = int((cat["mag_w"] >= 4.5).sum())
    est["catalogue_3.5_to_4.5"] = {"scope": "whole catalogue, raw",
                                   "counts": {3.5: n35, 4.5: n45},
                                   "b": b_eff(n35, n45, 3.5, 4.5)}
    out["deconfounded_estimates"] = est
    vals = [v["b"] for v in est.values() if v["b"]]
    out["deconfounded_range"] = [round(min(vals), 3), round(max(vals), 3)]

    # --- the test-era split ---------------------------------------------------------------
    te = cat[cat["datetime_utc"] >= TEST_START]
    split = {}
    for lab, sel in (("pre_mainshock", te[te["datetime_utc"] < KUMBURGAZ]),
                     ("post_mainshock", te[te["datetime_utc"] >= KUMBURGAZ]),
                     ("full_test_era", te)):
        a = int((sel["mag_w"] >= 3.0).sum()); c = int((sel["mag_w"] >= 3.5).sum())
        split[lab] = {"n3.0": a, "n3.5": c, "b_eff_3.0_to_3.5": b_eff(a, c, 3.0, 3.5)}
    out["test_era_split"] = split

    # --- era-dependent effective slopes (section 2's mixture narrative) -------------------
    eras = {}
    for lab, lo, hi in (("md_conversion_era_2003_2012", pd.Timestamp("2003-01-01"), MODERN_START),
                        ("modern_ml_era_2013_2021", MODERN_START, pd.Timestamp("2022-01-01")),
                        ("test_era_2024_on", TEST_START, cat["datetime_utc"].max())):
        sel = cat[(cat["datetime_utc"] >= lo) & (cat["datetime_utc"] < hi)]
        a = int((sel["mag_w"] >= 3.0).sum()); c = int((sel["mag_w"] >= 3.5).sum())
        yrs = (hi - lo) / pd.Timedelta(days=365.25)
        eras[lab] = {"n3.0": a, "n3.5": c, "b_eff_3.0_to_3.5": b_eff(a, c, 3.0, 3.5),
                     "rate35_per_yr": round(c / yrs, 1) if yrs > 0 else None}
    out["era_effective_slopes"] = eras

    # --- verdict: does each claimed value reproduce? --------------------------------------
    meas = {"modern_3.0_to_3.5": est["modern_3.0_to_3.5"]["b"],
            "all_pretest_3.0_to_4.0": est["all_pretest_3.0_to_4.0"]["b"],
            "catalogue_3.5_to_4.5": est["catalogue_3.5_to_4.5"]["b"],
            "test_era_pre": split["pre_mainshock"]["b_eff_3.0_to_3.5"],
            "test_era_post": split["post_mainshock"]["b_eff_3.0_to_3.5"],
            "test_era_full": split["full_test_era"]["b_eff_3.0_to_3.5"]}
    check = {}
    for k, claimed in out["claims_under_test"].items():
        m = meas[k]
        check[k] = {"claimed": claimed, "measured": round(m, 4) if m else None,
                    "abs_diff": round(abs(m - claimed), 4) if m else None,
                    "reproduces_to_0.01": bool(m and abs(m - claimed) < 0.01)}
    out["claim_check"] = check
    out["all_claims_reproduce"] = all(v["reproduces_to_0.01"] for v in check.values())
    out["runtime_s"] = round(time.time() - t0, 1)
    json.dump(out, open(R4 / "r19_deconfounded_b.json", "w"), indent=2, default=str)

    print("=== DECONFOUNDED MAGNITUDE LAW (b = log10(N_lo/N_hi)/dM) ===")
    for k, v in est.items():
        print(f"  {k:26s} counts {v['counts']}  ->  b = {v['b']:.4f}")
    print(f"  range: {out['deconfounded_range']}")
    print("\n=== TEST-ERA SPLIT (effective 3.0->3.5 slope) ===")
    for k, v in split.items():
        print(f"  {k:16s} n3.0={v['n3.0']:5d} n3.5={v['n3.5']:4d}  ->  b = {v['b_eff_3.0_to_3.5']:.4f}")
    print("\n=== ERA EFFECTIVE SLOPES ===")
    for k, v in eras.items():
        print(f"  {k:28s} b = {v['b_eff_3.0_to_3.5']:.4f}   M>=3.5 rate {v['rate35_per_yr']}/yr")
    print("\n=== CLAIM CHECK (Amendment 7's printed values vs measured) ===")
    for k, v in check.items():
        flag = "OK " if v["reproduces_to_0.01"] else "MISMATCH"
        print(f"  {flag} {k:26s} claimed {v['claimed']:.3f}  measured {v['measured']}  "
              f"diff {v['abs_diff']}")
    print(f"\nall claims reproduce to 0.01: {out['all_claims_reproduce']}  ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
