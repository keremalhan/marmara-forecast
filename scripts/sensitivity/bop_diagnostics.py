"""Run 14 — the two b_op diagnostics of Amendment 6 (SHA-256 d6dc9516...), hashed before this ran.

D1 — era-split sweep: corrected simulator, full windows, restricted to MODERN-era pre-test windows
     (t0 >= 2013-01-01, per section 2's own completeness narrative). Prediction on record: argmin
     returns to ~1.12-1.15. If it does, the full-window argmin of 1.00 is contamination by the
     Md-conversion era (whose effective 3.0->3.5 slope is 0.840 against the modern era's 1.358).

D2 — threshold-split sweep: the M>=4.0-count calibration that section 6 already names in print as
     the repair, on full pre-test windows. The ~3.45 Md pile sits far below 4.0 and within-cell
     multiplicity at M>=4.0 is ~1. Prediction on record: ~1.3. Anything >> 1.00 proves
     threshold-dependence.

Both use the shipped procedure otherwise: same params, K=300, per-window seed 5000+i, slope by
regression of realized on predicted through the origin, argmin |slope - 1| over the candidate list.
Seeds {5000, 6000, 7000} are reported to quantify seed variability -- never to select.

Reported unconditionally. Writes results/round4/r14_bop_diagnostics.json.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/sensitivity/bop_diagnostics.py
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd

from marmara import cascade as CASC
from marmara import grid as G
from marmara.cascade import cascade_forecast
from marmara.paths import RESULTS

# D2 calibrates on M>=4.0 counts -- the repair section 6 names in print. The shipped cascade emits
# X_LEVELS = (3.5, 4.5, 5.0, 5.5, 6.0), so 4.0 is added here to obtain lam4.0. M>=4.5 would be the
# only alternative already emitted, and at 156 pre-2024 events it is badly underpowered for a slope.
CASC.X_LEVELS = (3.5, 4.0, 4.5, 5.0, 5.5, 6.0)

CANDS = [0.9, 1.0, 1.05, 1.1, 1.12, 1.15, 1.18, 1.2, 1.542]
SEEDS = [5000, 6000, 7000]
K_CAL = 300
B_AKI = 1.0185696913880833
R4 = RESULTS / "round4"


def main():
    t0 = time.time()
    params = pickle.load(open(RESULTS / "etas" / "etas_params.pkl", "rb"))
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    spec = G.MODEL_SPEC
    EV = G.build_event_bundle(cat, 3.0)
    hist = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]
    starts = G.window_starts(cat["datetime_utc"].max())
    pre = [t for t in starts if t + pd.Timedelta(days=30) <= pd.Timestamp("2024-01-01")]
    modern = [t for t in pre if t >= pd.Timestamp("2013-01-01")]

    def sweep(windows, ev_times, thr_col, label):
        """ev_times: sorted event times (days from REF) at the calibration threshold;
        thr_col: the cascade's lam key for that threshold."""
        real = np.array([float(np.searchsorted(ev_times, float(G._to_days(t)) + 30.0, "left")
                               - np.searchsorted(ev_times, float(G._to_days(t)), "left"))
                         for t in windows])
        rows = {}
        for b in CANDS:
            ss = []
            for base in SEEDS:
                pred = np.array([float(cascade_forecast(
                    params, hist[cat["datetime_utc"] < t], float(G._to_days(t)), 30.0,
                    spec.lon_c, spec.lat_c, K=K_CAL, seed=base + i, b=b,
                    preserve_branching=True)[thr_col].sum()) for i, t in enumerate(windows)])
                ss.append(float((pred * real).sum() / max((pred * pred).sum(), 1e-12)))
            rows[b] = {"slopes": [round(x, 4) for x in ss],
                       "mean_slope": round(float(np.mean(ss)), 4),
                       "sd_slope": round(float(np.std(ss, ddof=1)), 4),
                       "abs_mean_minus_1": round(abs(float(np.mean(ss)) - 1.0), 4)}
            print(f"  [{label}] b={b:.3f}: slope {rows[b]['mean_slope']:.4f} "
                  f"(sd {rows[b]['sd_slope']:.4f})  |s-1| {rows[b]['abs_mean_minus_1']:.4f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
        argmin = min(CANDS, key=lambda b: rows[b]["abs_mean_minus_1"])
        return {"label": label, "n_windows": len(windows), "realized_total": float(real.sum()),
                "per_b": {str(b): rows[b] for b in CANDS}, "argmin": argmin,
                "argmin_slope": rows[argmin]["mean_slope"]}

    out = {"governed_by": {"amendment": "docs/preregistration/v2_analysis_amendment_6.md",
                           "sha256": "d6dc9516a1554f816e855afafc9adcc91a12be6ee03e66fcb355148bce713af6"},
           "seeds": SEEDS, "candidates": CANDS, "b_aki": B_AKI,
           "predictions_on_record": {"D1": "argmin returns to ~1.12-1.15", "D2": "~1.3"}}

    print(f"D1: era-split — modern-era pre-test windows (t0 >= 2013): "
          f"{len(modern)} of {len(pre)}", flush=True)
    out["D1_era_split_modern"] = sweep(modern, EV["e35"]["t"], "lam35", "D1 modern-era, M>=3.5")
    print(f"\nD2: threshold-split — M>=4.0 counts, all {len(pre)} pre-test windows "
          f"({int((cat['mag_w'] >= 4.0).sum())} such events in catalogue)", flush=True)
    t40 = np.sort(np.asarray((cat.loc[cat["mag_w"] >= 4.0, "datetime_utc"] - G.REF)
                             / pd.Timedelta(days=1), dtype=float))
    out["D2_threshold_split_m40"] = sweep(pre, t40, "lam4.0", "D2 M>=4.0")

    d1, d2 = out["D1_era_split_modern"]["argmin"], out["D2_threshold_split_m40"]["argmin"]
    rule_fires = d1 >= B_AKI
    out["decision"] = {
        "D1_argmin": d1, "D2_argmin": d2, "b_aki": B_AKI,
        "rule": ("if the modern-era argmin >= b_Aki it becomes the operational calibration; "
                 "else rebuild at the corrected full-window argmin (1.00)"),
        "D1_ge_b_aki": bool(rule_fires),
        "operational_b_of_record": (d1 if rule_fires else 1.00),
        "contamination_proven": bool(d1 >= 1.10),
        "threshold_dependence_proven": bool(d2 > 1.00 + 1e-9),
    }
    out["runtime_s"] = round(time.time() - t0, 1)
    json.dump(out, open(R4 / "r14_bop_diagnostics.json", "w"), indent=2)
    print(f"\n=== DECISION (rule fixed before the answers) ===")
    print(f"  D1 modern-era argmin : {d1}  (slope {out['D1_era_split_modern']['argmin_slope']})")
    print(f"  D2 threshold argmin  : {d2}  (slope {out['D2_threshold_split_m40']['argmin_slope']})")
    print(f"  b_Aki                : {B_AKI:.4f}")
    print(f"  D1 >= b_Aki?         : {rule_fires}")
    print(f"  -> operational b of record: {out['decision']['operational_b_of_record']}")
    print(f"  contamination proven : {out['decision']['contamination_proven']}")
    print(f"  ({out['runtime_s']}s)")


if __name__ == "__main__":
    main()
