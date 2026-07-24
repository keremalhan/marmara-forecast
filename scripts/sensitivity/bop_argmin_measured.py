"""Run 18 — the corrected-simulator argmin, MEASURED on the registered candidate grid.

WHY THIS EXISTS. Amendment 6 states, and Amendment 7 repeats as established, that re-running the
registered procedure at the corrected simulator "yields 1.05 on the [::3] subsample and 1.00 on all
231 pre-test windows". **No artifact in the repository supports either number.** The only sweep ever
archived (r13) used the truncated candidate list {1.10, 1.12, 1.15, 1.18}; its argmin of 1.10 is
pinned at the list's own lower boundary, because every slope in it exceeds 1 and rises with b. So
1.05/1.00 were, at best, extrapolated off r13's two lowest points and never measured on the grid.

A number that reaches the manuscript must come from a run. This is that run.

WHAT. calibrate_b.py's procedure, unchanged, at its own registered candidate list
B_CANDIDATES = [0.9, 1.0, 1.05, 1.1, 1.12, 1.15, 1.18, 1.2, 1.542], on BOTH window sets:
the shipped [::3] subsample (77 windows) and all 231 pre-test windows. Same params, same K=300,
same per-window seed 5000+i, same origin-regression slope estimator, corrected simulator
(preserve_branching defaults True, cascade.py:236).

Two argmins are reported per window set, because the code and the manuscript disagree about the rule:
  * "plain"   — argmin |slope - 1| over all candidates          (what calibrate_b.py:55 does)
  * "banded"  — argmin |slope - 1| among candidates with slope in [0.8, 1.2]  (what §3 describes)
Both are reported unconditionally. No gate, no preferred outcome.

Writes results/round4/r18_bop_argmin_measured.json. Touches no frozen artifact.

Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/sensitivity/bop_argmin_measured.py
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd

from marmara import grid as G
from marmara.calibrate_b import B_CANDIDATES
from marmara.cascade import cascade_forecast
from marmara.paths import RESULTS

K_CAL = 300
R4 = RESULTS / "round4"


def sweep(params, cat, hist, e35, spec, windows, label, t_start):
    out = {}
    for b in B_CANDIDATES:
        pred, real = [], []
        for i, t0 in enumerate(windows):
            t0d = float(G._to_days(t0))
            c = cascade_forecast(params, hist[cat["datetime_utc"] < t0], t0d, 30.0,
                                 spec.lon_c, spec.lat_c, K=K_CAL, seed=5000 + i, b=b,
                                 preserve_branching=True)
            lo = np.searchsorted(e35["t"], t0d, "left")
            hi = np.searchsorted(e35["t"], t0d + 30.0, "left")
            pred.append(float(c["lam35"].sum()))
            real.append(float(hi - lo))
        pred, real = np.array(pred), np.array(real)
        slope = float((pred * real).sum() / max((pred * pred).sum(), 1e-12))
        out[b] = {"slope": round(slope, 4), "abs_slope_minus_1": round(abs(slope - 1.0), 4),
                  "pred_total": round(float(pred.sum()), 1),
                  "real_total": round(float(real.sum()), 1),
                  "in_band_0.8_1.2": bool(0.8 <= slope <= 1.2)}
        print(f"  [{label}] b={b:.3f}: slope {slope:.4f}  |s-1| {out[b]['abs_slope_minus_1']:.4f}"
              f"  pred {pred.sum():.0f} real {real.sum():.0f}  ({time.time()-t_start:.0f}s)",
              flush=True)
    plain = min(B_CANDIDATES, key=lambda b: out[b]["abs_slope_minus_1"])
    in_band = [b for b in B_CANDIDATES if out[b]["in_band_0.8_1.2"]]
    banded = min(in_band, key=lambda b: out[b]["abs_slope_minus_1"]) if in_band else None
    return {"per_b": {str(b): out[b] for b in B_CANDIDATES},
            "argmin_plain": plain, "argmin_plain_slope": out[plain]["slope"],
            "argmin_banded": banded,
            "argmin_banded_slope": (out[banded]["slope"] if banded else None),
            "candidates_in_band": [str(b) for b in in_band],
            "argmin_at_list_boundary": bool(plain == min(B_CANDIDATES)),
            "n_windows": len(windows)}


def main():
    t0 = time.time()
    params = pickle.load(open(RESULTS / "etas" / "etas_params.pkl", "rb"))
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    spec = G.MODEL_SPEC
    e35 = G.build_event_bundle(cat, 3.0)["e35"]
    hist = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]
    starts = G.window_starts(cat["datetime_utc"].max())
    pre = [t for t in starts if t + pd.Timedelta(days=30) <= pd.Timestamp("2024-01-01")]
    sub = pre[::3]

    print(f"registered candidate grid : {B_CANDIDATES}")
    print(f"windows                   : {len(sub)} ([::3] subsample) / {len(pre)} (all pre-test)\n",
          flush=True)

    out = {
        "why": ("Amendments 6 and 7 assert argmin 1.05 ([::3]) and 1.00 (all windows) at the "
                "corrected simulator. No archived artifact supported either: r13 swept only "
                "{1.10,1.12,1.15,1.18} and its argmin sat on that list's lower boundary. "
                "This measures the argmin on the registered candidate grid."),
        "procedure": "calibrate_b.py unchanged: K=300, seed 5000+i, origin regression, pb=True",
        "candidates": B_CANDIDATES,
        "claim_under_test": {"subsample_[::3]": 1.05, "all_windows": 1.00},
    }
    out["subsample_[::3]"] = sweep(params, cat, hist, e35, spec, sub, "[::3]", t0)
    print(flush=True)
    out["all_windows"] = sweep(params, cat, hist, e35, spec, pre, "ALL", t0)

    s, a = out["subsample_[::3]"], out["all_windows"]
    out["shipped_b_op"] = 1.15
    out["argmin_moves_with_window_set"] = bool(s["argmin_plain"] != a["argmin_plain"])
    out["amendment_claim_reproduced"] = {
        "subsample_1.05": bool(s["argmin_plain"] == 1.05),
        "all_windows_1.00": bool(a["argmin_plain"] == 1.00),
    }
    out["runtime_s"] = round(time.time() - t0, 1)
    json.dump(out, open(R4 / "r18_bop_argmin_measured.json", "w"), indent=2)

    print("\n=== MEASURED ARGMIN ON THE REGISTERED CANDIDATE GRID ===")
    print(f"{'b':>7} {'slope [::3]':>12} {'slope ALL':>11} {'|s-1| [::3]':>12} {'|s-1| ALL':>11}")
    for b in B_CANDIDATES:
        sb, ab = s["per_b"][str(b)], a["per_b"][str(b)]
        print(f"{b:>7.3f} {sb['slope']:>12.4f} {ab['slope']:>11.4f} "
              f"{sb['abs_slope_minus_1']:>12.4f} {ab['abs_slope_minus_1']:>11.4f}")
    print(f"\n  [::3] subsample : argmin_plain {s['argmin_plain']} (slope {s['argmin_plain_slope']})"
          f" | banded {s['argmin_banded']} | at list boundary: {s['argmin_at_list_boundary']}")
    print(f"  ALL windows     : argmin_plain {a['argmin_plain']} (slope {a['argmin_plain_slope']})"
          f" | banded {a['argmin_banded']} | at list boundary: {a['argmin_at_list_boundary']}")
    print(f"\n  argmin moves with window set : {out['argmin_moves_with_window_set']}")
    print(f"  amendment claim reproduced   : {out['amendment_claim_reproduced']}")
    print(f"  ({out['runtime_s']}s) -> results/round4/r18_bop_argmin_measured.json")


if __name__ == "__main__":
    main()
