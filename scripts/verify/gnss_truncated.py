"""gnss_traj truncated self-test at >=12 monthly cutoffs across 2024-2026
(bit-for-bit) + one ADVERSARIAL mid-month cutoff (t0 not aligned to a grid window)
to confirm the monthly-fit-caching granularity never uses epochs >= t0.
Writes results/verify/gnss_truncated.json.
"""
import json
import numpy as np
import pandas as pd
from marmara.paths import RESULTS
from marmara.sources.gnss_traj import GnssTrajSource

OUT = RESULTS; VER = OUT / "verify"; VER.mkdir(exist_ok=True)


def main():
    src = GnssTrajSource()
    ok, why = src.available()
    if not ok:
        json.dump({"skipped": why}, open(VER / "gnss_truncated.json", "w"), indent=2)
        print("SKIPPED:", why); return
    g = pd.read_parquet(OUT / "grid_hybrid.parquet",
                        columns=["window", "t0", "cell_lon", "cell_lat", "ir", "ic"])
    tw = g[pd.to_datetime(g["t0"]) >= pd.Timestamp("2024-01-01")]
    wins = sorted(tw["window"].unique())
    chosen = sorted({wins[i] for i in np.linspace(0, len(wins) - 1, 13).astype(int)})
    res = []
    for w in chosen:
        cw = g[g.window == w].copy(); t0 = pd.Timestamp(cw["t0"].iloc[0])
        a = src.add_columns(cw).to_numpy(); b = src.recompute_truncated(cw, t0).to_numpy()
        res.append({"window": int(w), "t0": str(t0.date()),
                    "bitforbit": bool(np.array_equal(np.nan_to_num(a), np.nan_to_num(b))),
                    "max_abs_dev": float(np.nanmax(np.abs(a - b)))})
        print(f"  w{w} {t0.date()} bitforbit={res[-1]['bitforbit']} dev={res[-1]['max_abs_dev']:.2e}")
    # adversarial mid-month cutoff (not a grid t0)
    cw = g[g.window == chosen[len(chosen) // 2]].copy()
    mid = pd.Timestamp("2024-06-15")
    cw = cw.assign(t0=np.datetime64(mid))
    a = src.add_columns(cw).to_numpy(); b = src.recompute_truncated(cw, mid).to_numpy()
    adv = {"t0": str(mid.date()), "bitforbit": bool(np.array_equal(np.nan_to_num(a), np.nan_to_num(b))),
           "max_abs_dev": float(np.nanmax(np.abs(a - b)))}
    print(f"  ADVERSARIAL mid-month {mid.date()} bitforbit={adv['bitforbit']} dev={adv['max_abs_dev']:.2e}")
    out = {"n_monthly_cutoffs": len(res),
           "all_bitforbit": bool(all(r["bitforbit"] for r in res) and adv["bitforbit"]),
           "cutoffs": res, "adversarial_midmonth": adv,
           "verdict": "PASS: recompute_truncated bit-for-bit at all cutoffs incl. mid-month"
           if (all(r["bitforbit"] for r in res) and adv["bitforbit"]) else "FAIL: leakage"}
    json.dump(out, open(VER / "gnss_truncated.json", "w"), indent=2)
    print("all_bitforbit:", out["all_bitforbit"])


if __name__ == "__main__":
    main()
