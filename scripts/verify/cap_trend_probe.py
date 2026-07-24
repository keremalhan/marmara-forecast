"""Item A (governed program 2026-07-12): cap-trend probe -- does the 0.95 branching
cap MANUFACTURE the spatial misallocation, or is the misallocation a genuine temporal
non-stationarity?

REFIT ETAS with the branching-ratio cap at n in {0.95, 0.98, 0.99} (not 0.999 --
near-critical simulation is numerically meaningless). Each refit RE-ESTIMATES the
background mu vs triggering partition (this is why the earlier EM probe cannot
substitute: EM inherits the capped kernel's partition). For each fit, regenerate the
surgical cascade over the test windows and recompute the 4-way splits
(pre/post-Kumburgaz; aftershock zone <60 km vs background). Plot the four ratios vs n.

Decision rule (fixed in advance):
  * ratios move MATERIALLY toward 1 as n rises  -> the 0.95 cap manufactured the
    misallocation (aftershocks under-booked, smeared into an inflated background);
    the temporal-non-stationarity claim is DEAD and the mis-partition is the finding.
  * ratios FLAT in n                            -> non-stationarity confirmed, the
    mis-partition alternative excluded.

Writes results/cap_trend_probe.{json,md,png}.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/verify/cap_trend_probe.py
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd

from marmara import grid as G
from marmara.paths import RESULTS
from marmara.train import split_masks
from marmara.cascade import cascade_forecast
from marmara.etas_model import branching_ratio
from marmara.etas_fit import fit_stai, MODEL_BOX, FIT_END

CAPS = [0.95, 0.98, 0.99]
B_OP = 1.15
K = 500
KUMBURGAZ = (28.23, 40.84)
SPLIT_DATE = pd.Timestamp("2025-04-23")
R_ZONE_KM = 60.0
KM_PER_DEG = 111.19


def main():
    t0 = time.time()
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    test_wins = [int(w) for w in np.sort(grid.loc[split_masks(grid)["test"], "window"].unique())]
    win_t0 = grid.groupby("window")["t0"].first()
    spec = G.MODEL_SPEC
    hist_all = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]
    windows = [(w, pd.Timestamp(win_t0.loc[w]), float(G._to_days(pd.Timestamp(win_t0.loc[w]))),
                hist_all[cat["datetime_utc"] < pd.Timestamp(win_t0.loc[w])]) for w in test_wins]

    # observed splits (config-independent)
    t0s = pd.to_datetime([win_t0.loc[w] for w in test_wins])
    lo, hi = t0s.min(), t0s.max() + pd.Timedelta(days=G.HORIZON_D)
    ev = cat[(cat["datetime_utc"] >= lo) & (cat["datetime_utc"] < hi) & (cat["mag_w"] >= 3.0 - 1e-9)]
    iir, iic = G.cell_index(ev["longitude"].to_numpy(), ev["latitude"].to_numpy()); on = iir >= 0
    ev, iir, iic = ev[on], iir[on], iic[on]
    O_cell = np.zeros((G.NLAT, G.NLON)); np.add.at(O_cell, (iir, iic), 1.0)
    O_pre = int((ev["datetime_utc"] < SPLIT_DATE).sum()); O_post = int((ev["datetime_utc"] >= SPLIT_DATE).sum())
    LO, LA = np.meshgrid(G.LON_C, G.LAT_C)
    dx = (LO - KUMBURGAZ[0]) * KM_PER_DEG * np.cos(np.radians(KUMBURGAZ[1])); dy = (LA - KUMBURGAZ[1]) * KM_PER_DEG
    zone = np.sqrt(dx * dx + dy * dy) < R_ZONE_KM
    O_zone = int(O_cell[zone].sum()); O_bg = int(O_cell[~zone].sum())

    cat_fit = cat[cat["datetime_utc"] < FIT_END].copy()
    rows = []
    for cap in CAPS:
        if abs(cap - 0.95) < 1e-9:
            params = pickle.load(open(RESULTS / "etas" / "etas_params.pkl", "rb"))     # canonical n=0.95
            src = "canonical etas_params.pkl"
        else:
            params, *_ = fit_stai(cat_fit, MODEL_BOX, cap)                    # refit re-estimates mu
            src = f"refit cap={cap}"
        n_un, n_tr, mu = branching_ratio(params), branching_ratio(params, mmax=7.6), params.mu_total
        F_cell = np.zeros((G.NLAT, G.NLON)); F_pre = F_post = F30 = 0.0
        for (w, t0_dt, t0d, h) in windows:
            casc = cascade_forecast(params, h, t0d, G.HORIZON_D, spec.lon_c, spec.lat_c,
                                    K=K, seed=1000 + w, b=B_OP, preserve_branching=True)
            F_cell += casc["lam30"]; s = float(casc["lam30"].sum()); F30 += s
            if t0_dt < SPLIT_DATE:
                F_pre += s
            else:
                F_post += s
        ratios = {"pre": F_pre / O_pre, "post": F_post / O_post,
                  "zone": float(F_cell[zone].sum()) / O_zone, "bg": float(F_cell[~zone].sum()) / O_bg}
        rows.append({"cap": cap, "src": src, "n_untrunc": round(n_un, 4), "n_mmax7.6": round(n_tr, 4),
                     "mu_total": round(float(mu), 4), "F_total_M3.0": round(F30, 1),
                     "ratios": {k: round(v, 3) for k, v in ratios.items()}})
        print(f"  cap={cap} n_un={n_un:.4f} mu={mu:.4f} F30={F30:.0f} "
              f"pre={ratios['pre']:.2f} post={ratios['post']:.2f} zone={ratios['zone']:.2f} "
              f"bg={ratios['bg']:.2f} ({time.time()-t0:.0f}s)")

    # verdict: mean |ratio-1| across the 4 splits, at lowest vs highest cap
    def mad(r): return float(np.mean([abs(r["ratios"][k] - 1.0) for k in ("pre", "post", "zone", "bg")]))
    mad_lo, mad_hi = mad(rows[0]), mad(rows[-1])
    frac_closed = (mad_lo - mad_hi) / mad_lo if mad_lo else 0.0
    verdict = ("cap_manufactured_misallocation (temporal non-stationarity DEAD; mis-partition is the finding)"
               if frac_closed >= 0.25 else
               "non-stationarity CONFIRMED (mis-partition alternative excluded)"
               if frac_closed <= 0.10 else "AMBIGUOUS (partial closure)")
    result = {"meta": {"caps": CAPS, "b_op": B_OP, "K": K, "runtime_s": round(time.time() - t0, 1),
                       "observed": {"pre": O_pre, "post": O_post, "zone": O_zone, "background": O_bg}},
              "fits": rows,
              "mean_abs_ratio_minus_1": {"at_0.95": round(mad_lo, 3), "at_0.99": round(mad_hi, 3),
                                         "fraction_closed": round(frac_closed, 3)},
              "verdict": verdict}
    (RESULTS / "verify" / "cap_trend_probe.json").write_text(json.dumps(result, indent=2))
    _write_md(result); _plot(result)
    print("\nVERDICT:", verdict)
    print("wrote results/cap_trend_probe.{json,md,png}")


def _write_md(r):
    L = ["# Item A: cap-trend probe -- did the 0.95 cap manufacture the misallocation?", "",
         f"Observed splits: pre-Kumburgaz {r['meta']['observed']['pre']}, post {r['meta']['observed']['post']}, "
         f"aftershock-zone {r['meta']['observed']['zone']}, background {r['meta']['observed']['background']}.", "",
         "| cap (n) | n_untrunc | mu_total | F total | pre f/o | post f/o | zone f/o | bg f/o |",
         "|---|---|---|---|---|---|---|---|"]
    for x in r["fits"]:
        rt = x["ratios"]
        L.append(f"| {x['cap']} | {x['n_untrunc']} | {x['mu_total']} | {x['F_total_M3.0']} | "
                 f"{rt['pre']} | {rt['post']} | {rt['zone']} | {rt['bg']} |")
    m = r["mean_abs_ratio_minus_1"]
    L += ["", f"Mean |ratio-1| across the 4 splits: {m['at_0.95']} (n=0.95) -> {m['at_0.99']} (n=0.99); "
          f"fraction closed = {m['fraction_closed']}.", "",
          f"**VERDICT: {r['verdict']}**", "",
          "Decision rule (pre-fixed): fraction_closed >= 0.25 => cap manufactured the misallocation "
          "(temporal non-stationarity dead); <= 0.10 => non-stationarity confirmed; between => ambiguous."]
    (RESULTS / "verify" / "cap_trend_probe.md").write_text("\n".join(L))


def _plot(r):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        caps = [x["cap"] for x in r["fits"]]
        fig, ax = plt.subplots(figsize=(6, 4))
        for key, lab in (("pre", "pre-Kumburgaz (bg-dominated)"), ("post", "post-Kumburgaz"),
                         ("zone", "aftershock zone <60km"), ("bg", "background")):
            ax.plot(caps, [x["ratios"][key] for x in r["fits"]], "o-", label=lab)
        ax.axhline(1.0, color="k", lw=0.8, ls="--")
        ax.set_xlabel("branching-ratio cap n"); ax.set_ylabel("forecast / observed")
        ax.set_title("Item A: split ratios vs branching cap (toward 1 = cap-induced)")
        ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(RESULTS / "verify" / "cap_trend_probe.png", dpi=130)
        plt.close(fig)
    except Exception as e:
        print(f"plot skipped ({e})")


if __name__ == "__main__":
    main()
