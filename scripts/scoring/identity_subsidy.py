"""Verify the s = 1/multiplicity identity and the per-model IG count-handicap ('subsidy').

metrics.information_gain scores Poisson LL on the BINARY indicator y in {0,1}:
  IG(A vs B) = (1/N_pos) * sum[ y*(lnLa - lnLb) - (La - Lb) ],  L = -ln(1-P).
For a global rescale L -> s*L, IG(s vs 1) = ln s - (s-1)*sum(L)/N_pos, maximized at s* = N_pos/sum(L).
Claim: s*_cascade = 1/mean-multiplicity, where multiplicity = total events / occupied cell-windows.
Also tabulate sum(L)/N_pos per model -- the count handicap the occurrence-IG imposes.
Writes results/round3/t2_identity_subsidy.json.
"""
import json
import numpy as np
import pandas as pd
from marmara.paths import RESULTS
from marmara.metrics import p_to_lambda

EPS = 1e-9
MODELS = ["hybrid", "cascade", "sv_etas", "firstgen_etas", "modern_etas", "smoothed", "poisson"]


def lam(p):
    return np.clip(p_to_lambda(np.clip(np.asarray(p, float), 0.0, 1.0)), EPS, None)


def main():
    df = pd.read_parquet(RESULTS / "grid" / "predictions_y30.parquet")
    te = df[df.split == "test"]; va = df[df.split == "val"]
    yv = va.y.to_numpy(); yt = te.y.to_numpy()
    npos_t = float(yt.sum()); npos_v = float(yv.sum())

    # mean multiplicity = total M>=3.0 events / occupied cell-windows, test period
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    t0s = pd.to_datetime(te.t0.unique()); lo, hi = t0s.min(), t0s.max() + pd.Timedelta(days=30)
    ev = cat[(cat.datetime_utc >= lo) & (cat.datetime_utc < hi) & (cat.mag_w >= 3.0 - 1e-9)]
    total_events = int(len(ev))
    mult = total_events / npos_t

    out = {"n_pos_test": int(npos_t), "total_events_test_period": total_events,
           "mean_multiplicity": round(mult, 3), "inv_multiplicity": round(1 / mult, 4),
           "scoring": "Poisson LL on binary indicator y in {0,1} (metrics.information_gain)",
           "models": {}}

    for m in MODELS:
        Lt = lam(te[m].to_numpy()); Lv = lam(va[m].to_numpy())
        sumL_t = float(Lt.sum()); sumL_v = float(Lv.sum())
        s_star_val = npos_v / sumL_v                       # optimal global rescale, val-fit
        # closed-form IG of s_star_val applied to TEST (matches t1_recalibration scalar)
        ig_closed = float(np.log(s_star_val) - (s_star_val - 1) * sumL_t / npos_t)
        out["models"][m] = {"sumL_test": round(sumL_t, 1),
                            "sumL_over_npos_test": round(sumL_t / npos_t, 3),   # the handicap
                            "s_star_val": round(s_star_val, 4),
                            "closed_form_IG_of_val_scalar_on_test": round(ig_closed, 4)}

    c = out["models"]["cascade"]
    out["identity_check_cascade"] = {
        "s_star ~ 1/multiplicity": [c["s_star_val"], round(1 / mult, 4)],
        "sumL ~ npos*multiplicity": [c["sumL_test"], round(npos_t * mult, 1)],
        "closed_form_IG": c["closed_form_IG_of_val_scalar_on_test"]}
    (RESULTS / "round3" / "t2_identity_subsidy.json").write_text(json.dumps(out, indent=2))
    print(f"mean multiplicity {mult:.3f} (= {total_events}/{int(npos_t)}), 1/mult = {1/mult:.4f}")
    print("\nmodel            sumL/N_pos (handicap)   s*_val    closed-form IG(scalar on test)")
    for m in MODELS:
        v = out["models"][m]
        print(f"  {m:14s}  {v['sumL_over_npos_test']:6.3f}                {v['s_star_val']:.4f}    {v['closed_form_IG_of_val_scalar_on_test']:+.4f}")
    print("\nIDENTITY (cascade):", json.dumps(out["identity_check_cascade"]))


if __name__ == "__main__":
    main()
