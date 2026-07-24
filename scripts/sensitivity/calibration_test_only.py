"""Run 12, item B (Amendment 5, SHA-256 c97db8f5...): test-only calibration table.

QUESTION. How do the Table S4 calibration products read when computed only over data that never
informed b_op?

The shipped battery evaluates over val+test (2022-01-01 -> 2026-03). b_op was calibrated on
pre-test windows -- which includes the validation years -- so the val+test battery is partly
in-sample with respect to the operational magnitude law. Here every product is recomputed over the
26 test windows alone (t0 >= 2024-01-22), and reported ALONGSIDE the existing battery, not in
place of it.

Reproducing the val+test arm with identical code also settles a discrepancy: the shipped
`results/validation_final/validation_final.json` records y35 obs/exp = 1.019 (277/271.76,
slope 0.951), while Table S4 and section 4 print 1.07 (277/258.0, slope 1.026). One of the two is
stale; this run says which.

Reported unconditionally. No gate.

Writes results/round4/r12_item_B.json. Reads only.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/sensitivity/calibration_test_only.py
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from marmara import grid as G
from marmara.metrics import lambda_to_p
from marmara.paths import RESULTS
from marmara.train import split_masks
from marmara.validation_final import realized_col, reliability

EPS = 1e-9
R4 = RESULTS / "round4"
R4.mkdir(exist_ok=True)


def hybrid_product_P(grid, count_col, lam_col):
    d = pickle.load(open(RESULTS / "models" / f"{count_col}_hybrid.pkl", "rb"))
    X = grid[G.FEATURES].copy()
    X["ln_lam_sim"] = np.log(grid[lam_col].to_numpy() + EPS)
    lam_ml = np.clip(d["reg"].predict(X), EPS, None)
    lam_sim = np.clip(grid[lam_col].to_numpy(), EPS, None)
    lam_h = lam_sim ** (1 - d["w"]) * lam_ml ** d["w"]
    return d["iso"].transform(lambda_to_p(lam_h)), d["w"]


def arm(grid, cat, mask, label):
    products = []
    for ycol, ccol, lcol in (("y35", "count35", "lam35_sim"), ("y45", "count45", "lam45_sim")):
        P, w = hybrid_product_P(grid, ccol, lcol)
        products.append((f"{ycol} 30d (hybrid)", P[mask], grid[ycol].to_numpy()[mask], w))
    for lvl, col in ((5.0, "Psim5.0"), (5.5, "Psim5.5"), (6.0, "Psim6.0")):
        obs = realized_col(grid, cat, lvl, G.MODEL_SPEC)
        products.append((f"P(M>={lvl}) 30d (cascade)", grid[col].to_numpy()[mask], obs[mask], None))
    rows = {}
    for name, pred, obs, w in products:
        r = reliability(pred, obs)
        agg = r.get("aggregate", {})
        rows[name] = {
            "hybrid_w": w,
            "observed_pos": agg.get("observed_pos"),
            "expected_pos": agg.get("expected_pos"),
            "obs_over_exp": agg.get("obs_over_exp"),
            "n_cellwindows": r.get("n"),
            "shape_testable": r.get("testable"),
            "slope": r.get("slope") if r.get("testable") else None,
            "intercept": r.get("intercept") if r.get("testable") else None,
            "verdict": r.get("verdict") if r.get("testable") else "too rare to bin",
        }
    return {"arm": label, "products": rows}


def main():
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    m = split_masks(grid)
    valtest = m["val"] | m["test"]
    test = m["test"]
    t0 = pd.to_datetime(grid["t0"])
    out = {"governed_by": {"amendment": "docs/preregistration/v2_analysis_amendment_5.md",
                           "sha256": "c97db8f54374ac4ff1b5fbfafc1a1e76c63d68077144b338319603170ce846c2",
                           "item": "B"},
           "arms": {},
           "test_window_span": [str(t0[test].min().date()), str(t0[test].max().date())],
           "n_test_windows": int(pd.Series(grid["window"].to_numpy()[test]).nunique()),
           "rationale": ("b_op was calibrated on pre-test windows, which include the validation "
                         "years; the val+test battery is therefore partly in-sample with respect "
                         "to the operational magnitude law. The test-only arm is not.")}

    out["arms"]["val_plus_test_2022_2026_shipped_scope"] = arm(grid, cat, valtest, "val+test")
    out["arms"]["test_only_2024_onward"] = arm(grid, cat, test, "test only")

    # settle the Table S4 vs validation_final.json discrepancy
    vf = json.load(open(RESULTS / "validation_final" / "validation_final.json"))
    vf_y35 = next(p for p in vf["battery"]["products"] if p["product"] == "y35 30d (hybrid)")
    mine = out["arms"]["val_plus_test_2022_2026_shipped_scope"]["products"]["y35 30d (hybrid)"]
    out["table_s4_reconciliation"] = {
        "recomputed_val_plus_test": {"obs": mine["observed_pos"], "exp": mine["expected_pos"],
                                     "obs_over_exp": mine["obs_over_exp"], "slope": mine["slope"]},
        "stored_validation_final_json": {"obs": vf_y35["before"]["aggregate"]["observed_pos"],
                                         "exp": vf_y35["before"]["aggregate"]["expected_pos"],
                                         "obs_over_exp": vf_y35["before"]["aggregate"]["obs_over_exp"],
                                         "slope": vf_y35["before"]["slope"]},
        "printed_in_table_s4_and_section4": {"obs": 277, "exp": 258.0, "obs_over_exp": 1.07,
                                             "slope": 1.026},
        "recompute_matches_stored_json": bool(
            abs(mine["obs_over_exp"] - vf_y35["before"]["aggregate"]["obs_over_exp"]) < 0.005),
        "recompute_matches_printed": bool(abs(mine["obs_over_exp"] - 1.07) < 0.005),
    }
    json.dump(out, open(R4 / "r12_item_B.json", "w"), indent=2)

    print(f"test windows: {out['n_test_windows']}, span {out['test_window_span']}\n")
    for key, a in out["arms"].items():
        print(f"=== {a['arm']} ===")
        print(f"{'product':28s} {'obs':>5} {'exp':>8} {'obs/exp':>8} {'slope':>7}  verdict")
        for nm, r in a["products"].items():
            sl = f"{r['slope']:.3f}" if r["slope"] is not None else "—"
            print(f"{nm:28s} {r['observed_pos']:>5} {r['expected_pos']:>8.2f} "
                  f"{r['obs_over_exp']:>8.3f} {sl:>7}  {r['verdict']}")
        print()
    rec = out["table_s4_reconciliation"]
    print("=== Table S4 reconciliation (y35, val+test) ===")
    print(f"  recomputed here        : obs {rec['recomputed_val_plus_test']['obs']} "
          f"exp {rec['recomputed_val_plus_test']['exp']} "
          f"obs/exp {rec['recomputed_val_plus_test']['obs_over_exp']} "
          f"slope {rec['recomputed_val_plus_test']['slope']}")
    print(f"  stored validation_final: obs {rec['stored_validation_final_json']['obs']} "
          f"exp {rec['stored_validation_final_json']['exp']} "
          f"obs/exp {rec['stored_validation_final_json']['obs_over_exp']} "
          f"slope {rec['stored_validation_final_json']['slope']}")
    print(f"  printed in Table S4    : obs 277 exp 258.0 obs/exp 1.07 slope 1.026")
    print(f"  -> recompute matches stored JSON: {rec['recompute_matches_stored_json']}")
    print(f"  -> recompute matches printed    : {rec['recompute_matches_printed']}")


if __name__ == "__main__":
    main()
