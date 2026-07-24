"""Run 4 — Hybrid CSEP construction audit.

QUESTION. Which occupancy construction generated the hybrid catalogues fed to pyCSEP
(results/csep/inputs/hybrid_catalogs.npz -> results/round3/hybrid_csep.json, N mean 1,041)?
The builder script is not in the repo, and the file predates the round-3 top-up work
(npz mtime 2026-07-13; the top-up tie-break was pre-registered in the Amendment-3 addendum on
2026-07-15), so the construction cannot be read off the source and must be established from
the artifact itself.

THE TWO CANDIDATES (marmara round-3, scripts/scoring/occupancy_debias_topup.py):
  * capped thinning : ratio = min(1, lam_hyb/lam_casc); cells where lam_hyb > lam_casc are
                      FROZEN at the cascade rate -> total = sum min(lam_hyb, lam_casc) < sum lam_hyb
  * top-up          : thinning plus an independent Poisson superposition at the excess rate
                      where lam_hyb > lam_casc -> total = sum lam_hyb exactly (rate-exact)

DECISIVE TEST. Compare the npz's per-cell mean count over the 500 catalogues against the
hybrid's expected per-cell count (sum over test windows of lam30_hybrid). The two constructions
agree everywhere EXCEPT the cells where lam_hyb > lam_casc; capped thinning under-counts exactly
there. We therefore test (a) the global total and (b) the excess cells specifically -- the total
alone is necessary but not sufficient.

Writes results/round4/r4_csep_construction_audit.json. Reads only (no CSEP env needed).
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/csep/construction_audit.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from marmara import grid as G
from marmara.metrics import p_to_lambda
from marmara.paths import RESULTS

EPS = 1e-9
OUT = RESULTS / "round4"
OUT.mkdir(exist_ok=True)
INP = RESULTS / "csep" / "inputs"


def lam(p):
    return np.clip(p_to_lambda(np.clip(np.asarray(p, float), 0.0, 1.0)), EPS, None)


def main():
    spec = G.MODEL_SPEC
    meta = json.load(open(INP / "region.json"))
    period = meta["test_period"]

    # ---- expected per-cell counts from the frozen predictions (test split, M>=3.0) ----
    df = pd.read_parquet(RESULTS / "grid" / "predictions_y30.parquet")
    te = df[df.split == "test"].reset_index(drop=True)
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet",
                           columns=["window", "ir", "ic", "t0", "lam30_sim"])
    key = ["window", "ir", "ic"]
    # predictions_*.parquet carries window/t0/y + model probabilities but not ir/ic; re-attach
    # by position: train.evaluate() wrote rows in grid order for the val|test mask.
    from marmara.train import split_masks
    m = split_masks(grid)
    sel = np.where(m["val"] | m["test"])[0]
    gsel = grid.iloc[sel].reset_index(drop=True)
    assert len(gsel) == len(df), f"row-count mismatch {len(gsel)} vs {len(df)}"
    assert (gsel["window"].to_numpy() == df["window"].to_numpy()).all(), "window order mismatch"
    gsel = gsel.assign(split=df["split"].to_numpy())
    gt = gsel[gsel.split == "test"].reset_index(drop=True)

    lam_h = lam(te["hybrid"].to_numpy())
    lam_c = lam(te["cascade"].to_numpy())
    lam_c_grid = np.clip(gt["lam30_sim"].to_numpy(), EPS, None)
    npos = float(te["y"].sum())

    # cross-check: the cascade prediction column must be 1-exp(-lam30_sim)
    dev_c = float(np.abs(lam_c - lam_c_grid).max())

    flat = gt["ir"].to_numpy() * spec.nlon + gt["ic"].to_numpy()
    exp_h = np.bincount(flat, weights=lam_h, minlength=spec.ncells)
    exp_c = np.bincount(flat, weights=lam_c, minlength=spec.ncells)
    exp_capped = np.bincount(flat, weights=np.minimum(lam_h, lam_c), minlength=spec.ncells)

    excess = lam_h > lam_c
    tot_h = float(lam_h.sum()); tot_c = float(lam_c.sum())
    tot_capped = float(np.minimum(lam_h, lam_c).sum())

    # ---- observed per-cell mean counts in the npz ----
    d = np.load(INP / "hybrid_catalogs.npz")
    n_sim = int(d["n_sim"])
    lon, lat = d["longitude"], d["latitude"]
    ir, ic = G.cell_index_spec(lon, lat, spec)
    ok = (ir >= 0) & (ir < spec.nlat) & (ic >= 0) & (ic < spec.nlon)
    obs_flat = (ir[ok] * spec.nlon + ic[ok])
    obs_cell = np.bincount(obs_flat, minlength=spec.ncells).astype(float) / n_sim
    npz_total = float(len(lon)) / n_sim

    dc = np.load(INP / "cascade_catalogs.npz")
    n_sim_c = int(dc["n_sim"])
    irc, icc = G.cell_index_spec(dc["longitude"], dc["latitude"], spec)
    okc = (irc >= 0) & (irc < spec.nlat) & (icc >= 0) & (icc < spec.nlon)
    obs_cell_c = np.bincount(irc[okc] * spec.nlon + icc[okc],
                             minlength=spec.ncells).astype(float) / n_sim_c
    npz_total_c = float(len(dc["longitude"])) / n_sim_c

    def rms(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))

    # residuals against each candidate, over ALL cells and over the EXCESS cells only
    ex_cells = np.unique(flat[excess])
    res = {
        "period": period, "n_sim": n_sim,
        "npz_mean_total_per_catalog": round(npz_total, 3),
        "npz_mean_total_cascade": round(npz_total_c, 3),
        "paper_reported_hybrid_N_mean": 1041.074,
        "paper_reported_cascade_N_mean": 1305.772,
        "expected_totals_from_frozen_predictions": {
            "sum_lam_hybrid_topup_rate_exact": round(tot_h, 3),
            "sum_min_lam_hybrid_lam_cascade_capped": round(tot_capped, 3),
            "sum_lam_cascade": round(tot_c, 3),
            "n_pos": int(npos),
            "h_hybrid_sum_over_npos": round(tot_h / npos, 4),
            "h_cascade_sum_over_npos": round(tot_c / npos, 4),
        },
        "cascade_column_matches_lam30_sim_max_dev": dev_c,
        "excess_cells_where_lam_hyb_gt_lam_casc": {
            "n_cellwindows": int(excess.sum()),
            "frac_of_cellwindows": round(float(excess.mean()), 6),
            "n_distinct_cells": int(len(ex_cells)),
            "excess_rate_mass": round(float((lam_h[excess] - lam_c[excess]).sum()), 3),
        },
        "total_discriminant": {
            "npz_minus_topup": round(npz_total - tot_h, 3),
            "npz_minus_capped": round(npz_total - tot_capped, 3),
            "closer_to": ("top-up (rate-exact)" if abs(npz_total - tot_h) < abs(npz_total - tot_capped)
                          else "capped thinning"),
        },
        "per_cell_fit": {
            "rms_npz_vs_topup_all_cells": round(rms(obs_cell, exp_h), 4),
            "rms_npz_vs_capped_all_cells": round(rms(obs_cell, exp_capped), 4),
            "rms_npz_vs_topup_excess_cells": round(rms(obs_cell[ex_cells], exp_h[ex_cells]), 4),
            "rms_npz_vs_capped_excess_cells": round(rms(obs_cell[ex_cells], exp_capped[ex_cells]), 4),
            "sum_npz_excess_cells": round(float(obs_cell[ex_cells].sum()), 3),
            "sum_topup_excess_cells": round(float(exp_h[ex_cells].sum()), 3),
            "sum_capped_excess_cells": round(float(exp_capped[ex_cells].sum()), 3),
        },
        "cascade_sanity": {
            "rms_npz_cascade_vs_expected": round(rms(obs_cell_c, exp_c), 4),
            "npz_cascade_total_minus_expected": round(npz_total_c - tot_c, 3),
        },
    }

    pc = res["per_cell_fit"]
    verdict_total = res["total_discriminant"]["closer_to"]
    verdict_cells = ("top-up (rate-exact)" if pc["rms_npz_vs_topup_excess_cells"]
                     < pc["rms_npz_vs_capped_excess_cells"] else "capped thinning")
    res["verdict"] = {
        "by_total": verdict_total,
        "by_excess_cells": verdict_cells,
        "agree": bool(verdict_total == verdict_cells),
        "construction_used": verdict_cells if verdict_total == verdict_cells else "AMBIGUOUS",
        "mean_total_reproduces_1041": bool(abs(npz_total - 1041.0) < 5.0),
    }
    json.dump(res, open(OUT / "r4_csep_construction_audit.json", "w"), indent=2)

    print(f"CSEP test period {period}, n_sim {n_sim}")
    print(f"npz hybrid mean total/cat : {npz_total:.3f}   (paper reports 1041.074)")
    print(f"npz cascade mean total/cat: {npz_total_c:.3f}  (paper reports 1305.772)")
    print()
    print(f"sum lam_hybrid  (top-up, rate-exact) : {tot_h:.3f}")
    print(f"sum min(lam_h, lam_c) (capped thin)  : {tot_capped:.3f}")
    print(f"sum lam_cascade                      : {tot_c:.3f}")
    print(f"  npz - topup  = {npz_total - tot_h:+.3f}")
    print(f"  npz - capped = {npz_total - tot_capped:+.3f}")
    print()
    e = res["excess_cells_where_lam_hyb_gt_lam_casc"]
    print(f"excess cell-windows (lam_hyb > lam_casc): {e['n_cellwindows']} "
          f"({100*e['frac_of_cellwindows']:.2f}%), {e['n_distinct_cells']} distinct cells, "
          f"excess rate mass {e['excess_rate_mass']:.3f}")
    print(f"  sum over excess cells: npz {pc['sum_npz_excess_cells']:.3f} | "
          f"topup {pc['sum_topup_excess_cells']:.3f} | capped {pc['sum_capped_excess_cells']:.3f}")
    print(f"  RMS excess cells: vs topup {pc['rms_npz_vs_topup_excess_cells']:.4f} | "
          f"vs capped {pc['rms_npz_vs_capped_excess_cells']:.4f}")
    print(f"  RMS all cells   : vs topup {pc['rms_npz_vs_topup_all_cells']:.4f} | "
          f"vs capped {pc['rms_npz_vs_capped_all_cells']:.4f}")
    print(f"\ncascade sanity: RMS {res['cascade_sanity']['rms_npz_cascade_vs_expected']:.4f}, "
          f"total diff {res['cascade_sanity']['npz_cascade_total_minus_expected']:+.3f}")
    print(f"\nVERDICT: {json.dumps(res['verdict'], indent=1)}")


if __name__ == "__main__":
    main()
