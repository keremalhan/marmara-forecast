"""Where does the hybrid's information-gain over the cascade actually live?

The primary-target headline is that the ETAS x ML hybrid is *inseparable* from the
physics cluster under the conjunctive (IG AND PR-AUC) rule. But on the IG axis alone the
hybrid separably beats the cascade (+0.29 nats/event [+0.20,+0.43], CI excludes zero), so
"the ML adds no value" over-reads a conjunctive-rule tie as a negative result. This script
localizes that +0.29 so the manuscript can say what the ML measurably *does*, not just what
the rule cannot resolve. It answers two questions with the SAME scoring code as metrics.IG:

  (1) FOUR-WAY split of the total LL gain over the 2x2 = {pre,post Kumburgaz} x
      {aftershock zone <60 km, background}. Same masks as ntest_residual_probe.py.
  (2) TERM split of the total LL gain into its two additive pieces:
        placement P = sum_i  y_i (ln L_h,i - ln L_c,i)   (positives only; "ranking" mass)
        count     C = -sum_i (L_h,i - L_c,i)             (all cells; total-rate calibration)
      IG_total = P + C. If C dominates and sits in the over-predicted quiet background, the
      hybrid's gain is a rate re-calibration, not a re-ranking (consistent with PR-AUC not
      separating). If P dominates in the aftershock zone, it is patching the productivity
      hole. The data decide the wording.

Every quantity carries a paper-matched stationary block bootstrap CI (Politis-Romano over
ordered test windows, mean block 3, B=2000, seed 42) via marmara.bootstrap, so no share is
reported that the resample does not license.

Writes results/verify/mechanism_ig_split.json (+ .md). Reads only.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/verify/mechanism_ig_split.py
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from marmara.paths import RESULTS
from marmara.metrics import p_to_lambda
from marmara.bootstrap import stationary_window_indices, MEAN_BLOCK, SEED

EPS = 1e-9
KUMBURGAZ = (28.23, 40.84)          # lon, lat of the 23 Apr 2025 Mw6.2 (ntest_residual_probe)
SPLIT_DATE = pd.Timestamp("2025-04-23")
R_ZONE_KM = 60.0
KM_PER_DEG = 111.19
B = 2000
BASE = "cascade"                    # the physics comparator the hybrid separably beats on IG
MODEL = "hybrid"


def _lam(p):
    return np.clip(p_to_lambda(np.clip(np.asarray(p, float), 0.0, 1.0)), EPS, None)


def load_test(target: str) -> pd.DataFrame:
    """Scored predictions for `target`, test split, with cell coords attached by
    verified positional alignment against grid_hybrid.parquet."""
    pred = pd.read_parquet(RESULTS / f"predictions_{target}.parquet")
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    ycol = target  # grid label col is y30/y35/y45
    gsub = grid[grid["window"].isin(pred["window"].unique())].reset_index(drop=True)
    # evaluate.py builds predictions in grid_hybrid row order over val+test windows.
    # Verify that alignment bit-for-bit before trusting a positional coord attach.
    assert len(gsub) == len(pred), f"row count {len(gsub)} != {len(pred)}"
    assert np.array_equal(gsub["window"].to_numpy(), pred["window"].to_numpy()), "window order differs"
    assert np.array_equal(gsub[ycol].to_numpy(float), pred["y"].to_numpy(float)), "labels differ -> misaligned"
    assert np.array_equal(gsub["t0"].to_numpy(), pred["t0"].to_numpy()), "t0 differs -> misaligned"
    pred = pred.copy()
    pred["cell_lon"] = gsub["cell_lon"].to_numpy()
    pred["cell_lat"] = gsub["cell_lat"].to_numpy()
    return pred[pred["split"] == "test"].reset_index(drop=True)


def per_window_partials(df: pd.DataFrame) -> dict:
    """For each test window, accumulate zone/background partial sums of the full LL
    gain and its placement/count terms, plus the window's pre/post class."""
    y = df["y"].to_numpy(float)
    lh, lc = _lam(df[MODEL]), _lam(df[BASE])
    dloglam = np.log(lh) - np.log(lc)
    place = y * dloglam                 # P_i
    count = -(lh - lc)                  # C_i
    dll = place + count                 # full per-cell contribution

    dx = (df["cell_lon"].to_numpy() - KUMBURGAZ[0]) * KM_PER_DEG * np.cos(np.radians(KUMBURGAZ[1]))
    dy = (df["cell_lat"].to_numpy() - KUMBURGAZ[1]) * KM_PER_DEG
    zone = np.sqrt(dx * dx + dy * dy) < R_ZONE_KM
    t0 = pd.to_datetime(df["t0"].to_numpy())
    is_pre = np.asarray(t0 < SPLIT_DATE)

    wins = np.sort(df["window"].unique())
    rec = {}
    for w in wins:
        m = df["window"].to_numpy() == w
        zin = zone & m
        bin_ = (~zone) & m
        rec[int(w)] = {
            "is_pre": bool(is_pre[m][0]),
            "zone": {"dll": float(dll[zin].sum()), "P": float(place[zin].sum()),
                     "C": float(count[zin].sum()), "npos": float(y[zin].sum())},
            "bg":   {"dll": float(dll[bin_].sum()), "P": float(place[bin_].sum()),
                     "C": float(count[bin_].sum()), "npos": float(y[bin_].sum())},
        }
    return rec, wins


def quad_key(is_pre: bool, is_zone: bool) -> str:
    return f"{'pre' if is_pre else 'post'}_Kumburgaz|{'aftershock_zone' if is_zone else 'background'}"


def aggregate(rec: dict, order: np.ndarray) -> dict:
    """Sum partials over an (possibly resampled) ordered list of window ids."""
    quads = {quad_key(p, z): {"dll": 0.0, "P": 0.0, "C": 0.0, "npos": 0.0}
             for p in (True, False) for z in (True, False)}
    tot = {"dll": 0.0, "P": 0.0, "C": 0.0, "npos": 0.0}
    for w in order:
        r = rec[int(w)]
        for zflag, part in ((True, r["zone"]), (False, r["bg"])):
            q = quads[quad_key(r["is_pre"], zflag)]
            for k in ("dll", "P", "C", "npos"):
                q[k] += part[k]
                tot[k] += part[k]
    return {"total": tot, "quads": quads}


def _ci(vals):
    a = np.asarray(vals, float)
    return [round(float(np.percentile(a, 2.5)), 4), round(float(np.percentile(a, 97.5)), 4)]


def run_target(target: str) -> dict:
    df = load_test(target)
    rec, wins = per_window_partials(df)
    point = aggregate(rec, wins)
    tot_dll = point["total"]["dll"]
    npos = point["total"]["npos"]
    ig = tot_dll / max(npos, 1.0)

    # ---- bootstrap: reuse the exact paper resampler over ordered test windows ----
    rng = np.random.default_rng(SEED)
    seqs = stationary_window_indices(len(wins), B, MEAN_BLOCK, rng)
    boot_ig, boot_P_share, boot_C_share = [], [], []
    boot_quad_share = {q: [] for q in point["quads"]}
    boot_quad_dll = {q: [] for q in point["quads"]}
    for row in seqs:
        order = wins[row]
        agg = aggregate(rec, order)
        d = agg["total"]["dll"]
        n = max(agg["total"]["npos"], 1.0)
        boot_ig.append(d / n)
        boot_P_share.append(agg["total"]["P"] / d if d else np.nan)
        boot_C_share.append(agg["total"]["C"] / d if d else np.nan)
        for q in point["quads"]:
            boot_quad_dll[q].append(agg["quads"][q]["dll"])
            boot_quad_share[q].append(agg["quads"][q]["dll"] / d if d else np.nan)

    quads_out = {}
    for q, v in point["quads"].items():
        quads_out[q] = {
            "ll_gain": round(v["dll"], 2),
            "pct_of_total": round(100 * v["dll"] / tot_dll, 1),
            "pct_ci": _ci(100 * np.asarray(boot_quad_share[q])),
            "ll_gain_ci": _ci(boot_quad_dll[q]),
            "placement_nats": round(v["P"], 2),
            "count_nats": round(v["C"], 2),
            "n_pos": int(round(v["npos"])),
        }

    return {
        "target": target, "model": MODEL, "base": BASE,
        "meta": {"b_op": 1.15, "zone_km": R_ZONE_KM, "split_date": str(SPLIT_DATE.date()),
                 "B": B, "seed": SEED, "mean_block": MEAN_BLOCK, "n_windows": int(len(wins)),
                 "note": "IG(model vs base) total LL gain decomposed 2x2 (time x space) and by "
                         "additive term (placement P = y*dln-lambda over positives; count "
                         "C = -d-lambda over all cells; IG = (P+C)/n_pos). Same scoring as "
                         "metrics.information_gain; same masks as ntest_residual_probe."},
        "total_ll_gain": round(tot_dll, 2),
        "n_pos_total": int(round(npos)),
        "ig_per_event": round(ig, 4),
        "ig_per_event_ci": _ci(boot_ig),
        "term_split": {
            "placement_P_nats": round(point["total"]["P"], 2),
            "count_C_nats": round(point["total"]["C"], 2),
            "placement_share_pct": round(100 * point["total"]["P"] / tot_dll, 1),
            "count_share_pct": round(100 * point["total"]["C"] / tot_dll, 1),
            "placement_share_ci": _ci(100 * np.asarray(boot_P_share)),
            "count_share_ci": _ci(100 * np.asarray(boot_C_share)),
        },
        "marginals": {
            "background_pct": round(sum(point["quads"][q]["dll"] for q in point["quads"] if "background" in q) / tot_dll * 100, 1),
            "aftershock_zone_pct": round(sum(point["quads"][q]["dll"] for q in point["quads"] if "aftershock_zone" in q) / tot_dll * 100, 1),
            "pre_Kumburgaz_pct": round(sum(point["quads"][q]["dll"] for q in point["quads"] if q.startswith("pre")) / tot_dll * 100, 1),
            "post_Kumburgaz_pct": round(sum(point["quads"][q]["dll"] for q in point["quads"] if q.startswith("post")) / tot_dll * 100, 1),
        },
        "quadrants": quads_out,
    }


def _md(res: dict) -> str:
    L = [f"# Mechanism of the hybrid's IG gain over the {res['base']} ({res['target']}, test)", "",
         f"IG(hybrid vs {res['base']}) = **{res['ig_per_event']}** nats/event "
         f"(95% block-bootstrap CI {res['ig_per_event_ci']}); total LL gain "
         f"{res['total_ll_gain']} nats over {res['n_pos_total']} positives.", "",
         "## Additive term split (IG = placement + count)", "",
         "| term | nats | share % | share 95% CI |", "|---|---|---|---|",
         f"| placement P (y*dln-lambda, positives; \"ranking\") | {res['term_split']['placement_P_nats']} | "
         f"{res['term_split']['placement_share_pct']} | {res['term_split']['placement_share_ci']} |",
         f"| count C (-d-lambda, all cells; \"rate calibration\") | {res['term_split']['count_C_nats']} | "
         f"{res['term_split']['count_share_pct']} | {res['term_split']['count_share_ci']} |", "",
         "## Four-way split (time x space)", "",
         "| quadrant | ll gain | % of total | % 95% CI | placement | count | n_pos |",
         "|---|---|---|---|---|---|---|"]
    for q, v in res["quadrants"].items():
        L.append(f"| {q} | {v['ll_gain']} | {v['pct_of_total']} | {v['pct_ci']} | "
                 f"{v['placement_nats']} | {v['count_nats']} | {v['n_pos']} |")
    m = res["marginals"]
    L += ["", "## Marginals", "",
          f"- background: **{m['background_pct']}%** | aftershock zone (<60 km): {m['aftershock_zone_pct']}%",
          f"- pre-Kumburgaz: **{m['pre_Kumburgaz_pct']}%** | post-Kumburgaz: {m['post_Kumburgaz_pct']}%"]
    return "\n".join(L)


def main():
    out = {}
    for target in ("y30", "y35"):
        res = run_target(target)
        out[target] = res
        (RESULTS / f"mechanism_ig_split_{target}.md").write_text(_md(res))
        print(f"\n=== {target} ===")
        print(f"IG(hybrid vs {BASE}) = {res['ig_per_event']} nats/event  CI {res['ig_per_event_ci']}")
        print(f"total LL gain {res['total_ll_gain']} over {res['n_pos_total']} pos")
        print(f"TERM: placement {res['term_split']['placement_share_pct']}% "
              f"(CI {res['term_split']['placement_share_ci']})  |  "
              f"count {res['term_split']['count_share_pct']}% (CI {res['term_split']['count_share_ci']})")
        print(f"MARGINAL: background {res['marginals']['background_pct']}%  "
              f"aftershock-zone {res['marginals']['aftershock_zone_pct']}%  |  "
              f"pre {res['marginals']['pre_Kumburgaz_pct']}%  post {res['marginals']['post_Kumburgaz_pct']}%")
        for q, v in res["quadrants"].items():
            print(f"  {q:38s} {v['ll_gain']:8.1f}  {v['pct_of_total']:5.1f}%  CI{v['pct_ci']}  "
                  f"P={v['placement_nats']:.1f} C={v['count_nats']:.1f} npos={v['n_pos']}")
    # keep the legacy y30 filename the manuscript era referenced
    (RESULTS / "verify" / "mechanism_ig_split.json").write_text(json.dumps(out["y30"], indent=2))
    (RESULTS / "verify" / "mechanism_ig_split_full.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
