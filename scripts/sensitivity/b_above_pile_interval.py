"""Run 20 — Item 1 of Amendment 8: the magnitude law above the pile, with an interval.

Governed by docs/preregistration/v2_analysis_amendment_8.md (SHA-256
c5684700aa656949908640faa326c6b6f15b3a699052f627272bc26a1186e690), hashed 2026-07-16T11:25:21Z
BEFORE this ran.

Three things, per the amendment:
  (A) HEADLINE estimators of b above the Md pile (mag_w >= 4.0), two independent ways:
      binning-corrected MLE (Aki/Utsu, 0.1 bin) and b-positive (van der Elst) over a pre-specified
      delta sweep, each whole-catalogue and modern-era, each with an event-resampling bootstrap CI.
  (B) CONSISTENCY CHECK: the deconfounded triple (established point estimates, Amendment 7 §2), each
      with (i) the exact binomial interval the reduction implies -- N2|N1 ~ Binomial(N1, 10^-b dM) --
      and (ii) a block bootstrap over the counted units; the WIDER is reported.
  (C) PROPAGATION: the headline b and its interval through the live cascade -> a P(M>=6) reference
      BAND (a labelled reference inside the ensemble range, never a second central).

Unconditional. If the two headline estimators disagree materially, that is reported as a finding;
neither delta nor the threshold is tuned toward agreement. The three deconfounded estimates are NOT
averaged (nested samples). Writes results/round4/r20_deconfounded_b_interval.json.

Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/sensitivity/b_above_pile_interval.py
"""
from __future__ import annotations

import json
import pickle
import time

import numpy as np
import pandas as pd

from marmara import grid as G
from marmara.cascade import cascade_forecast
from marmara.catalog import b_positive
from marmara.paths import RESULTS

R4 = RESULTS / "round4"
LN10 = np.log(10.0)
B_BOOT = 2000
SEED = 42
DELTAS = [0.1, 0.2, 0.3, 0.4, 0.5]
MODERN = pd.Timestamp("2013-01-01")


def binned_mle(m, mc, dm=0.1):
    m = np.asarray(m, float); m = m[m >= mc - 1e-9]
    return float(np.log10(np.e) / (m.mean() - (mc - dm / 2.0))) if len(m) else None


def boot_ci(fn, seed, B=B_BOOT):
    """Percentile 95% CI of fn over event-resamples; fn takes an index resample of the population."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        v = fn(rng)
        if v is not None and np.isfinite(v):
            vals.append(v)
    if len(vals) < B // 2:
        return None
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def binomial_se(N1, N2, dM):
    p = N2 / N1
    b = -np.log10(p) / dM
    se = np.sqrt(p * (1 - p) / N1) / (p * LN10 * dM)
    return float(b), float(se)


def stationary_blocks(n, mean_block, rng):
    """Politis-Romano stationary bootstrap index sequence of length n."""
    idx = []
    while len(idx) < n:
        start = rng.integers(0, n)
        L = rng.geometric(1.0 / mean_block)
        idx.extend((start + np.arange(L)) % n)
    return np.asarray(idx[:n])


def main():
    t0 = time.time()
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    cat = cat.sort_values("datetime_utc").reset_index(drop=True)
    out = {
        "governed_by": {"amendment": "docs/preregistration/v2_analysis_amendment_8.md",
                        "sha256": "c5684700aa656949908640faa326c6b6f15b3a699052f627272bc26a1186e690"},
        "framing": ("The crossing formula b=log10(N1/N2)/dM is the classical Utsu two-threshold count "
                    "estimator, cited as classical. The FINDING is the reduction: the corrected b_op "
                    "objective / (b-independent M>=3.0 slope) cancels the cascade and equals it, so the "
                    "sweep carries no information beyond the catalogue's own counts."),
        "B_bootstrap": B_BOOT, "seed": SEED,
    }

    # ---------- (A) HEADLINE estimators on mag_w >= 4.0 ----------------------------------
    print("(A) headline estimators above the pile (mag_w >= 4.0)", flush=True)
    head = {}
    for era, mask in (("whole", np.ones(len(cat), bool)),
                      ("modern_2013", (cat["datetime_utc"] >= MODERN).to_numpy())):
        m40 = cat.loc[mask & (cat["mag_w"] >= 4.0), "mag_w"].to_numpy()
        # binned MLE + event-resample bootstrap
        b_mle = binned_mle(m40, 4.0)
        ci_mle = boot_ci(lambda rng: binned_mle(rng.choice(m40, len(m40), replace=True), 4.0), SEED)
        # b-positive delta sweep (chronological >=4.0 sequence) + resample-of-diffs bootstrap
        seq = cat.loc[mask & (cat["mag_w"] >= 4.0), "mag_w"].to_numpy()  # already time-sorted
        bpos = {}
        for d in DELTAS:
            bp = b_positive(seq, dmc=d)
            diffs = np.diff(seq); diffs = diffs[diffs >= d]
            ci = None
            if bp is not None and len(diffs) >= 30:
                def one(rng, dd=diffs, delta=d):
                    r = rng.choice(dd, len(dd), replace=True)
                    return 1.0 / (LN10 * (r.mean() - delta))
                ci = boot_ci(one, SEED)
            bpos[f"{d}"] = {"b": round(bp, 4) if bp else None, "ci95": ci, "n_pos_diffs": int(len(diffs))}
        head[era] = {"n_events_ge4.0": int(len(m40)),
                     "binned_mle": {"b": round(b_mle, 4), "ci95": ci_mle},
                     "b_positive_delta_sweep": bpos}
        print(f"  {era}: binned MLE b={b_mle:.4f} {ci_mle} (n={len(m40)}); "
              f"b-pos(0.1)={bpos['0.1']['b']} .. b-pos(0.5)={bpos['0.5']['b']}", flush=True)
    out["A_headline"] = head

    # disagreement check (finding, not nuisance)
    wm = head["whole"]["binned_mle"]["b"]; wp = head["whole"]["b_positive_delta_sweep"]["0.1"]["b"]
    out["A_estimator_agreement"] = {
        "whole_binned_mle": wm, "whole_bpositive_0.1": wp,
        "abs_gap": round(abs(wm - wp), 4) if (wm and wp) else None,
        "materially_disagree_gt_0.10": bool(wm and wp and abs(wm - wp) > 0.10),
        "note": "if True this is reported as a finding; no tuning of delta or threshold is applied"}

    # ---------- (B) CONSISTENCY CHECK: deconfounded triple, binomial vs block bootstrap ----
    print("(B) deconfounded triple: binomial vs window/event block bootstrap", flush=True)
    starts = G.window_starts(cat["datetime_utc"].max())
    pre = [t for t in starts if t + pd.Timedelta(days=30) <= pd.Timestamp("2024-01-01")]
    modern_w = [t for t in pre if t >= MODERN]

    def per_window_counts(windows, thr):
        ts = np.sort(np.asarray((cat.loc[cat["mag_w"] >= thr, "datetime_utc"] - G.REF)
                                / pd.Timedelta(days=1), dtype=float))
        return np.array([int(np.searchsorted(ts, float(G._to_days(t)) + 30.0, "left")
                             - np.searchsorted(ts, float(G._to_days(t)), "left")) for t in windows])

    def block_boot_windowed(windows, lo, hi, dM, seed):
        c_lo = per_window_counts(windows, lo); c_hi = per_window_counts(windows, hi)
        rng = np.random.default_rng(seed); n = len(windows); vals = []
        for _ in range(B_BOOT):
            ix = stationary_blocks(n, 3.0, rng)
            slo, shi = c_lo[ix].sum(), c_hi[ix].sum()
            if shi > 0 and slo > 0:
                vals.append(np.log10(slo / shi) / dM)
        return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]

    def block_boot_raw(lo, hi, dM, seed):
        ev = np.sort(cat.loc[cat["mag_w"] >= lo, "mag_w"].to_numpy())  # values, threshold-counted
        # temporal block bootstrap over the chronological >=lo event series
        chrono = cat.loc[cat["mag_w"] >= lo].sort_values("datetime_utc")["mag_w"].to_numpy()
        rng = np.random.default_rng(seed); n = len(chrono); vals = []
        for _ in range(B_BOOT):
            ix = stationary_blocks(n, 3.0, rng); s = chrono[ix]
            slo, shi = int((s >= lo).sum()), int((s >= hi).sum())
            if shi > 0 and slo > 0:
                vals.append(np.log10(slo / shi) / dM)
        return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]

    triple = {}
    specs = [("modern_3.0_to_3.5", modern_w, 3.0, 3.5, 0.5, "window"),
             ("all_pretest_3.0_to_4.0", pre, 3.0, 4.0, 1.0, "window"),
             ("whole_catalogue_3.5_to_4.5", None, 3.5, 4.5, 1.0, "raw")]
    for name, windows, lo, hi, dM, kind in specs:
        if kind == "window":
            N1 = int(per_window_counts(windows, lo).sum()); N2 = int(per_window_counts(windows, hi).sum())
            bb = block_boot_windowed(windows, lo, hi, dM, SEED)
        else:
            N1 = int((cat["mag_w"] >= lo).sum()); N2 = int((cat["mag_w"] >= hi).sum())
            bb = block_boot_raw(lo, hi, dM, SEED)
        b, se = binomial_se(N1, N2, dM)
        binom = [round(b - 1.96 * se, 4), round(b + 1.96 * se, 4)]
        binom_w = binom[1] - binom[0]; bb_w = bb[1] - bb[0]
        wider = bb if bb_w >= binom_w else binom
        triple[name] = {"N1": N1, "N2": N2, "dM": dM, "b": round(b, 4),
                        "binomial_95": binom, "binomial_halfwidth": round(1.96 * se, 4),
                        "block_bootstrap_95": bb, "reported_95_wider": wider,
                        "reported_is": "block_bootstrap" if wider is bb else "binomial"}
        print(f"  {name:26s} b={b:.4f}  binom {binom} (±{1.96*se:.3f})  block {bb}  -> report {wider}",
              flush=True)
    out["B_deconfounded_triple"] = triple
    out["B_not_averaged"] = ("nested samples; quoted as mutually consistent, not averaged. "
                             "pairwise gaps within ~1-2 binomial sigma.")

    # ---------- (C) P(M>=6) reference BAND from the headline b interval -------------------
    print("(C) P(M>=6) reference band from the headline b interval", flush=True)
    params = pickle.load(open(RESULTS / "etas" / "etas_params.pkl", "rb"))
    T0 = pd.Timestamp("2026-07-05"); t0d = float(G._to_days(T0))
    hist = cat[cat["datetime_utc"] < T0][["datetime_utc", "longitude", "latitude", "mag_w"]]
    # headline = whole-catalogue binned MLE and its bootstrap CI
    b_head = head["whole"]["binned_mle"]["b"]; ci_head = head["whole"]["binned_mle"]["ci95"]
    band = {}
    for lab, b in (("lo_b_hi_end", ci_head[1]), ("central_b", b_head), ("hi_b_lo_end", ci_head[0])):
        c = cascade_forecast(params, hist, t0d, 30.0, G.LON_C, G.LAT_C, K=10000, seed=42,
                             b=float(b), per_sim_cap=50000)
        band[lab] = {"b": round(float(b), 4), "P_M6": round(float(c["Preg6.0"]), 5)}
        print(f"  b={b:.4f} -> P(M>=6) {band[lab]['P_M6']*100:.3f}%", flush=True)
    out["C_reference_band"] = {
        "headline_estimator": "whole-catalogue binned MLE on mag_w>=4.0",
        "headline_b": b_head, "headline_ci95": ci_head,
        "P_M6_band_pct": [round(band["lo_b_hi_end"]["P_M6"] * 100, 3),
                          round(band["central_b"]["P_M6"] * 100, 3),
                          round(band["hi_b_lo_end"]["P_M6"] * 100, 3)],
        "detail": band,
        "note": "a labelled reference band inside the b-ensemble range [0.13%, 3.74%]; NOT a second central"}

    out["runtime_s"] = round(time.time() - t0, 1)
    json.dump(out, open(R4 / "r20_deconfounded_b_interval.json", "w"), indent=1, default=str)
    print(f"\n=== HEADLINE b above the pile ===")
    print(f"  whole  binned-MLE {head['whole']['binned_mle']['b']} {head['whole']['binned_mle']['ci95']}")
    print(f"  modern binned-MLE {head['modern_2013']['binned_mle']['b']} {head['modern_2013']['binned_mle']['ci95']}")
    print(f"  estimator gap (MLE vs b-pos, whole): {out['A_estimator_agreement']['abs_gap']} "
          f"(material>0.10: {out['A_estimator_agreement']['materially_disagree_gt_0.10']})")
    print(f"  P(M>=6) reference band: {out['C_reference_band']['P_M6_band_pct']} %  "
          f"(headline b {b_head}, CI {ci_head})")
    print(f"  ({out['runtime_s']}s) -> results/round4/r20_deconfounded_b_interval.json")


if __name__ == "__main__":
    main()
