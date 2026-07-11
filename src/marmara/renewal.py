"""Historical M~7 record + BPT renewal priors for the Marmara segments.

Historical magnitudes are approximate (flagged); source Ambraseys BSSA 2002 &
Parsons JGR 2004. BPT (Brownian passage time = inverse Gaussian) with aperiodicity
alpha=0.5: invgauss(mu=alpha^2, scale=mu_r/alpha^2) has mean=mu_r, CV=alpha (verified).
30-day and 30-yr conditional probabilities P = [F(T+dt)-F(T)]/[1-F(T)] with T the
elapsed time since the segment's last M~7.

SANITY: the combined Princes-Islands + Central-Marmara 30-yr probability should land
in the Parsons-2004 ~30-50% ballpark (order-of-magnitude assertion, printed).

Provides: per-cell host-segment 30-day renewal P (an ML/forecast feature) and the
product blend P_combined(M>=6.8) = 1-(1-P_cascade)(1-P_renewal). Renewal is a
LARGE-EVENT layer only, never blended into the y35/y45 evaluation.

Output: results/renewal_report.json
Run:  "<venv>/bin/python3" -m marmara.renewal
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401
from scipy.stats import invgauss

OUT = RESULTS
NOW_YEAR = 2026.51            # 2026-07-06
ALPHA = 0.5                   # BPT aperiodicity
DAY_YR = 1.0 / 365.25

# Historical M~7 record (approximate magnitudes, flagged).
HISTORICAL = [
    (1509, 7.2, "Central Marmara/Cinarcik"),
    (1719, 7.4, "Izmit"),
    (1754, 6.9, "Cinarcik (east)"),
    (1766.4, 7.1, "Cinarcik-Princes Islands"),
    (1766.6, 7.4, "Ganos (west)"),
    (1894, 7.0, "Princes Islands/Yalova"),
    (1912, 7.3, "Ganos/Murefte"),
    (1999, 7.6, "Izmit"),
]

# Segment renewal table: last M~7 year, mean recurrence (yr), approx midpoint (lon,lat).
SEGMENTS = {
    "Izmit":            {"last": 1999, "mu_r": 250, "mid": (30.1, 40.72)},
    "PrincesIslands":   {"last": 1894, "mu_r": 250, "mid": (29.05, 40.75)},
    "CentralMarmara":   {"last": 1766.4, "mu_r": 250, "mid": (28.2, 40.85)},
    "Ganos":            {"last": 1912, "mu_r": 280, "mid": (27.2, 40.62)},
}


def _bpt(mu_r):
    return invgauss(mu=ALPHA ** 2, scale=mu_r / ALPHA ** 2)


def conditional_prob(mu_r, elapsed_yr, dt_yr):
    """P(rupture in [T, T+dt) | survived to T) for a BPT renewal process."""
    d = _bpt(mu_r)
    F_T = d.cdf(elapsed_yr)
    F_T2 = d.cdf(elapsed_yr + dt_yr)
    denom = max(1.0 - F_T, 1e-12)
    return float((F_T2 - F_T) / denom)


def segment_probs() -> dict:
    out = {}
    for name, s in SEGMENTS.items():
        elapsed = NOW_YEAR - s["last"]
        out[name] = {
            "last_rupture": s["last"], "mean_recurrence_yr": s["mu_r"],
            "elapsed_yr": round(elapsed, 1),
            "P_30day": conditional_prob(s["mu_r"], elapsed, 30 * DAY_YR),
            "P_30yr": conditional_prob(s["mu_r"], elapsed, 30.0),
            "midpoint": s["mid"],
        }
    return out


def renewal_prob_per_cell(cell_lon, cell_lat, dt_yr=30 * DAY_YR) -> np.ndarray:
    """Per-cell host-segment renewal probability over dt (default 30 days):
    each cell inherits the 30-d conditional P of its nearest renewal segment."""
    cell_lon = np.asarray(cell_lon, float); cell_lat = np.asarray(cell_lat, float)
    names = list(SEGMENTS)
    mids = np.array([SEGMENTS[n]["mid"] for n in names])
    probs = np.array([conditional_prob(SEGMENTS[n]["mu_r"], NOW_YEAR - SEGMENTS[n]["last"], dt_yr)
                      for n in names])
    # nearest segment midpoint (simple degrees distance; adequate for assignment)
    d2 = ((cell_lon[:, None] - mids[None, :, 0]) ** 2
          + (cell_lat[:, None] - mids[None, :, 1]) ** 2)
    nearest = np.argmin(d2, axis=1)
    return probs[nearest]


def combined_large_event(P_cascade, P_renewal):
    """Independence-approximation blend for M>=6.8+ (documented approximation)."""
    return 1.0 - (1.0 - np.asarray(P_cascade)) * (1.0 - np.asarray(P_renewal))


def main():
    OUT.mkdir(exist_ok=True)
    segs = segment_probs()
    # sanity anchor: combined Princes + Central 30-yr probability vs Parsons ~30-50%
    p_pr = segs["PrincesIslands"]["P_30yr"]; p_ce = segs["CentralMarmara"]["P_30yr"]
    combined_30yr = 1.0 - (1.0 - p_pr) * (1.0 - p_ce)
    parsons_ok = 0.15 <= combined_30yr <= 0.70   # order-of-magnitude around 30-50%

    report = {
        "now_year": NOW_YEAR, "bpt_aperiodicity": ALPHA,
        "historical_M7_record": [{"year": y, "M_approx": m, "segment": s}
                                 for y, m, s in HISTORICAL],
        "historical_note": "Magnitudes approximate (Ambraseys 2002; Parsons 2004).",
        "segments": segs,
        "sanity_anchor": {
            "combined_Princes_Central_30yr": round(combined_30yr, 3),
            "Parsons_2004_ballpark": "0.30-0.50",
            "within_order_of_magnitude": bool(parsons_ok),
        },
    }
    json.dump(report, open(OUT / "renewal_report.json", "w"), indent=2)
    print("segment 30-day / 30-yr conditional P(M~7):")
    for n, s in segs.items():
        print(f"  {n:16s} elapsed {s['elapsed_yr']:.0f}y  P30d={s['P_30day']*100:.4f}%  P30yr={s['P_30yr']*100:.1f}%")
    print(f"\ncombined Princes+Central 30-yr = {combined_30yr*100:.1f}%  "
          f"(Parsons 2004 ~30-50%; within band: {parsons_ok})")


if __name__ == "__main__":
    main()
