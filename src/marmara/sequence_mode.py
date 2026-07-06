"""FTLS traffic-light overlay + operational sequence report.

FTLS (Gulia & Wiemer 2019) is an ALERT LAYER, not a rate multiplier. Triggered by
any M>=4.5 in the model box:
  b_ref = b-positive of the host segment's background (>=150 events, pre-sequence,
          >=2 yr); b_seq = b-positive over 50-event moving windows in the sequence.
  RED   b_seq <= 0.9 * b_ref  (relative b drop -> elevated large-aftershock hazard)
  GREEN b_seq >= 1.1 * b_ref
  ORANGE otherwise.

The sequence report combines the FTLS color with the cascade P(M>=5.5)/P(M>=6) at
7/30 d, the conditional "bigger-one-ahead" classifier score (Step 4), the host-
segment BPT renewal prior (Step 6), and a Science-2025 context line. Generated
retro-actively for the 2025 Kumburgaz sequence as the demo.

CAVEAT printed verbatim: the Gulia-Wiemer FTLS was reported to reach ~68% hit rate
but with LOW spatial coverage in the independent Italy test (Mizrahi/Gulia GJI 2025);
treat the color as a coarse alert, not a probability.

Output: results/sequence/<label>.{json,md}
Run:  "<venv>/bin/python3" -m marmara.sequence_mode
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401

from marmara import grid as G
from marmara.cascade import cascade_forecast
from marmara.catalog import b_positive
from marmara.renewal import SEGMENTS, conditional_prob

OUT = RESULTS
SEQ_DIR = OUT / "sequence"
GW_CAVEAT = ("Gulia-Wiemer FTLS caveat (Mizrahi/Gulia GJI 2025 Italy test): the "
             "traffic light reached ~68% hit rate but with LOW spatial coverage; "
             "treat the color as a coarse alert, NOT a calibrated probability.")
SCIENCE_2025 = ("Context (Science 2025, adz0072): M>5 activity has migrated eastward "
                "toward the locked Princes Islands segment, with a quiet patch near "
                "Avcilar; the Central Marmara / Kumburgaz segment is the most overdue.")
DAY_YR = 1.0 / 365.25


def _nearest_segment(lon, lat):
    names = list(SEGMENTS)
    d = [((lon - SEGMENTS[n]["mid"][0]) * 85) ** 2 + ((lat - SEGMENTS[n]["mid"][1]) * 111) ** 2
         for n in names]
    return names[int(np.argmin(d))]


def _within(cat, lon, lat, km):
    x = (cat["longitude"].to_numpy() - lon) * 85.0
    y = (cat["latitude"].to_numpy() - lat) * 111.0
    return (x * x + y * y) <= km * km


def ftls(cat, trig_time, lon, lat, seg_name):
    """Return (color, b_ref, b_seq, ratio)."""
    t = pd.to_datetime(cat["datetime_utc"])
    near = _within(cat, lon, lat, 50.0)
    # background: >=2 yr before the trigger, within 50 km
    bg = cat[near & (t < trig_time - pd.Timedelta(days=730))]
    b_ref = b_positive(bg.sort_values("datetime_utc")["mag_w"].to_numpy())
    # sequence: after the trigger start (use 30-day window from trigger), 50-event tail
    seq = cat[near & (t >= trig_time - pd.Timedelta(days=2)) & (t <= trig_time + pd.Timedelta(days=30))]
    seq = seq.sort_values("datetime_utc")
    mags = seq["mag_w"].to_numpy()
    # b-positive needs ~60+ events for 30 positive diffs; the nominal 50-event GW
    # window is too small, so use the largest recent window that yields a valid b.
    b_seq = None
    for win in (150, 300, len(mags)):
        b_seq = b_positive(mags[-win:])
        if b_seq is not None:
            break
    if b_ref is None or b_seq is None:
        return "UNKNOWN", b_ref, b_seq, None
    ratio = b_seq / b_ref
    color = "RED" if ratio <= 0.9 else "GREEN" if ratio >= 1.1 else "ORANGE"
    return color, float(b_ref), float(b_seq), float(ratio)


def sequence_report(label, trig_time, lon, lat, K=3000):
    SEQ_DIR.mkdir(parents=True, exist_ok=True)
    trig_time = pd.Timestamp(trig_time)
    with open(OUT / "etas_params.pkl", "rb") as f:
        params = pickle.load(f)
    b_op = json.load(open(OUT / "etas_fit_report.json"))["operational_b_for_cascade"]
    cat = pd.read_csv(OUT / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    hist = cat[cat["datetime_utc"] < trig_time]
    spec = G.MODEL_SPEC
    seg = _nearest_segment(lon, lat)

    # cascade P(M>=5.5)/P(M>=6) at 7 and 30 d (regional = 1 - prod(1-P_cell))
    t0d = float(G._to_days(trig_time))
    casc = {}
    for H in (7.0, 30.0):
        out = cascade_forecast(params, hist[["datetime_utc", "longitude", "latitude", "mag_w"]],
                               t0d, H, spec.lon_c, spec.lat_c, K=K, seed=77, b=b_op)
        casc[f"{int(H)}d"] = {
            "P_M5.5_region": float(1 - np.prod(1 - out["P5.5"])),
            "P_M6.0_region": float(1 - np.prod(1 - out["P6.0"])),
            "P_M5.5_peakcell": float(out["P5.5"].max()),
            "P_M6.0_peakcell": float(out["P6.0"].max()),
        }

    color, b_ref, b_seq, ratio = ftls(cat, trig_time, lon, lat, seg)

    # conditional bigger-one-ahead classifier score
    from marmara.synthetic import score_real_sequence, FEATS
    clf_d = pickle.load(open(OUT / "models" / "bigger_ahead.pkl", "rb"))
    fv = score_real_sequence(cat, trig_time, lon, lat, clf_d["ref_b"])
    clf_score = None if fv is None else float(
        clf_d["clf"].predict_proba(np.array([[fv[k] for k in FEATS]]))[:, 1][0])

    # renewal prior for host segment (30-day)
    s = SEGMENTS[seg]
    from marmara.renewal import NOW_YEAR
    renew_30d = conditional_prob(s["mu_r"], NOW_YEAR - s["last"], 30 * DAY_YR)

    report = {
        "label": label, "trigger_time": str(trig_time), "location": [lon, lat],
        "host_segment": seg,
        "ftls": {"color": color, "b_ref": b_ref, "b_seq": b_seq, "ratio": ratio,
                 "caveat": GW_CAVEAT},
        "cascade": casc,
        "bigger_one_ahead_score": clf_score,
        "renewal_prior_30d_M7": renew_30d,
        "science_2025_context": SCIENCE_2025,
    }
    json.dump(report, open(SEQ_DIR / f"{label}.json", "w"), indent=2)

    bline = (f"- b_ref (background) = {b_ref:.2f}; b_seq (sequence) = {b_seq:.2f}; "
             f"ratio = {ratio:.2f}") if ratio is not None else \
            f"- b undetermined (b_ref={b_ref}, b_seq={b_seq})"
    md = [f"# Sequence report — {label}", "",
          f"**Trigger:** {trig_time.date()} at ({lon}, {lat}); host segment **{seg}**", "",
          f"## FTLS traffic light: **{color}**",
          bline,
          f"- {GW_CAVEAT}", "",
          "## Cascade Monte-Carlo (conditional forward simulation)",
          "| horizon | P(M>=5.5) region | P(M>=6) region | P(M>=6) peak cell |",
          "|---|---|---|---|"]
    for H in ("7d", "30d"):
        c = casc[H]
        md.append(f"| {H} | {c['P_M5.5_region']*100:.2f}% | {c['P_M6.0_region']*100:.2f}% | "
                  f"{c['P_M6.0_peakcell']*100:.2f}% |")
    md += ["",
           f"## Conditional 'bigger-one-ahead' score: "
           f"**{clf_score:.3f}**" if clf_score is not None else "## classifier: n/a",
           "(probability a sim event >= largest_so_far - 0.3 follows within 30 d, "
           "from the synthetic conditional classifier)", "",
           f"## Host-segment renewal prior (BPT): P(M~7 in 30 d) = **{renew_30d*100:.4f}%**",
           f"({seg}, {NOW_YEAR - s['last']:.0f} yr since last, mean recurrence {s['mu_r']} yr)",
           "", f"## {SCIENCE_2025}"]
    (SEQ_DIR / f"{label}.md").write_text("\n".join(md))
    rtxt = f"{ratio:.2f}" if ratio is not None else f"n/a (b_ref={b_ref},b_seq={b_seq})"
    cs = f"{clf_score:.3f}" if clf_score is not None else "n/a"
    print(f"{label}: FTLS {color} (b_seq/b_ref={rtxt}); cascade 30d P(M6) "
          f"{casc['30d']['P_M6.0_region']*100:.2f}%; bigger-ahead {cs}; "
          f"renewal30d {renew_30d*100:.4f}%")
    return report


def main():
    # demo: 2025 Kumburgaz sequence, the day after the Apr-23 M6.2
    sequence_report("kumburgaz_2025", "2025-04-24", 28.23, 40.84)


if __name__ == "__main__":
    main()
