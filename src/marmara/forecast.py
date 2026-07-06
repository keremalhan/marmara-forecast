"""Live forecast products from the current catalog end (2026-07-05).

Maps P(y35)/P(y45) from the hybrid; per-cell cascade P_sim(M>=5)/P_sim(M>=6) with a
b-ensemble range {b_aki, b_pos} bracketing the central b_op; per-segment BPT renewal
priors; combined large-event line P(M>=6.8) = 1-(1-P_cascade_M6)(1-P_renewal). The
Kumburgaz sequence report (Step 5) is generated separately.

Output: results/forecast/forecast_2026-07-05/
Run:  "<venv>/bin/python3" -m marmara.forecast
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401

from marmara import grid as G
from marmara.cascade import cascade_forecast
from marmara.metrics import p_to_lambda, lambda_to_p
from marmara.renewal import SEGMENTS, NOW_YEAR, conditional_prob, renewal_prob_per_cell

OUT = RESULTS
MODELS = OUT / "models"
FC = OUT / "forecast" / "forecast_2026-07-05"
T0 = pd.Timestamp("2026-07-05")
K_LIVE = 10000
DAY_YR = 1.0 / 365.25
EPS = 1e-9


def _map(P, title, path):
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(P, origin="lower", aspect="auto", cmap="inferno",
                   extent=[G.LON_C[0] - .05, G.LON_C[-1] + .05, G.LAT_C[0] - .05, G.LAT_C[-1] + .05])
    ax.set_title(title); ax.set_xlabel("longitude"); ax.set_ylabel("latitude")
    fig.colorbar(im, ax=ax, label="P in 30 d")
    ax.text(0.01, 0.01, "hybrid (cascade x ML); preliminary tail in features",
            transform=ax.transAxes, fontsize=7, color="cyan")
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def nearest_place(cat, lon, lat):
    d = (cat["longitude"] - lon) ** 2 + (cat["latitude"] - lat) ** 2
    return str(cat.iloc[int(d.values.argmin())]["location_name"])


def hybrid_P(feats19, lam_sim, ycol):
    d = pickle.load(open(MODELS / f"count{ycol}_hybrid.pkl", "rb"))
    X = pd.DataFrame({k: feats19[k].ravel() for k in G.FEATURES})
    X["ln_lam_sim"] = np.log(lam_sim.ravel() + EPS)
    lam_ml = np.clip(d["reg"].predict(X), EPS, None)
    lam_sim_c = np.clip(lam_sim.ravel(), EPS, None)
    lam_h = lam_sim_c ** (1 - d["w"]) * lam_ml ** d["w"]
    return d["iso"].transform(lambda_to_p(lam_h))


def main():
    FC.mkdir(parents=True, exist_ok=True)
    params = pickle.load(open(OUT / "etas_params.pkl", "rb"))
    rep = json.load(open(OUT / "etas_fit_report.json"))
    b_op = rep["operational_b_for_cascade"]; b_aki, b_pos = rep["b_aki"], rep["b_positive"]
    cat = pd.read_csv(OUT / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    spec = G.MODEL_SPEC
    EV = G.build_event_bundle(cat, 3.0)
    ctx = G.build_static_context()
    t0d = float(G._to_days(T0))
    hist = cat[cat["datetime_utc"] < T0][["datetime_utc", "longitude", "latitude", "mag_w"]]

    feats = G.features_at_window(EV, t0d, T0, ctx, params)
    casc_op = cascade_forecast(params, hist, t0d, 30.0, spec.lon_c, spec.lat_c, K=K_LIVE, seed=11, b=b_op)
    P35 = hybrid_P(feats, casc_op["lam35"], "35").reshape(G.NLAT, G.NLON)
    P45 = hybrid_P(feats, casc_op["lam45"], "45").reshape(G.NLAT, G.NLON)
    _map(P35, f"P(M>=3.5 in 30d) from {T0.date()}", FC / "P_y35.png")
    _map(P45, f"P(M>=4.5 in 30d) from {T0.date()}", FC / "P_y45.png")

    # cascade M5/M6 b-ensemble (per-cell P_sim)
    ens = {}
    for tag, bb in (("b_aki", b_aki), ("b_op", b_op), ("b_pos", b_pos)):
        c = cascade_forecast(params, hist, t0d, 30.0, spec.lon_c, spec.lat_c, K=K_LIVE, seed=11, b=bb)
        ens[tag] = c

    def reg_P(c, lvl):
        return float(1 - np.prod(1 - c[f"P{lvl}"]))

    renew_cell = renewal_prob_per_cell(np.repeat(G.LAT_C * 0 + 1, 1) * 0 + 0, 0) if False else None
    LO, LA = np.meshgrid(G.LON_C, G.LAT_C)
    renew_cell = renewal_prob_per_cell(LO.ravel(), LA.ravel())

    # top-20 by P35
    P5_lo = ens["b_pos"]["P5.0"].ravel(); P5_hi = ens["b_aki"]["P5.0"].ravel()
    P6_lo = ens["b_pos"]["P6.0"].ravel(); P6_hi = ens["b_aki"]["P6.0"].ravel()
    P6_op = ens["b_op"]["P6.0"].ravel()
    combined = 1 - (1 - P6_op) * (1 - renew_cell)
    df = pd.DataFrame({"lon": LO.ravel(), "lat": LA.ravel(), "P35": P35.ravel(), "P45": P45.ravel(),
                       "Psim_M5_lo": P5_lo, "Psim_M5_hi": P5_hi,
                       "Psim_M6_lo": P6_lo, "Psim_M6_hi": P6_hi,
                       "renewal_30d": renew_cell, "combined_M6plus": combined})
    df = df.sort_values("P35", ascending=False).head(20).reset_index(drop=True)
    df["place"] = [nearest_place(cat, r.lon, r.lat) for r in df.itertuples()]
    df.to_csv(FC / "top20_cells.csv", index=False)

    # regional summary + per-segment renewal
    seg_renew = {n: conditional_prob(s["mu_r"], NOW_YEAR - s["last"], 30 * DAY_YR)
                 for n, s in SEGMENTS.items()}
    reg = {
        "t0": str(T0.date()), "K_live": K_LIVE, "b_op": b_op, "b_ensemble": [b_aki, b_pos],
        "regional_30d": {
            "P_M5_range": [reg_P(ens["b_pos"], 5.0), reg_P(ens["b_aki"], 5.0)],
            "P_M5.5_range": [reg_P(ens["b_pos"], 5.5), reg_P(ens["b_aki"], 5.5)],
            "P_M6_range": [reg_P(ens["b_pos"], 6.0), reg_P(ens["b_aki"], 6.0)],
            "P_M6_central_b_op": reg_P(ens["b_op"], 6.0),
        },
        "segment_renewal_30d_M7": seg_renew,
        "combined_note": ("combined_M6plus per cell = 1-(1-P_cascade_M6)(1-P_renewal); "
                          "independence approximation, large-event layer only."),
        "top_cell": {"place": df.iloc[0]["place"], "lon": float(df.iloc[0]["lon"]),
                     "lat": float(df.iloc[0]["lat"]), "P35": float(df.iloc[0]["P35"])},
    }
    json.dump(reg, open(FC / "forecast_summary.json", "w"), indent=2)

    rr = reg["regional_30d"]
    md = [f"# Live forecast — {T0.date()}, next 30 days", "",
          "Hybrid (cascade x ML) probability maps + cascade-simulation M5/M6 with a "
          "b-ensemble range, and BPT segment renewal priors.", "",
          "## Regional 30-day probabilities (cascade Monte-Carlo, b-ensemble range)",
          "| magnitude | P (region, 30d) |", "|---|---|",
          f"| M>=5   | {rr['P_M5_range'][0]*100:.1f}% - {rr['P_M5_range'][1]*100:.1f}% |",
          f"| M>=5.5 | {rr['P_M5.5_range'][0]*100:.2f}% - {rr['P_M5.5_range'][1]*100:.2f}% |",
          f"| M>=6   | {rr['P_M6_range'][0]*100:.2f}% - {rr['P_M6_range'][1]*100:.2f}% (central {rr['P_M6_central_b_op']*100:.2f}%) |",
          "", "(range = b_pos 1.54 low end to b_aki 1.02 high end; central b_op 1.2)", "",
          "## Segment renewal priors (BPT, P(M~7 in 30 d))", "| segment | elapsed yr | P(30d) |", "|---|---|---|"]
    for n, s in SEGMENTS.items():
        md.append(f"| {n} | {NOW_YEAR - s['last']:.0f} | {seg_renew[n]*100:.4f}% |")
    md += ["", f"Highest 30-day M>=3.5 cell: **{reg['top_cell']['place']}** "
           f"(lon {reg['top_cell']['lon']}, lat {reg['top_cell']['lat']}), "
           f"P(M>=3.5)={reg['top_cell']['P35']*100:.1f}%.",
           "", "See top20_cells.csv for per-cell M5/M6 and combined large-event numbers, "
           "and sequence/kumburgaz_2025.md for the FTLS sequence report."]
    (FC / "m5_m6_summary.md").write_text("\n".join(md))
    print(f"forecast written to {FC}")
    print(f"  regional 30d P(M>=6): {rr['P_M6_range'][0]*100:.2f}%-{rr['P_M6_range'][1]*100:.2f}% "
          f"(central {rr['P_M6_central_b_op']*100:.2f}%)")
    print(f"  top cell: {reg['top_cell']['place']} P35={reg['top_cell']['P35']*100:.1f}%")


if __name__ == "__main__":
    main()
