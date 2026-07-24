"""Item F: Sindirgi (2025 M6.1, Balikesir, below the model box) wide-box worked example
+ information-arrival freeze schedule. Third worked event, opposite pole from the
cold-start Kumburgaz Mw6.2: Sindirgi was sequence-preceded, so we expect earlier temporal
information. Uses the WIDE box (Sindirgi ~39.2N is outside the model box lat>=39.6) and the
canonical surgical cascade. Writes results/sindirgi_worked_example.{json,md}.
"""
from __future__ import annotations
import json, pickle, time
import numpy as np, pandas as pd
from marmara.paths import RESULTS
from marmara import grid as G
from marmara.cascade import cascade_forecast

OUT = RESULTS; B_OP = 1.15; K = 500


def main():
    t0_ = time.time()
    cat = pd.read_csv(OUT / "catalog" / "catalog_widebox.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    cand = cat[(cat.datetime_utc >= pd.Timestamp("2025-08-01")) & (cat.datetime_utc < pd.Timestamp("2025-09-01"))
               & (cat.mag_w >= 5.8) & (cat.latitude < 39.6)]
    if not len(cand):
        print("no Sindirgi M>=5.8 found in Aug 2025 wide-box catalogue"); return
    main = cand.loc[cand.mag_w.idxmax()]
    lon0, lat0, tm, mm = float(main.longitude), float(main.latitude), main.datetime_utc, float(main.mag_w)
    print(f"Sindirgi mainshock: {tm} ({lat0:.2f}N,{lon0:.2f}E) M{mm}", flush=True)

    spec = G.WIDE_SPEC
    hist = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]
    with open(OUT / "etas" / "etas_params.pkl", "rb") as f:
        params = pickle.load(f)
    ic = int(np.argmin(np.abs(np.asarray(spec.lon_c) - lon0)))
    ir = int(np.argmin(np.abs(np.asarray(spec.lat_c) - lat0)))
    ncells = spec.nlat * spec.nlon

    leads = [("T-365d", 365), ("T-90d", 90), ("T-30d", 30), ("T-7d", 7), ("T-1d (2025-08-09)", 1)]
    rows = []
    for lab, dd in leads:
        t0 = tm.normalize() - pd.Timedelta(days=dd); t0d = float(G._to_days(t0))
        casc = cascade_forecast(params, hist[cat.datetime_utc < t0], t0d, G.HORIZON_D,
                                spec.lon_c, spec.lat_c, K=K, seed=9000, b=B_OP, preserve_branching=True)
        lam35 = casc["lam35"]; cell_lam = float(lam35[ir, ic])
        pctile = float((lam35.ravel() <= cell_lam).mean() * 100.0)
        p6 = casc.get("P6.0"); p6cell = float(p6[ir, ic]) if p6 is not None else 0.0
        preg6 = float(casc.get("Preg6.0", 0.0)); base6 = preg6 / ncells
        gain = round(p6cell / base6, 1) if base6 > 0 else None
        rows.append({"freeze": lab, "t0": str(t0.date()), "epicentral_lam35": round(cell_lam, 4),
                     "seismicity_pctile": round(pctile, 1), "P6_cell": round(p6cell, 5),
                     "regional_P6": round(preg6, 5), "gain6_vs_uniform": gain})
        print(f"  {lab}: seismicity pctile {pctile:.1f}, P6_cell {p6cell:.5f}, "
              f"regional_P6 {preg6:.5f}, gain {gain} ({time.time()-t0_:.0f}s)", flush=True)

    out = {"mainshock": {"time": str(tm), "lon": lon0, "lat": lat0, "mag": mm},
           "epicentral_cell": {"ir": ir, "ic": ic, "lon": round(float(spec.lon_c[ic]), 3),
                               "lat": round(float(spec.lat_c[ir]), 3)},
           "freeze_schedule": rows,
           "note": "seismicity pctile = rank of the epicentral cell's lam35 among all wide-box cells; "
                   "rising toward T-0 = temporal information arriving (Sindirgi sequence-preceded)."}
    (OUT / "sindirgi_worked_example.json").write_text(json.dumps(out, indent=2))
    L = ["# Item F: Sindirgi (2025 M6.1) wide-box worked example + freeze schedule", "",
         f"Mainshock {tm} ({lat0:.2f}N, {lon0:.2f}E) M{mm}; epicentral wide-box cell "
         f"({out['epicentral_cell']['lat']}N, {out['epicentral_cell']['lon']}E). Canonical surgical cascade.", "",
         "| freeze | seismicity pctile | P6 cell | regional P6 | gain6 (vs uniform) |", "|---|---|---|---|---|"]
    for r in rows:
        L.append(f"| {r['freeze']} | {r['seismicity_pctile']} | {r['P6_cell']} | {r['regional_P6']} | {r['gain6_vs_uniform']} |")
    L += ["", "Compare Kumburgaz (cold-start Mw6.2): epicentral cell was top ~1% (99th pctile) all year "
          "yet near-unforecastable in time. Sindirgi (sequence-preceded) is the opposite pole -- read the "
          "percentile trajectory toward T-0 for earlier temporal information."]
    (OUT / "sindirgi_worked_example.md").write_text("\n".join(L))
    print("wrote results/sindirgi_worked_example.{json,md}")


if __name__ == "__main__":
    main()
