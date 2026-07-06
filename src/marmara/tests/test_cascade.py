"""Gate — cascade forecaster.

(a) causality: future events (>= t0) in the history change nothing.
(b) reliability: on 50 synthetic catalogs, cascade-predicted per-cell lambda vs
    realized counts has regression slope in [0.8, 1.2].
(c) cascade >= first-generation expected_counts at matched threshold (secondary
    triggering adds events, by construction).

Run:  "<venv>/bin/python3" python -m marmara.tests.test_cascade.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import marmara.grid as G                                            # noqa: E402
from marmara.cascade import cascade_forecast, simulate_window       # noqa: E402
from marmara.etas_model import expected_counts, simulate_catalogs   # noqa: E402

OUT = RESULTS


def _cells_df(spec):
    LO, LA = np.meshgrid(spec.lon_c, spec.lat_c)
    return pd.DataFrame({"cell_lon": LO.ravel(), "cell_lat": LA.ravel()})


def test_causality(params, spec):
    cat = pd.read_csv(OUT / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    t0 = pd.Timestamp("2015-06-01"); t0d = float(G._to_days(t0))
    hist = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]
    clean = cascade_forecast(params, hist[cat["datetime_utc"] < t0], t0d, 30.0,
                             spec.lon_c, spec.lat_c, K=300, seed=7)
    dirty = cascade_forecast(params, hist, t0d, 30.0, spec.lon_c, spec.lat_c, K=300, seed=7)
    ok = np.allclose(clean["lam35"], dirty["lam35"]) and np.allclose(clean["P6.0"], dirty["P6.0"])
    print(f"  (a) causality: future events ignored = {ok}")
    return ok


def test_reliability(params, spec, n_cats=50):
    """Predicted per-cell lambda vs realized counts over synthetic catalogs."""
    dur = 12 * 365.25
    sims = simulate_catalogs(params, dur, n_cats, mmax=7.6, seed=123)
    pred, real = [], []
    for s in sims:
        s = s.rename(columns={})  # cols: time,lat,lon,mag
        for t0d in (2000.0, 3000.0, 3800.0):
            hist = pd.DataFrame({"time": s["time"].to_numpy(), "lon": s["lon"].to_numpy(),
                                 "lat": s["lat"].to_numpy(), "mag": s["mag"].to_numpy()})
            # isotropic mode: the synthetic truth is generated isotropically by the
            # (protected) simulate_catalogs, so calibration is only fairly testable
            # isotropically; anisotropy is a real-data refinement (aftershocks elongate
            # along faults) validated separately in test_anisotropy_effect.
            out = cascade_forecast(params, hist, t0d, 30.0, spec.lon_c, spec.lat_c,
                                   K=400, seed=1, anisotropy=False)
            # realized >=3.5 counts per cell in [t0, t0+30)
            fut = s[(s["time"] >= t0d) & (s["time"] < t0d + 30.0) & (s["mag"] >= 3.5)]
            ir, ic = G.cell_index_spec(fut["lon"].to_numpy(), fut["lat"].to_numpy(), spec)
            on = ir >= 0
            gr = np.zeros((spec.nlat, spec.nlon))
            np.add.at(gr, (ir[on], ic[on]), 1.0)
            pred.append(out["lam35"].ravel()); real.append(gr.ravel())
    pred = np.concatenate(pred); real = np.concatenate(real)
    slope = float((pred * real).sum() / max((pred * pred).sum(), 1e-12))  # regress through 0
    print(f"  (b) reliability slope (realized ~ predicted) = {slope:.3f}  "
          f"(pred sum {pred.sum():.0f}, real sum {real.sum():.0f})")
    return 0.8 <= slope <= 1.2


def test_cascade_gt_firstgen(params, spec):
    """Day after an M6, cascade (all generations) > first-generation, matched >=mc."""
    cat = pd.read_csv(OUT / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    t0 = pd.Timestamp("2025-04-25"); t0d = float(G._to_days(t0))
    hist = cat[cat["datetime_utc"] < t0]
    h = hist[hist["mag_w"] >= params.mc]
    td = G._to_days(h["datetime_utc"]).to_numpy()
    rng = np.random.default_rng(0)
    K = 400
    sim, lon, lat, mag = simulate_window(params, td, h["longitude"].to_numpy(),
                                         h["latitude"].to_numpy(), h["mag_w"].to_numpy(),
                                         t0d, 30.0, K, rng, params.b)
    casc_mc = float((mag >= params.mc).sum()) / K            # mean count >=mc, all gens
    hdf = pd.DataFrame({"datetime": h["datetime_utc"].values, "latitude": h["latitude"].values,
                        "longitude": h["longitude"].values, "mag": h["mag_w"].values})
    fg = float(expected_counts(params, hdf, _cells_df(spec), t0, 30.0).sum())
    print(f"  (c) day-after-M6: cascade(>=mc) sum {casc_mc:.1f} vs first-gen {fg:.1f}  "
          f"ratio {casc_mc/max(fg,1e-9):.2f}")
    return casc_mc > fg


def test_anisotropy_effect(params, spec):
    """Anisotropy must elongate an M>=5.5 parent's offspring along the fault: the
    along-strike spread of its direct offspring exceeds the across-strike spread."""
    from marmara.cascade import _place_offspring, _strike_angle
    rng = np.random.default_rng(3)
    lon0, lat0 = 28.23, 40.84                       # Kumburgaz, on the Marmara fault
    n = 20000
    px = np.zeros(n); py = np.zeros(n); pm = np.full(n, 6.2)
    a = float(_strike_angle(np.array([lon0]), np.array([lat0]))[0])
    orient = np.full(n, a)
    from marmara.etas_model import _project_km
    # shift params ref so projection is around the source (use params region center ok)
    lo_i, la_i = _place_offspring(np.full(n, 0.0), np.full(n, 0.0), pm, params, rng, orient=None)
    rng = np.random.default_rng(3)
    lo_a, la_a = _place_offspring(np.full(n, 0.0), np.full(n, 0.0), pm, params, rng, orient=orient)
    # project offsets back to km in strike frame
    ref_lon, ref_lat = params.background_xy.ref_lon, params.background_xy.ref_lat
    def strike_frame_spread(lo, la):
        dx, dy = _project_km(lo, la, ref_lon, ref_lat)
        # rotate into strike frame
        along = dx * np.cos(a) + dy * np.sin(a)
        across = -dx * np.sin(a) + dy * np.cos(a)
        return np.std(along), np.std(across)
    al_i, ac_i = strike_frame_spread(lo_i, la_i)
    al_a, ac_a = strike_frame_spread(lo_a, la_a)
    ratio_iso = al_i / max(ac_i, 1e-9); ratio_ani = al_a / max(ac_a, 1e-9)
    print(f"  (d) anisotropy: along/across spread iso {ratio_iso:.2f} -> aniso {ratio_ani:.2f} "
          f"(expect ~3x elongation)")
    return ratio_ani > 2.0 and ratio_iso < 1.5


def main():
    params = G.load_params()   # canonical ETAS params (the gate is param-agnostic)
    spec = G.MODEL_SPEC
    t = time.time()
    a = test_causality(params, spec)
    c = test_cascade_gt_firstgen(params, spec)
    d = test_anisotropy_effect(params, spec)
    b = test_reliability(params, spec)
    ok = a and b and c and d
    print(f"\n{'PASS' if ok else 'FAIL'} cascade gate ({time.time()-t:.0f}s): "
          f"causality={a} reliability={b} cascade>firstgen={c} anisotropy={d}")
    if ok:
        import json
        (OUT / "cascade_ok.json").write_text(json.dumps({"passed": True}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
