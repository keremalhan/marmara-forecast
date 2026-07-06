"""Tests for marmara.etas_model (ETAS fit / forecast / simulation).

Run with pytest, or directly (no pytest needed):
    "<venv>/bin/python3" python -m marmara.tests.test_etas.py

Everything is seeded; total runtime is well under 15 minutes.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from marmara.etas_model import (  # noqa: E402
    BackgroundField,
    EtasParams,
    branching_ratio,
    expected_counts,
    fit_etas,
    prob_at_least_one,
    simulate_catalogs,
)

CATALOG_CSV = DATA / "marmara_catalog_processed.csv"   # bundled ETAS-test fixture

# Known synthetic truth: subcritical (branching ratio ~0.38 with mmax=7.6),
# ~8k events over 20 years in a Marmara-sized box.
REGION = {"min_lon": 27.0, "max_lon": 29.5, "min_lat": 40.2, "max_lat": 41.3}
TRUE = EtasParams(mu_total=0.7, background_xy=BackgroundField.uniform(REGION),
                  k=0.09, alpha=1.8, c=0.012, p=1.15, d=1.5, q=1.9, gamma=0.4,
                  mc=2.5, b=1.0, region=REGION)
DURATION = 7300.0  # days (~20 y)
MMAX = 7.6

_cache = {}


def _grid_cells():
    lons = np.arange(REGION["min_lon"] + 0.05, REGION["max_lon"], 0.1)
    lats = np.arange(REGION["min_lat"] + 0.05, REGION["max_lat"], 0.1)
    LO, LA = np.meshgrid(lons, lats)
    return pd.DataFrame({"cell_lon": LO.ravel(), "cell_lat": LA.ravel()})


def _synth_catalog() -> pd.DataFrame:
    if "cat" not in _cache:
        _cache["cat"] = simulate_catalogs(TRUE, DURATION, 1, mmax=MMAX, seed=42)[0]
    return _cache["cat"]


def test_1_synthetic_recovery():
    """THE correctness test: simulate from known params, refit, recover them."""
    cat = _synth_catalog()
    n_true = branching_ratio(TRUE, MMAX)
    assert n_true < 0.9, "synthetic truth must be subcritical"
    assert 5000 <= len(cat) <= 15000, f"unexpected synthetic size {len(cat)}"
    # tau_max_days=730 captures ~81% of the true Omori mass (c=0.012, p=1.15) and
    # keeps this fit ~4x faster than the 3650 d default used for the real catalog.
    fit = fit_etas(cat, mc=2.5, fit_start=0.0, fit_end=DURATION, region=REGION,
                   tau_max_days=730.0)
    n_fit = branching_ratio(fit, MMAX)
    print(f"  recovered vs true: p {fit.p:.3f}/{TRUE.p} alpha {fit.alpha:.2f}/{TRUE.alpha} "
          f"n {n_fit:.3f}/{n_true:.3f} mu {fit.mu_total:.3f}/{TRUE.mu_total} "
          f"c {fit.c:.4f}/{TRUE.c} k {fit.k:.4f}/{TRUE.k} b {fit.b:.2f}/{TRUE.b}")
    assert abs(fit.p - TRUE.p) <= 0.15, f"p not recovered: {fit.p}"
    assert abs(fit.alpha - TRUE.alpha) <= 0.4, f"alpha not recovered: {fit.alpha}"
    assert abs(n_fit - n_true) <= 0.2, f"branching ratio not recovered: {n_fit}"
    _cache["fit_synth"] = fit


def test_2_real_marmara_fit():
    """Fit the real Marmara catalog (mc=2.5, 2003 -> 2020) and sanity-check params."""
    df = pd.read_csv(CATALOG_CSV)
    fit = fit_etas(df, mc=2.5, fit_start=pd.Timestamp("2003-01-01"),
                   fit_end=pd.Timestamp("2020-01-01"))
    n = branching_ratio(fit)
    print(f"  Marmara mc=2.5 2003-2020: mu {fit.mu_total:.4f}/d k {fit.k:.4f} "
          f"alpha {fit.alpha:.3f} c {fit.c:.4f}d p {fit.p:.3f} d {fit.d:.2f}km2 "
          f"q {fit.q:.2f} gamma {fit.gamma:.2f} b {fit.b:.3f} branching {n:.3f}")
    assert getattr(fit, "_fit_result").success, "optimizer did not converge"
    assert 0.8 < fit.p < 1.6, f"p out of range: {fit.p}"
    assert 0.7 < fit.b < 1.4, f"b out of range: {fit.b}"
    assert 0.05 < n < 0.98, f"branching ratio out of range: {n}"
    _cache["fit_real"] = fit


def test_3_causality():
    """expected_counts must ignore events at/after t0 even if present in history_df."""
    cat = _synth_catalog()
    cells = _grid_cells()
    t0 = 3000.0
    past = cat[cat["time"] < t0].copy()
    fake_future = pd.DataFrame({
        "time": [t0 + 50.0, t0], "lat": [40.75, 40.85], "lon": [28.25, 28.35],
        "mag": [7.0, 6.5], "gen": [0, 0], "parent_id": [-1, -1],
    })
    with_future = pd.concat([past, fake_future], ignore_index=True)
    out_clean = expected_counts(TRUE, past, cells, t0, horizon_days=30.0)
    out_dirty = expected_counts(TRUE, with_future, cells, t0, horizon_days=30.0)
    assert np.array_equal(out_clean, out_dirty), \
        "future events changed the forecast (causality violated)"
    assert out_clean.shape == (len(cells),) and np.all(out_clean >= 0)
    # datetime path: history entirely at/after t0 must reduce to background-only
    t0_ts = pd.Timestamp("2020-01-01")
    hist_dt = pd.DataFrame({
        "datetime": t0_ts + pd.to_timedelta([0.0, 3.0, 10.0], unit="D"),
        "latitude": [40.7, 40.8, 40.9], "longitude": [28.2, 28.3, 28.4],
        "magnitude": [7.0, 6.0, 5.0],
    })
    out_dt = expected_counts(TRUE, hist_dt, cells, t0_ts, horizon_days=30.0)
    out_bg = expected_counts(TRUE, hist_dt.iloc[:0], cells, t0_ts, horizon_days=30.0)
    assert np.array_equal(out_dt, out_bg), "datetime causal filtering failed"
    print(f"  causality OK ({len(cells)} cells; future M7 ignored, "
          f"datetime path consistent)")


def test_4_rate_sanity():
    """After an M6, the source cell rate is >> background and decays with elapsed time."""
    cells = _grid_cells()
    src_lon, src_lat = 28.25, 40.75  # exactly a 0.1-degree cell center
    hist = pd.DataFrame({"time": [-1.0], "lat": [src_lat], "lon": [src_lon],
                         "mag": [6.0]})
    idx = int(np.argmin((cells["cell_lon"] - src_lon) ** 2
                        + (cells["cell_lat"] - src_lat) ** 2))
    bg = expected_counts(TRUE, hist.iloc[:0], cells, 0.0, horizon_days=7.0)
    out = expected_counts(TRUE, hist, cells, 0.0, horizon_days=7.0)
    ratio = out[idx] / bg[idx]
    assert ratio > 50.0, f"M6 cell only {ratio:.1f}x background"
    # Omori decay: same history, forecast start moving away from the mainshock
    trig = []
    for t0 in (0.0, 7.0, 30.0):
        o = expected_counts(TRUE, hist, cells, t0, horizon_days=7.0)
        trig.append(float((o - bg).sum()))
    assert trig[0] > trig[1] > trig[2] > 0, f"no Omori decay: {trig}"
    # prob_at_least_one basics
    pr = prob_at_least_one(out)
    assert np.all((pr > 0) & (pr < 1)) and np.allclose(pr, 1 - np.exp(-out))
    assert pr[idx] > prob_at_least_one(bg)[idx]
    print(f"  M6 cell/bg ratio {ratio:.0f}x; triggered sum decays "
          f"{trig[0]:.2f} -> {trig[1]:.2f} -> {trig[2]:.2f}; P(>=1) max {pr.max():.3f}")


def test_5_simulation_gr_and_m6_rate():
    """Pooled simulated magnitudes reproduce b; M>=6 rate matches GR extrapolation."""
    n_sims, years = 15, DURATION / 365.25
    sims = simulate_catalogs(TRUE, DURATION, n_sims, mmax=MMAX, seed=7)
    mags = np.concatenate([s["mag"].to_numpy() for s in sims])
    b_hat = 1.0 / (np.log(10.0) * (mags.mean() - TRUE.mc))  # Aki MLE, continuous mags
    assert abs(b_hat - TRUE.b) <= 0.1, f"b not reproduced: {b_hat:.3f}"
    n_m6 = sum(int((s["mag"] >= 6.0).sum()) for s in sims)
    sim_rate = n_m6 / (n_sims * years)
    n_tr = branching_ratio(TRUE, MMAX)
    total_daily = TRUE.mu_total / (1.0 - n_tr)  # mu-implied total M>=mc rate
    gr_rate = total_daily * 365.25 * 10.0 ** (-TRUE.b * (6.0 - TRUE.mc))
    ratio = sim_rate / gr_rate
    print(f"  pooled b {b_hat:.3f} (true {TRUE.b}); M>=6 rate: simulated "
          f"{sim_rate:.4f}/yr vs GR extrapolation {gr_rate:.4f}/yr "
          f"(ratio {ratio:.2f}, {n_m6} events in {n_sims * years:.0f} sim-years)")
    assert 1.0 / 3.0 < ratio < 3.0, f"M6+ rate off by more than 3x: {ratio:.2f}"
    # simulator output contract
    s0 = sims[0]
    assert list(s0.columns) == ["time", "lat", "lon", "mag", "gen", "parent_id"]
    ok_parent = s0["parent_id"].to_numpy() < np.arange(len(s0))
    assert np.all(ok_parent), "parent must precede child"
    aft = s0[s0["gen"] > 0]
    assert len(aft) > 0 and np.all(s0["mag"] >= TRUE.mc) and s0["mag"].max() <= MMAX
    inreg = ((s0["lon"] >= REGION["min_lon"]) & (s0["lon"] <= REGION["max_lon"])
             & (s0["lat"] >= REGION["min_lat"]) & (s0["lat"] <= REGION["max_lat"]))
    assert inreg.all(), "simulated events left the region"


if __name__ == "__main__":
    tests = [test_1_synthetic_recovery, test_2_real_marmara_fit, test_3_causality,
             test_4_rate_sanity, test_5_simulation_gr_and_m6_rate]
    t_all = time.time()
    failed = 0
    for fn in tests:
        t0 = time.time()
        try:
            fn()
            print(f"PASS {fn.__name__} ({time.time() - t0:.1f}s)")
        except AssertionError as e:
            failed += 1
            print(f"FAIL {fn.__name__} ({time.time() - t0:.1f}s): {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed in {time.time() - t_all:.1f}s")
    sys.exit(1 if failed else 0)
