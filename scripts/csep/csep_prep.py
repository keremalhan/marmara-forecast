"""Generate pyCSEP inputs (event-level catalogs) for the catalog-based consistency tests.

v2 (task #6): cascade and sv-ETAS now use NATIVE CLUSTERED catalogues -- each stochastic
catalogue is one realization of the cascade simulator (background + real-history offspring +
recursive in-window triggering), so it reproduces the within-window CLUSTERING of real
seismicity. This is the correct input for the catalog-based S (spatial) and PL tests, which
the previous Poisson (cell-independent) draws under-dispersed and auto-rejected. Magnitudes
come from the cascade's own Gutenberg-Richter draw at the operational b_op (not the ad-hoc
fixed b=1.2). modern_etas is a FIRST-GENERATION intensity with no clustered simulator, so it
keeps the Poisson draw (disclosed as such).

Observed = the real on-grid M>=3.0 events in the test period.

Writes results/csep/inputs/{region.json, observed.csv, <model>_catalogs.npz}.
The driver scripts/csep/csep_run.py loads these into pyCSEP objects and runs the N/M/S/PL tests.
Run:  PYTHONPATH=src .venv/bin/python scripts/csep/csep_prep.py
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd

from marmara import grid as G
from marmara.paths import RESULTS
from marmara import csep_eval as CE
from marmara.cascade import cascade_forecast

INP = RESULTS / "csep" / "inputs"
DH = 0.1
N_CAT_NATIVE = 500                        # cascade realizations for the clustered catalogues
NATIVE_MODELS = {"cascade": "etas_params.pkl", "sv_etas": "etas_sv_params.pkl"}
SEED_BASE = 5000


def cell_center(flat):
    ir, ic = np.divmod(flat, G.NLON)
    return G.LON_EDGE0 + (ic + 0.5) * DH, G.LAT_EDGE0 + (ir + 0.5) * DH


def cell_origin(flat):
    ir, ic = np.divmod(flat, G.NLON)
    return G.LON_EDGE0 + ic * DH, G.LAT_EDGE0 + ir * DH


def native_catalogs(params, b, n_cat, test_wins, grid, cat, spec):
    """N_cat native clustered catalogues from the cascade simulator over the test windows.
    Realization j = the j-th cascade sim in every window, concatenated over the test period."""
    hist_all = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]
    win_t0 = grid.groupby("window")["t0"].first()
    S, LO, LA, MA = [], [], [], []
    for w in test_wins:
        w = int(w); t0_dt = pd.Timestamp(win_t0.loc[w]); t0d = float(G._to_days(t0_dt))
        ev = cascade_forecast(params, hist_all[cat["datetime_utc"] < t0_dt], t0d, G.HORIZON_D,
                              spec.lon_c, spec.lat_c, K=n_cat, seed=SEED_BASE + w, b=b,
                              return_events=True)          # canonical surgical cascade (preserve_branching default)
        m = ev["mag"] >= CE.MC - 1e-9
        S.append(ev["sim"][m]); LO.append(ev["lon"][m]); LA.append(ev["lat"][m]); MA.append(ev["mag"][m])
    sim = np.concatenate(S); lon = np.concatenate(LO); lat = np.concatenate(LA); mag = np.concatenate(MA)
    order = np.argsort(sim, kind="stable")                 # csep_run splits per catalog_id (must be sorted)
    return (sim[order].astype(np.int32), lon[order].astype(np.float32),
            lat[order].astype(np.float32), mag[order].astype(np.float32),
            np.bincount(sim, minlength=n_cat))


def poisson_catalogs(lam, b, n_sim, seed):
    """Cell-independent Poisson catalogues (first-gen intensity; appropriate for modern_etas)."""
    _, mpmf = CE._mag_pmf(b)
    medges, _ = CE._mag_pmf(1.2); mag_centers = medges + CE.DM / 2.0
    rng = np.random.default_rng(seed)
    nz_idx = np.where(lam > 0)[0]; clon, clat = cell_center(nz_idx)
    S, LO, LA, MA = [], [], [], []; counts = np.empty(n_sim, int)
    for j in range(n_sim):
        c = rng.poisson(lam); n = int(c.sum()); counts[j] = n
        if n:
            cnz = c[nz_idx]
            LO.append(np.repeat(clon, cnz)); LA.append(np.repeat(clat, cnz))
            MA.append(mag_centers[rng.choice(len(mpmf), size=n, p=mpmf)])
            S.append(np.full(n, j, dtype=np.int32))
    return (np.concatenate(S).astype(np.int32), np.concatenate(LO).astype(np.float32),
            np.concatenate(LA).astype(np.float32), np.concatenate(MA).astype(np.float32), counts)


def main():
    INP.mkdir(parents=True, exist_ok=True)
    grid = pd.read_parquet(RESULTS / "grid" / "grid_hybrid.parquet")
    test_wins = CE._test_windows(grid)
    lams = CE.model_lambda(grid, test_wins)
    obs_counts, obs_mags, period = CE.observed(grid, test_wins)
    b_op = float(json.load(open(RESULTS / "etas" / "etas_fit_report.json"))["operational_b_for_cascade"])
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    spec = G.MODEL_SPEC
    print(f"prep: test period {period[0]}..{period[1]}, {int(obs_counts.sum())} observed M>=3.0, "
          f"{len(test_wins)} windows, b_op={b_op}, models {list(lams)}")

    flats = np.arange(G.NCELLS); olon, olat = cell_origin(flats)
    mags = np.round(np.arange(CE.MC, CE.MMAX, CE.DM), 2).tolist()
    json.dump({"dh": DH, "nlon": int(G.NLON), "nlat": int(G.NLAT), "ncells": int(G.NCELLS),
               "lon_edge0": G.LON_EDGE0, "lat_edge0": G.LAT_EDGE0,
               "origins": np.column_stack([olon, olat]).round(4).tolist(),
               "magnitudes": mags, "mc": CE.MC, "mmax": CE.MMAX, "dm": CE.DM,
               "test_period": period, "n_windows": int(len(test_wins)),
               "n_sim": N_CAT_NATIVE, "seed": SEED_BASE, "b_op": b_op,
               "catalogue_kind": {"cascade": "native-clustered", "sv_etas": "native-clustered",
                                  "modern_etas": "poisson-first-gen"}},
              open(INP / "region.json", "w"), indent=2)

    # observed events (real on-grid M>=3.0)
    tw = grid[grid["window"].isin(test_wins)].groupby("window")["t0"].first()
    t0s = pd.to_datetime(tw.values); lo = t0s.min(); hi = t0s.max() + pd.Timedelta(days=G.HORIZON_D)
    ev = cat[(cat["datetime_utc"] >= lo) & (cat["datetime_utc"] < hi) & (cat["mag_w"] >= CE.MC - 1e-9)]
    iir, iic = G.cell_index(ev["longitude"].to_numpy(), ev["latitude"].to_numpy()); on = iir >= 0
    pd.DataFrame({"longitude": ev["longitude"].to_numpy()[on], "latitude": ev["latitude"].to_numpy()[on],
                  "magnitude": ev["mag_w"].to_numpy()[on], "time": ev["datetime_utc"].to_numpy()[on]}
                 ).to_csv(INP / "observed.csv", index=False)
    print(f"  observed: {int(on.sum())} events written")

    for model, lam in lams.items():
        if model in NATIVE_MODELS:
            params = pickle.load(open(RESULTS / NATIVE_MODELS[model], "rb"))
            sim, lon, lat, mag, counts = native_catalogs(params, b_op, N_CAT_NATIVE, test_wins, grid, cat, spec)
            b_used, kind = b_op, "native-clustered"
        else:
            sim, lon, lat, mag, counts = poisson_catalogs(lam, CE.MODELS_B[model], CE.N_SIM, CE.SEED)
            b_used, kind = CE.MODELS_B[model], "poisson-first-gen"
        np.savez_compressed(INP / f"{model}_catalogs.npz", catalog_id=sim, longitude=lon,
                            latitude=lat, magnitude=mag, counts=counts, b=b_used,
                            n_sim=len(counts), kind=kind)
        print(f"  {model:12s} [{kind:18s}] n_sim={len(counts)} mean_count={counts.mean():.1f} "
              f"total_events={int(counts.sum())}")
    print("wrote results/csep/inputs/{region.json, observed.csv, <model>_catalogs.npz}")


if __name__ == "__main__":
    main()
