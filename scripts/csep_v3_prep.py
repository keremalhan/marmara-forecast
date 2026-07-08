"""Phase C prep — generate genuine pyCSEP inputs (event-level catalogs) from the
in-house model rates, replicating csep_eval's EXACT rng sequence so the real-pyCSEP
N/M numbers cross-check the in-house ones.

For each model (cascade, sv_etas, modern_etas) we build N_SIM=1000 stochastic
catalogues: per cell count ~ Poisson(Lambda_cell) (Lambda = summed lam30 over the
test windows), magnitudes ~ Gutenberg-Richter(b) discretised in 0.1 bins (Mc=3.0,
Mmax=7.6). Poisson cell-sampling is drawn with the SAME default_rng(42) sequence as
marmara.csep_eval.run_model, then EXPANDED to event level (events placed at cell
centres, magnitudes at bin centres) — the format genuine pyCSEP consumes.

Observed = the real on-grid M>=3.0 events in the test period.

Writes results/csep_v3/inputs/{region.json, observed.csv, <model>_catalogs.npz}.
The driver scripts/csep_v3_run.py (APFS venv-csep) loads these into pyCSEP
CSEPCatalog/CatalogForecast objects and runs number/magnitude/spatial/PL tests.
Run:  PYTHONPATH=src .venv/bin/python scripts/csep_v3_prep.py
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from marmara import grid as G
from marmara.paths import RESULTS
from marmara import csep_eval as CE

INP = RESULTS / "csep_v3" / "inputs"
DH = 0.1


def cell_center(flat):
    ir, ic = np.divmod(flat, G.NLON)
    return G.LON_EDGE0 + (ic + 0.5) * DH, G.LAT_EDGE0 + (ir + 0.5) * DH


def cell_origin(flat):
    ir, ic = np.divmod(flat, G.NLON)
    return G.LON_EDGE0 + ic * DH, G.LAT_EDGE0 + ir * DH


def main():
    INP.mkdir(parents=True, exist_ok=True)
    grid = pd.read_parquet(RESULTS / "grid_hybrid.parquet")
    test_wins = CE._test_windows(grid)
    lams = CE.model_lambda(grid, test_wins)
    obs_counts, obs_mags, period = CE.observed(grid, test_wins)
    print(f"prep: test period {period[0]}..{period[1]}, {int(obs_counts.sum())} observed "
          f"M>=3.0, {len(test_wins)} windows, models {list(lams)}")

    # region: all NCELLS cell origins in flat order ir*NLON+ic (matches csep_eval flat)
    flats = np.arange(G.NCELLS)
    olon, olat = cell_origin(flats)
    mags = np.round(np.arange(CE.MC, CE.MMAX, CE.DM), 2).tolist()   # bin left-edges 3.0..7.5
    json.dump({"dh": DH, "nlon": int(G.NLON), "nlat": int(G.NLAT), "ncells": int(G.NCELLS),
               "lon_edge0": G.LON_EDGE0, "lat_edge0": G.LAT_EDGE0,
               "origins": np.column_stack([olon, olat]).round(4).tolist(),
               "magnitudes": mags, "mc": CE.MC, "mmax": CE.MMAX, "dm": CE.DM,
               "test_period": period, "n_windows": int(len(test_wins)),
               "n_sim": CE.N_SIM, "seed": CE.SEED,
               "models_b": CE.MODELS_B},
              open(INP / "region.json", "w"), indent=2)

    # observed events (real on-grid M>=3.0): recover per-event lon/lat/mag
    cat = pd.read_csv(RESULTS / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    tw = grid[grid["window"].isin(test_wins)].groupby("window")["t0"].first()
    t0s = pd.to_datetime(tw.values)
    lo = t0s.min(); hi = t0s.max() + pd.Timedelta(days=G.HORIZON_D)
    ev = cat[(cat["datetime_utc"] >= lo) & (cat["datetime_utc"] < hi) & (cat["mag_w"] >= CE.MC - 1e-9)]
    iir, iic = G.cell_index(ev["longitude"].to_numpy(), ev["latitude"].to_numpy())
    on = iir >= 0
    obs = pd.DataFrame({"longitude": ev["longitude"].to_numpy()[on],
                        "latitude": ev["latitude"].to_numpy()[on],
                        "magnitude": ev["mag_w"].to_numpy()[on],
                        "time": ev["datetime_utc"].to_numpy()[on]})
    obs.to_csv(INP / "observed.csv", index=False)
    assert len(obs) == int(obs_counts.sum()), (len(obs), int(obs_counts.sum()))
    print(f"  observed: {len(obs)} events written")

    # magnitude bin machinery — identical to csep_eval._mag_pmf
    medges, _ = CE._mag_pmf(1.2)          # edges independent of b; pmf is not
    mag_centers = medges + CE.DM / 2.0    # bin centre for a sampled index

    # per-model stochastic catalogues — REPLICATE csep_eval.run_model rng sequence
    for model, lam in lams.items():
        b = CE.MODELS_B[model]
        _, mpmf = CE._mag_pmf(b)
        rng = np.random.default_rng(CE.SEED)
        cat_id_list, lon_list, lat_list, mag_list, counts = [], [], [], [], np.empty(CE.N_SIM, int)
        nz = lam > 0
        nz_idx = np.where(nz)[0]
        clon, clat = cell_center(nz_idx)
        for j in range(CE.N_SIM):
            c = rng.poisson(lam)                                   # per-cell counts (full vector, matches in-house)
            n = int(c.sum()); counts[j] = n
            mags_idx = rng.choice(len(mpmf), size=n, p=mpmf) if n else np.empty(0, int)
            if n:
                cnz = c[nz_idx]
                lon_list.append(np.repeat(clon, cnz))
                lat_list.append(np.repeat(clat, cnz))
                mag_list.append(mag_centers[mags_idx])
                cat_id_list.append(np.full(n, j, dtype=np.int32))
        np.savez_compressed(
            INP / f"{model}_catalogs.npz",
            catalog_id=np.concatenate(cat_id_list).astype(np.int32),
            longitude=np.concatenate(lon_list).astype(np.float32),
            latitude=np.concatenate(lat_list).astype(np.float32),
            magnitude=np.concatenate(mag_list).astype(np.float32),
            counts=counts, b=b, n_sim=CE.N_SIM)
        print(f"  {model:12s} N_sim={CE.N_SIM} mean_count={counts.mean():.1f} "
              f"total_events={int(counts.sum())}")

    print("wrote results/csep_v3/inputs/{region.json, observed.csv, <model>_catalogs.npz}")


if __name__ == "__main__":
    main()
