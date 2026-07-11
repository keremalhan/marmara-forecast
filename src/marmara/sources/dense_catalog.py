"""Source 3, the dense sub-Mc3 catalogue (Mc -> ~1.5): the single biggest data lever.

The KOERI catalog completes near Mc~3. A dense, template-matched / ML-picked catalog
(Mc~1.5) carries ~10-30x more events, giving far richer rate, b-value and clustering
signal, the microseismicity that tracks a nucleating sequence before it reaches M3.5.

Reliable/safe/easy data (external fetch script, sources/fetch_data.py dense): a published
enhanced Marmara catalog. Candidates: deep-learning re-picked catalogs on Zenodo
(EQTransformer/PhaseNet studies of the Marmara/North Anatolian Fault) or the AFAD/KOERI
extended bulletin. Drop a CSV `marmara_dense_catalog.csv`
[datetime_utc, latitude, longitude, mag] into Raw_Data/external/.

Features (recomputed at DENSE_MC on the dense catalog, strictly events < t0):
  dense_cnt30_mc15, dense_cnt365_mc15   -- cell counts at Mc=1.5
  dense_bpos_50km_730d                  -- b-positive within ~50 km, last 2 yr
The columns mirror the existing seismicity features so their marginal IG over the hybrid
grid is exactly what the harness measures.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401

from marmara.catalog import b_positive
from marmara.grid import MODEL_SPEC, REF, build_bundle, cell_index_spec
from .base import DATA, FeatureSource

PATH = DATA / "marmara_dense_catalog.csv"
DENSE_MC = 1.5


class DenseCatalogSource(FeatureSource):
    name = "dense_catalog"
    columns = ["dense_cnt30_mc15", "dense_cnt365_mc15", "dense_bpos_50km_730d"]
    data_spec = {
        "requires": str(PATH),
        "fetch": "sources/fetch_data.py dense  (published DL/template catalog, Zenodo)",
        "format": "CSV [datetime_utc, latitude, longitude, mag], Mc~1.5",
        "provenance": "Deep-learning re-picked Marmara catalogs (EQTransformer/PhaseNet "
                      "studies) or AFAD/KOERI extended bulletin.",
        "why": "Microseismicity below Mc3 is the single biggest missing data lever; it "
               "tracks nucleation before events reach the M3.5 target.",
    }

    def available(self):
        return (PATH.exists(), "" if PATH.exists() else f"missing {PATH}")

    def add_columns(self, cells: pd.DataFrame) -> pd.DataFrame:
        d = pd.read_csv(PATH)
        d["datetime_utc"] = pd.to_datetime(d["datetime_utc"])
        spec = MODEL_SPEC
        td = ((d["datetime_utc"] - REF) / pd.Timedelta(days=1)).to_numpy()
        EV = build_bundle(td, d["longitude"].to_numpy(), d["latitude"].to_numpy(),
                          d["mag"].to_numpy(), np.full(len(d), 10.0), spec, DENSE_MC)
        mc = EV["mc"]; tmc = mc["t"]
        out = {c: np.zeros(len(cells)) for c in self.columns}
        for w, idx in cells.groupby("window").groups.items():
            p0 = cells.index.get_loc(idx[0])
            t0d = float((pd.Timestamp(cells["t0"].iloc[p0]) - REF) / pd.Timedelta(days=1))
            hi = np.searchsorted(tmc, t0d, "left")
            for D, col in ((30.0, "dense_cnt30_mc15"), (365.0, "dense_cnt365_mc15")):
                lo = np.searchsorted(tmc, t0d - D, "left")
                g = np.bincount(mc["ir"][lo:hi] * spec.nlon + mc["ic"][lo:hi],
                                minlength=spec.ncells).astype(float)
                for r_ in idx:
                    p = cells.index.get_loc(r_)
                    out[col][p] = g[int(cells["ir"].iloc[p]) * spec.nlon + int(cells["ic"].iloc[p])]
            # b-positive within ~50 km (9x9) & last 730 d
            lo730 = np.searchsorted(tmc, t0d - 730.0, "left")
            ir7, ic7, mg7 = mc["ir"][lo730:hi], mc["ic"][lo730:hi], mc["mag"][lo730:hi]
            reg_b = b_positive(mg7) if len(mg7) else None
            for r_ in idx:
                p = cells.index.get_loc(r_)
                i, j = int(cells["ir"].iloc[p]), int(cells["ic"].iloc[p])
                m = (np.abs(ir7 - i) <= 4) & (np.abs(ic7 - j) <= 4)
                b = b_positive(mg7[m]) if m.sum() >= 31 else None
                out["dense_bpos_50km_730d"][p] = b if b is not None else (reg_b or 1.0)
        return pd.DataFrame(out, index=cells.index)
