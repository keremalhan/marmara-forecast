"""T0-2: does the HYBRID pass the CSEP spatial (S) / pseudolikelihood (PL) test?

The paper runs S/PL only on the physics models. The hybrid's whole edge is trimming the
over-predicted background — the exact thing the S-test punishes — so its S/PL verdict is the
missing readout. hybrid_catalogs.npz already exists (native-clustered, ratio-consistent with the
hybrid rate; 500 cats, mean 1041 events/cat vs cascade 1306). Score it (and cascade, for
reference) through the SAME pyCSEP path as the paper. Writes results/round3/hybrid_csep.json.
Run in the CSEP env:  MPLBACKEND=Agg .venv-csep/bin/python scripts/csep/hybrid_csep.py
"""
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))   # .../scripts/csep (holds csep_run.py)
import csep_run as CR  # noqa: E402  (marmara-free; reuses build_region/make_catalog/run_tests)
from csep.core.forecasts import CatalogForecast  # noqa: E402
from csep.utils.time_utils import datetime_to_utc_epoch  # noqa: E402

OUTDIR = Path(__file__).resolve().parents[2] / "results" / "round3"   # repo/results/round3


def score(model, region, observed, t_lo, t_hi, t_epoch):
    d = np.load(CR.INP / f"{model}_catalogs.npz")
    cid = d["catalog_id"]; lon = d["longitude"]; lat = d["latitude"]; mag = d["magnitude"]
    n_sim = int(d["n_sim"])
    bounds = np.searchsorted(cid, np.arange(n_sim + 1))
    catalogs = [CR.make_catalog(j, lon[bounds[j]:bounds[j + 1]], lat[bounds[j]:bounds[j + 1]],
                                mag[bounds[j]:bounds[j + 1]], region, t_epoch) for j in range(n_sim)]
    fc = CatalogForecast(catalogs=catalogs, region=region, name=model, n_cat=n_sim,
                         start_time=t_lo, end_time=t_hi)
    return CR.run_tests(fc, observed)


def main():
    meta = json.load(open(CR.INP / "region.json"))
    region, mags = CR.build_region(meta)
    period = meta["test_period"]
    t_lo = datetime.fromisoformat(period[0]).replace(tzinfo=timezone.utc)
    t_hi = datetime.fromisoformat(period[1]).replace(tzinfo=timezone.utc)
    t_epoch = datetime_to_utc_epoch(t_lo)
    olon, olat, omag = [], [], []
    with open(CR.INP / "observed.csv") as f:
        for row in csv.DictReader(f):
            olon.append(float(row["longitude"])); olat.append(float(row["latitude"])); omag.append(float(row["magnitude"]))
    observed = CR.make_catalog("observed", np.array(olon), np.array(olat), np.array(omag), region, t_epoch)

    out = {"n_observed": len(olon), "test_period": period, "models": {}}
    for model in ("hybrid", "cascade"):
        r = score(model, region, observed, t_lo, t_hi, t_epoch)
        out["models"][model] = r
        print(f"{model:8s} N-pass {r.get('Number',{}).get('pass')} "
              f"M-pass {r.get('Magnitude',{}).get('pass')} "
              f"S {r.get('Spatial',{}).get('gamma')} (pass {r.get('Spatial',{}).get('pass')}) "
              f"PL {r.get('Pseudolikelihood',{}).get('gamma')} (pass {r.get('Pseudolikelihood',{}).get('pass')})")
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "hybrid_csep.json").write_text(json.dumps(out, indent=2))
    print("wrote results/round3/hybrid_csep.json")


if __name__ == "__main__":
    main()
