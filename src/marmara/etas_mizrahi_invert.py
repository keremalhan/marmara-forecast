"""Phase 1.1 — invert the INDEPENDENT third-party lmizrahi/etas 3.0.0 model on the
Marmara catalogue and write results/etas_mizrahi_fit.json.

RUNS UNDER .venv-etas (NOT the pinned core env) — it imports `etas`, which needs
numpy>=2.5 / pandas>=3.0 (see requirements-etas.txt). It deliberately does NOT
import `marmara` (incompatible pinned deps). Same catalogue slice as etas_fit.py:
events < 2021-12-31, Mc=3.0, on the model-box polygon.

  COPYFILE_DISABLE=1 UV_LINK_MODE=copy VIRTUAL_ENV="$PWD/.venv-etas" \
      uv pip install -r requirements-etas.txt
  MARMARA_ROOT="$PWD" .venv-etas/bin/python -m marmara.etas_mizrahi_invert
      # or: .venv-etas/bin/python src/marmara/etas_mizrahi_invert.py

The fitted params feed marmara.etas_modern (main env), which integrates the
Mizrahi kernels into the gridded `modern_etas` forecast. The fit is a MLE + EM
declustering; re-runs may differ at the 3rd decimal (optimizer tolerance). The
committed JSON is from git commit 097f08b of the package.
"""
import json
import os

import numpy as np
import pandas as pd

REPO = os.environ.get("MARMARA_ROOT", "/Volumes/Kerem SSD/Desktop/marmara-forecast")
FIT_END = pd.Timestamp("2021-12-31")
MC = 3.0
# model box (etas_fit.MODEL_BOX) as (lat, lon) corners — the OEF entrypoint feeds
# shape_coords in (lat, lon) order (it swaps WKT lon/lat).
SHAPE_LATLON = [[39.6, 25.6], [39.6, 30.9], [41.9, 30.9], [41.9, 25.6], [39.6, 25.6]]


def main():
    from etas.inversion import (ETASParameterCalculation, estimate_beta_positive)

    cat = pd.read_csv(f"{REPO}/results/catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    cat = cat[(cat["datetime_utc"] < FIT_END) & (cat["mag_w"] >= MC - 1e-9)].copy()
    df = pd.DataFrame({"latitude": cat["latitude"].to_numpy(),
                       "longitude": cat["longitude"].to_numpy(),
                       "magnitude": cat["mag_w"].to_numpy(),
                       "time": cat["datetime_utc"].values})
    print(f"catalog: {len(df)} events >= {MC}, {df['time'].min()} .. {df['time'].max()}", flush=True)

    md = {"name": "marmara_mizrahi", "catalog": df, "shape_coords": SHAPE_LATLON,
          "delta_m": 0.1, "mc": MC, "m_ref": MC, "coppersmith_multiplier": 100,
          "auxiliary_start": df["time"].min(), "timewindow_start": df["time"].min(),
          "timewindow_end": FIT_END, "b_positive": True, "n_iterations": 6}
    calc = ETASParameterCalculation(md)
    calc.prepare()
    calc.invert()
    theta = dict(calc.theta)          # {log10_mu, log10_k0, a, log10_c, omega, log10_tau, log10_d, gamma, rho, ...}
    print("theta:", theta, flush=True)

    beta = float(estimate_beta_positive(df["magnitude"].to_numpy(), delta_m=0.1))

    def g(key):
        v = theta.get(key)
        return None if v is None else float(v)

    out = {
        "source": "lmizrahi/etas 3.0.0 inversion (independent third-party fit)",
        "fit_end": str(FIT_END.date()), "mc": MC, "n_events_fit": int(len(df)),
        "coppersmith_multiplier": 100, "n_iterations": 6,
        "param_dict": {k: g(k) for k in ("log10_mu", "log10_iota", "log10_k0", "a",
                                         "log10_c", "omega", "log10_tau", "log10_d",
                                         "gamma", "rho")},
        "beta": beta, "b_value_implied": beta / np.log(10.0),
        "derived": {
            "mu_per_day_per_km2": 10.0 ** g("log10_mu"),
            "c_days": 10.0 ** g("log10_c"),
            "p_effective_1_plus_omega": 1.0 + g("omega"),
            "tau_days": 10.0 ** g("log10_tau"),
            "d_spatial": 10.0 ** g("log10_d"),
            "rho_spatial_q": g("rho"), "gamma": g("gamma"),
        },
        "note": ("Best fit is a TAPERED Omori with p=1+omega<1 and finite tau; "
                 "structurally incompatible with our pure-Omori cascade. modern_etas "
                 "is therefore evaluated via first-generation INTENSITY INTEGRATION of "
                 "these kernels (marmara.etas_modern), comparable to firstgen_etas."),
    }
    with open(f"{REPO}/results/etas_mizrahi_fit.json", "w") as f:
        json.dump(out, f, indent=2)
    print("WROTE results/etas_mizrahi_fit.json", flush=True)
    print(json.dumps(out["derived"], indent=2))


if __name__ == "__main__":
    main()
