"""V2(b) — GNSS placebo battery. Three placebos, each a permutation/scramble that
preserves some structure but BREAKS the time-alignment with seismicity; a genuine
signal must collapse to IG~0 under all three.

  (i)  time-shuffle    : per cell, permute the gnss feature rows across windows
                         (keeps location + marginal, destroys time-alignment).
  (ii) circular-shift  : per cell, circularly shift the window-ordered features by a
                         random offset >= 2 years (>=24 monthly windows) — preserves
                         temporal autocorrelation, breaks alignment with seismicity.
  (iii) coverage-only  : replace the physical features with a single per-cell
                         station-COVERAGE-vs-time feature (count of stations with data
                         within 60 km at t0). If this alone reproduces the gain, the
                         "signal" is station availability, not deformation.

N_PERM permutations per placebo (retraining the HGB each time). N_PERM is set for
tractability (each retrain ~8 s); it resolves the null cleanly (centred ~0). The real
aug-vs-base test IG must lie OUTSIDE the placebo null for the channel to be genuine.

Writes results/verify/gnss_placebos.json.
"""
import json
import numpy as np
import pandas as pd
from marmara.paths import RESULTS
from marmara.grid import FEATURES, LON_C, LAT_C, NLAT, NLON, HORIZON_D, REF
from marmara.metrics import information_gain, lambda_to_p
from marmara.train import split_masks
from sklearn.ensemble import HistGradientBoostingRegressor

EPS = 1e-9
OUT = RESULTS
VER = OUT / "verify"; VER.mkdir(exist_ok=True)
GNSS4 = ["gnss_resid_rate_90", "gnss_resid_rate_365", "gnss_resid_max_365", "gnss_strain_rate_365"]
BASE = FEATURES + ["ln_lam_sim"]
N_PERM = 30
SEED = 42
SHIFT_MIN_WIN = 24            # >= 2 years at 30-day windows


def load():
    g = pd.read_parquet(OUT / "grid_hybrid.parquet")
    gc = pd.read_parquet(OUT / "gnss_v2_columns.parquet").set_index(["window", "ir", "ic"])
    idx = pd.MultiIndex.from_frame(g[["window", "ir", "ic"]])
    for c in GNSS4:
        g[c] = gc[c].reindex(idx).to_numpy()
    g["ln_lam_sim"] = np.log(g["lam35_sim"].to_numpy() + EPS)
    return g, split_masks(g)


def ig_of(g, m, cols):
    tr, te = m["train"], m["test"]
    r = HistGradientBoostingRegressor(loss="poisson", learning_rate=0.05, max_iter=400,
                                      max_depth=6, random_state=42)
    r.fit(g.loc[tr, cols], g.loc[tr, "count35"].to_numpy())
    P = lambda_to_p(np.clip(r.predict(g[cols]), EPS, None))
    return P, r


def _cellkey(g):
    return g["ir"].to_numpy() * 1000 + g["ic"].to_numpy()


def coverage_feature(g):
    """Per (window,cell): count of GNSS stations with data in [t0-365,t0) within 60 km."""
    from marmara.sources.gnss_v2 import GnssV2Source, KM_PER_DEG_LAT, R_KM, _decimal_year
    src = GnssV2Source()
    ok, why = src.available()
    if not ok:
        return None, why
    series, pos, _ = src._load()
    names = list(series)
    slon = np.array([pos[s][0] for s in names]); slat = np.array([pos[s][1] for s in names])
    cov = np.zeros(len(g))
    for w, sub in g.groupby("window"):
        t0 = pd.Timestamp(sub["t0"].iloc[0]); t0dy = _decimal_year(t0)
        lo = _decimal_year(t0 - pd.Timedelta(days=365))
        active = np.array([(series[s]["yr"] >= lo).sum() > 0 and (series[s]["yr"] < t0dy).sum() >= 5
                           and ((series[s]["yr"] >= lo) & (series[s]["yr"] < t0dy)).sum() >= 5
                           for s in names])
        clon = sub["cell_lon"].to_numpy(); clat = sub["cell_lat"].to_numpy()
        coslat = np.cos(np.radians(float(slat.mean())))
        d = np.sqrt(((clon[:, None] - slon[None, :]) * KM_PER_DEG_LAT * coslat) ** 2
                    + ((clat[:, None] - slat[None, :]) * KM_PER_DEG_LAT) ** 2)
        within = (d <= R_KM) & active[None, :]
        cov[sub.index.to_numpy()] = within.sum(axis=1)
    return cov, ""


def main():
    g, m = load()
    te = m["test"]; y = g["y35"].to_numpy().astype(float)
    Pb, _ = ig_of(g, m, BASE)
    Pa, _ = ig_of(g, m, BASE + GNSS4)
    ig_real = float(information_gain(Pa[te], Pb[te], y[te]))
    print(f"real aug-vs-base test IG = {ig_real:+.4f}")

    ck = _cellkey(g); wins = g["window"].to_numpy()
    order_within = {k: np.where(ck == k)[0][np.argsort(wins[ck == k])] for k in np.unique(ck)}
    prng = np.random.default_rng(SEED)

    def run_placebo(kind):
        igs = []
        for _ in range(N_PERM):
            gp = g.copy()
            arr = {c: g[c].to_numpy().copy() for c in GNSS4}
            for k, idx in order_within.items():
                nrow = len(idx)
                if kind == "shuffle":
                    perm = prng.permutation(nrow)
                else:  # circular shift
                    off = int(prng.integers(SHIFT_MIN_WIN, max(SHIFT_MIN_WIN + 1, nrow)))
                    perm = (np.arange(nrow) + off) % nrow
                for c in GNSS4:
                    arr[c][idx] = arr[c][idx][perm]
            for c in GNSS4:
                gp[c] = arr[c]
            Pp, _ = ig_of(gp, m, BASE + GNSS4)
            igs.append(float(information_gain(Pp[te], Pb[te], y[te])))
        igs = np.array(igs)
        return {"n_perm": N_PERM, "mean_ig": round(float(igs.mean()), 5),
                "p95_range": [round(float(np.percentile(igs, 2.5)), 5), round(float(np.percentile(igs, 97.5)), 5)],
                "real_ig_exceeds_all": bool(ig_real > igs.max()),
                "collapsed": bool(abs(igs.mean()) < 0.5 * abs(ig_real) and ig_real > np.percentile(igs, 97.5))}

    out = {"real_aug_vs_base_test_ig": round(ig_real, 5), "n_perm": N_PERM,
           "note": "N_PERM per placebo (retrain each); resolves the null centred ~0."}
    print("time-shuffle ..."); out["time_shuffle"] = run_placebo("shuffle")
    print("circular-shift ..."); out["circular_shift"] = run_placebo("shift")

    cov, why = coverage_feature(g)
    if cov is None:
        out["coverage_only"] = {"skipped": why}
    else:
        gc = g.copy(); gc["gnss_coverage"] = cov
        Pc, _ = ig_of(gc, m, BASE + ["gnss_coverage"])
        ig_cov = float(information_gain(Pc[te], Pb[te], y[te]))
        out["coverage_only"] = {"test_ig": round(ig_cov, 5),
                                "reproduces_gain": bool(ig_cov > 0.5 * ig_real),
                                "verdict": "coverage reproduces gain -> ARTIFACT" if ig_cov > 0.5 * ig_real
                                else "coverage does NOT reproduce gain (OK)"}
    out["verdict"] = ("GENUINE — collapses under time-shuffle & circular-shift and is not "
                      "reproduced by coverage") if (out["time_shuffle"]["collapsed"] and
                      out["circular_shift"]["collapsed"] and
                      not out.get("coverage_only", {}).get("reproduces_gain", False)) else \
                     "FLAG — a placebo did not behave as required; inspect"
    json.dump(out, open(VER / "gnss_placebos.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
