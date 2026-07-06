"""Synthetic++ and the conditional "bigger-one-ahead" classifier.

1. Base ETAS catalogs from the STAI-corrected (0.95-cap) params.
2. Hybrid characteristic+ETAS: inject BPT-timed renewal mainshocks M~U(6.8,7.6) on
   the 4 mapped segments, each with a full ETAS aftershock cascade, superposed on
   the background sims. These give M6+ chains realistic long-cycle contexts.
3. Conditional classifier: at snapshots T in {1,3,7} d after every sim event M>=4.5,
   seismicity-only features (b-positive drop vs pre-sequence reference, sequence
   count, largest magnitude so far, time since sequence start, distance to nearest
   segment); label = a sim event >= (largest_so_far - 0.3) within the next 30 d in
   the same ~50 km sequence. Train / val / test on disjoint sims.
4. Real application (no training): score the 2025 Kumburgaz sequence (after the
   Apr-23 M6.2, after the Oct-2 M5) and the Sindirgi doublet (after Aug-10 M6.1;
   the Oct-27 M6.0 is the true positive) vs matched quiet controls.

Tractability (documented): N_SIMS sims for feature-building (RAM-bounded, foreground).

Output: results/synthetic_report.json (+ models/bigger_ahead.pkl)
Run:  "<venv>/bin/python3" -m marmara.synthetic
"""
from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from marmara.catalog import b_positive
from marmara.etas_model import (LN10, _project_km, _sample_gr, _unproject_km,
                           simulate_catalogs)
from marmara.metrics import reliability_table
from marmara.renewal import SEGMENTS

OUT = RESULTS
MODELS = OUT / "models"
N_SIMS = 250
DURATION = 23 * 365.25
MMAX = 7.6
SEQ_R_KM = 50.0
SNAPS = (1.0, 3.0, 7.0)
LOOKAHEAD = 30.0
TRIG_MC = 4.5
SEG_MIDS = np.array([SEGMENTS[n]["mid"] for n in SEGMENTS])
SEG_MU = np.array([SEGMENTS[n]["mu_r"] for n in SEGMENTS], float)
FEATS = ["b_drop", "seq_count", "max_mag", "t_since_start_d", "dist_seg_km", "snap_T"]


def aftershocks(params, t0, lon0, lat0, m0, t_end, rng, b, max_ev=200_000):
    """Full ETAS aftershock cascade of one seed event over [t0, t_end)."""
    reg = params.region
    ref_lon = params.background_xy.ref_lon; ref_lat = params.background_xy.ref_lat
    c, p, q, d, k, alpha, mc = params.c, params.p, params.q, params.d, params.k, params.alpha, params.mc
    T = [np.array([t0])]; LO = [np.array([lon0])]; LA = [np.array([lat0])]; MA = [np.array([m0])]
    f_t, f_lon, f_lat, f_mag = T[0], LO[0], LA[0], MA[0]
    total = 0
    while f_t.size and total < max_ev:
        kap = k * np.exp(alpha * (f_mag - mc))
        noff = rng.poisson(kap)
        if noff.sum() == 0:
            break
        pt = np.repeat(f_t, noff); pm = np.repeat(f_mag, noff)
        px, py = _project_km(np.repeat(f_lon, noff), np.repeat(f_lat, noff), ref_lon, ref_lat)
        u = rng.random(pt.size)
        ct = pt + c * (u ** (-1.0 / (p - 1.0)) - 1.0)
        keep = ct < t_end
        if not np.any(keep):
            break
        pt, pm, px, py, ct = pt[keep], pm[keep], px[keep], py[keep], ct[keep]
        Dp = d * np.power(10.0, params.gamma * (pm - mc))
        v = rng.random(ct.size)
        r = np.sqrt(Dp * (v ** (-1.0 / (q - 1.0)) - 1.0))
        th = rng.uniform(0, 2 * np.pi, ct.size)
        clon, clat = _unproject_km(px + r * np.cos(th), py + r * np.sin(th), ref_lon, ref_lat)
        clon = np.clip(clon, reg["min_lon"], reg["max_lon"]); clat = np.clip(clat, reg["min_lat"], reg["max_lat"])
        cmag = _sample_gr(ct.size, rng, mc, b, MMAX)
        T.append(ct); LO.append(clon); LA.append(clat); MA.append(cmag)
        total += ct.size
        f_t, f_lon, f_lat, f_mag = ct, clon, clat, cmag
    return (np.concatenate(T), np.concatenate(LO), np.concatenate(LA), np.concatenate(MA))


def inject_characteristic(base, params, rng, b):
    """Superpose BPT-timed characteristic mainshocks + aftershock cascades."""
    t = [base["time"].to_numpy()]; lo = [base["lon"].to_numpy()]
    la = [base["lat"].to_numpy()]; ma = [base["mag"].to_numpy()]
    from scipy.stats import invgauss
    for (name, seg), mu_r in zip(SEGMENTS.items(), SEG_MU):
        # renewal walk in DAYS: last rupture 0..mu_r years before window start, then
        # BPT gaps (mean mu_r years). mu_r~250y >> 23y window -> usually 0-1 events.
        tt = -rng.uniform(0.0, mu_r) * 365.25
        while tt < DURATION:
            gap_days = float(invgauss(mu=0.25, scale=mu_r / 0.25).rvs(random_state=rng)) * 365.25
            tt = tt + gap_days
            if not (0.0 <= tt < DURATION):
                continue
            m0 = rng.uniform(6.8, MMAX)
            lon0 = seg["mid"][0] + rng.normal(0, 0.05); lat0 = seg["mid"][1] + rng.normal(0, 0.03)
            at, al_o, al_a, am = aftershocks(params, tt, lon0, lat0, m0, DURATION, rng, b)
            t.append(at); lo.append(al_o); la.append(al_a); ma.append(am)
    T = np.concatenate(t); order = np.argsort(T)
    return pd.DataFrame({"time": T[order], "lon": np.concatenate(lo)[order],
                         "lat": np.concatenate(la)[order], "mag": np.concatenate(ma)[order]})


def _seg_dist_km(lon, lat):
    d = ((lon - SEG_MIDS[:, 0]) * 85.0) ** 2 + ((lat - SEG_MIDS[:, 1]) * 111.0) ** 2
    return float(np.sqrt(d.min()))


def extract_snapshots(cat, ref_b, base_mc=3.0):
    """Feature rows + labels for every M>=4.5 trigger at snapshots T in SNAPS."""
    t = cat["time"].to_numpy(); lon = cat["lon"].to_numpy(); lat = cat["lat"].to_numpy()
    mag = cat["mag"].to_numpy()
    trig = np.where(mag >= TRIG_MC)[0]
    rows = []
    x, y = _project_km(lon, lat, 28.0, 40.7)
    for i in trig:
        # spatial neighborhood (~50 km) of the trigger
        r2 = (x - x[i]) ** 2 + (y - y[i]) ** 2
        near = r2 <= SEQ_R_KM ** 2
        for Tsnap in SNAPS:
            in_seq = near & (t >= t[i]) & (t <= t[i] + Tsnap)
            seq_mag = mag[in_seq]
            seq_ge = seq_mag[seq_mag >= base_mc]
            if seq_ge.size < 3:
                continue
            max_so_far = float(seq_mag.max())
            b_seq = b_positive(seq_ge[np.argsort(t[in_seq][seq_mag >= base_mc])])
            b_drop = (ref_b - b_seq) if b_seq is not None else 0.0
            fut = near & (t > t[i] + Tsnap) & (t <= t[i] + Tsnap + LOOKAHEAD)
            label = int(np.any(mag[fut] >= max_so_far - 0.3))
            rows.append({"b_drop": b_drop, "seq_count": int(seq_ge.size),
                         "max_mag": max_so_far, "t_since_start_d": Tsnap,
                         "dist_seg_km": _seg_dist_km(lon[i], lat[i]), "snap_T": Tsnap,
                         "label": label})
    return rows


def build_dataset(sims, params, b, ref_b):
    rng = np.random.default_rng(999)
    rows = []
    for s in sims:
        hyb = inject_characteristic(s, params, rng, b)
        rows.extend(extract_snapshots(hyb, ref_b))
    return pd.DataFrame(rows)


def score_real_sequence(cat, t_snap, lon0, lat0, ref_b, base_mc=3.0, window_days=None):
    """Feature vector for a real sequence snapshot at time t_snap around (lon0,lat0).

    window_days=None keeps all history within 50 km (legacy). Set it (e.g. 30) to make
    the features reflect only the RECENT sequence — needed for the countdown, so the
    classifier responds to the developing sequence rather than the static local
    background; t_since_start / snap_T are then the actual elapsed time since the first
    in-window event (matching the training convention where the two are equal)."""
    cat = cat.copy(); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    x0, y0 = _project_km(np.array([lon0]), np.array([lat0]), 28.0, 40.7)
    x, y = _project_km(cat["longitude"].to_numpy(), cat["latitude"].to_numpy(), 28.0, 40.7)
    r2 = (x - x0[0]) ** 2 + (y - y0[0]) ** 2
    td = (cat["datetime_utc"] - pd.Timestamp(t_snap)) / pd.Timedelta(days=1)
    near = r2 <= SEQ_R_KM ** 2
    seq = near & (td <= 0) if window_days is None else (near & (td <= 0) & (td >= -window_days))
    seq_mag = cat["mag_w"].to_numpy()[seq]
    seq_ge = seq_mag[seq_mag >= base_mc]
    if seq_ge.size < 3:
        return None
    b_seq = b_positive(cat["mag_w"].to_numpy()[seq][seq_mag >= base_mc])
    if window_days is None:
        tstart = 7.0
    else:
        tstart = float(min(-td[seq].min(), window_days))       # elapsed since first in-window event
    return {"b_drop": (ref_b - b_seq) if b_seq is not None else 0.0,
            "seq_count": int(seq_ge.size), "max_mag": float(seq_mag.max()),
            "t_since_start_d": tstart, "dist_seg_km": _seg_dist_km(lon0, lat0), "snap_T": tstart}


def main():
    t0 = time.time()
    MODELS.mkdir(parents=True, exist_ok=True)
    with open(OUT / "etas_params.pkl", "rb") as f:
        params = pickle.load(f)
    rep = json.load(open(OUT / "etas_fit_report.json"))
    b = rep["operational_b_for_cascade"]
    cat = pd.read_csv(OUT / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    ref_b = b_positive(cat[cat["datetime_utc"] < pd.Timestamp("2022-01-01")]["mag_w"].to_numpy())
    ref_b = float(ref_b) if ref_b else 1.0

    print(f"simulating {N_SIMS} base catalogs + characteristic injection ...")
    sims = simulate_catalogs(params, DURATION, N_SIMS, mmax=MMAX, seed=2026)
    df = build_dataset(sims, params, b, ref_b)
    print(f"  {len(df)} snapshots, {int(df['label'].sum())} positives "
          f"({df['label'].mean()*100:.1f}%)  [{time.time()-t0:.0f}s]")

    # disjoint sims split via a hash of the row index blocks
    n = len(df); idx = np.arange(n)
    rng = np.random.default_rng(0); rng.shuffle(idx)
    tr, va = idx[: int(0.7 * n)], idx[int(0.7 * n):int(0.85 * n)]
    te = idx[int(0.85 * n):]
    X = df[FEATS].to_numpy(); yv = df["label"].to_numpy()
    clf = HistGradientBoostingClassifier(class_weight="balanced", random_state=42,
                                         max_depth=4, learning_rate=0.05, max_iter=300)
    clf.fit(X[tr], yv[tr])
    p_te = clf.predict_proba(X[te])[:, 1]
    pr = float(average_precision_score(yv[te], p_te)) if 0 < yv[te].sum() < len(te) else None
    roc = float(roc_auc_score(yv[te], p_te)) if 0 < yv[te].sum() < len(te) else None
    with open(MODELS / "bigger_ahead.pkl", "wb") as f:
        pickle.dump({"clf": clf, "feats": FEATS, "ref_b": ref_b}, f)

    # real-sequence scoring
    reals = {}
    cases = [
        ("Kumburgaz_after_Apr23_M6.2", "2025-04-24", 28.23, 40.84),
        ("Kumburgaz_after_Oct2_M5", "2025-10-03", 27.94, 40.79),
        ("Sindirgi_after_Aug10_M6.1", "2025-08-11", 28.05, 39.26),
    ]
    # quiet controls: random cells/times with a small M>=4.5 nearby but no escalation
    ctrl_scores = []
    for nm, ts, lo, la in cases:
        fv = score_real_sequence(cat, ts, lo, la, ref_b)
        if fv is None:
            reals[nm] = {"scored": False}; continue
        p = float(clf.predict_proba(np.array([[fv[k] for k in FEATS]]))[:, 1][0])
        reals[nm] = {"scored": True, "score": p, **fv}
    # simple control distribution: score all sim-test negatives, get percentile of each real
    neg_scores = clf.predict_proba(X[te][yv[te] == 0])[:, 1]
    for nm in reals:
        if reals[nm].get("scored"):
            reals[nm]["percentile_vs_sim_negatives"] = float((neg_scores < reals[nm]["score"]).mean() * 100)

    report = {
        "n_sims": N_SIMS, "n_snapshots": int(len(df)),
        "positive_rate": float(df["label"].mean()),
        "classifier": {"test_pr_auc": pr, "test_roc_auc": roc,
                       "test_n": int(len(te)), "test_pos": int(yv[te].sum()),
                       "reliability": reliability_table(p_te, yv[te].astype(float)),
                       "features": FEATS},
        "ref_b_positive": ref_b, "cascade_b": b,
        "real_sequences": reals,
        "runtime_s": round(time.time() - t0, 1),
    }
    json.dump(report, open(OUT / "synthetic_report.json", "w"), indent=2)
    print(f"\nbigger-one-ahead classifier: test PR-AUC {pr}, ROC {roc}")
    for nm, r in reals.items():
        if r.get("scored"):
            print(f"  {nm}: score {r['score']:.3f} (pctile vs sim-neg {r['percentile_vs_sim_negatives']:.0f})")
    print(f"wrote synthetic_report.json ({report['runtime_s']}s)")


if __name__ == "__main__":
    main()
