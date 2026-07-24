"""sv-ETAS: our ETAS refit with a CONVERGED spatially-variable
background mu(x,y) via EM stochastic declustering (Zhuang 2002).

FRAMING NOTE (verify before citing). The first-generation fit (etas_fit.py)
ALREADY estimates a spatially-variable, probabilistically-declustered weighted-KDE
background (etas_model.BackgroundField; the published fit runs bg_iterations=2, i.e.
one reweighting pass). So sv-ETAS does NOT introduce a variable mu where the baseline
had a constant one; the design.s "constant mu -> mu(x,y)" premise does not hold
for this codebase. What sv-ETAS isolates instead is the effect of running the
declustering to CONVERGENCE (<=10 EM iterations, or dLL < 0.1) with a Silverman
bandwidth computed on the background-weighted points and floored at 5 km (design
spec), versus the published 2-stage background with a 1.5 km floor.

EVERYTHING ELSE IS IDENTICAL to the published fit and reuses the protected
etas_model likelihood unmodified: STAI time-varying completeness Mc(t), the b-value
handling (b-positive), the branching cap (0.95, FALLBACK_CAP), FIT_END 2021-12-31,
the same pair truncation and inits. Only the background estimator changes, so the
sv-ETAS vs first-gen comparison attributes any difference to the background alone.

Does NOT overwrite etas_fit.py (a published result). Output:
  results/etas/etas_sv_params.pkl , results/etas/etas_sv_fit_report.json
Run:  "<venv>/bin/python" -m marmara.etas_fit_sv
"""
from __future__ import annotations

import json
import pickle

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401

from marmara.catalog import b_positive
from marmara.etas_fit import BASE_MC, FIT_END, MODEL_BOX, FALLBACK_CAP, mc_of_time, _fit_stage
from marmara.etas_model import (BackgroundField, EtasParams, LN10, _aki_b, _build_pairs,
                                _project_km, _region_center, _rho_background,
                                _theta_pack, _theta_unpack, branching_ratio)

OUT = RESULTS
PARAMS_PKL = OUT / "etas" / "etas_sv_params.pkl"
REPORT = OUT / "etas" / "etas_sv_fit_report.json"

MAX_EM = 10          # design: <=10 EM iterations
DLL_TOL = 0.1        # design: or dLL < 0.1
BW_FLOOR_KM = 5.0    # design: Silverman on background-weighted points, floor 5 km
BW_CEIL_KM = 12.0    # keep the published ceiling


def silverman_bw_km(lon, lat, weights, region, floor=BW_FLOOR_KM, ceil=BW_CEIL_KM) -> float:
    """Silverman bandwidth (km) on background-weighted points. REPLICATES
    etas_model.BackgroundField's internal rule exactly, but with the 5 km floor
    (the published field floors at 1.5 km). Returned bw is passed explicitly to
    from_weighted_events so the field uses this bandwidth rather than its default."""
    ref_lon, ref_lat = _region_center(region)
    x, y = _project_km(np.asarray(lon, float), np.asarray(lat, float), ref_lon, ref_lat)
    w = np.asarray(weights, float).clip(min=0.0)
    if w.sum() <= 0 or x.size == 0:
        return floor
    w = w / w.sum()
    n_eff = 1.0 / (w * w).sum()                     # Kish effective sample size (w normalized)
    sx = np.sqrt(np.average((x - np.average(x, weights=w)) ** 2, weights=w))
    sy = np.sqrt(np.average((y - np.average(y, weights=w)) ** 2, weights=w))
    silverman = max(n_eff, 2.0) ** (-1.0 / 6.0) * 0.5 * (sx + sy)
    return float(np.clip(0.5 * silverman, floor, ceil))


def fit_sv(cat: pd.DataFrame, region: dict, cap: float):
    """MLE ETAS fit on the STAI-filtered catalogue with a CONVERGED weighted-KDE
    background (5 km-floor bandwidth). Structurally identical to etas_fit.fit_stai
    except the background loop runs to convergence and uses the 5 km floor."""
    cat = cat.sort_values("datetime_utc").reset_index(drop=True)
    mag = cat["mag_w"].to_numpy()
    mct = mc_of_time(cat["datetime_utc"], mag)
    complete = mag >= mct - 1e-9                     # STAI: keep only above Mc(t)
    n_dropped = int((~complete & (mag >= BASE_MC)).sum())

    sub = cat[complete & (mag >= BASE_MC)].reset_index(drop=True)
    t = ((sub["datetime_utc"] - pd.Timestamp("2003-01-01")) / pd.Timedelta(days=1)).to_numpy()
    lat = sub["latitude"].to_numpy(); lon = sub["longitude"].to_numpy()
    mag = sub["mag_w"].to_numpy()
    T_end = float((FIT_END - pd.Timestamp("2003-01-01")) / pd.Timedelta(days=1))

    order = np.argsort(t, kind="stable")
    t, lat, lon, mag = t[order], lat[order], lon[order], mag[order]
    ref_lon, ref_lat = _region_center(region)
    x, y = _project_km(lon, lat, ref_lon, ref_lat)
    dm = mag - BASE_MC

    tgt_idx = np.flatnonzero(t >= 0.0)
    n_tgt = tgt_idx.size
    b_pos = b_positive(mag[tgt_idx])
    b_val = float(b_pos) if b_pos is not None else 1.0
    b_aki = float(_aki_b(mag[tgt_idx], BASE_MC))

    pairs, tau_used = _build_pairs(t, x, y, dm, tgt_idx, 3650.0, 150.0)
    D = dict(pairs)
    D.update(n_tgt=n_tgt, T=T_end, R=150.0 ** 2, beta=b_val * LN10, dm_src=dm,
             tau2=np.minimum(T_end - t, tau_used), tau1=np.clip(-t, 0.0, tau_used))
    lon_t, lat_t = lon[tgt_idx], lat[tgt_idx]

    # stage-0 background: unit-weight KDE with the 5 km floor (matches published
    # code's stage-0 choice, only the floor differs).
    bw0 = silverman_bw_km(lon_t, lat_t, np.ones(n_tgt), region)
    bg = BackgroundField.from_weighted_events(lon_t, lat_t, np.ones(n_tgt), region, bw_km=bw0)
    D["u_tgt"] = bg.pdf_at_support_loo()

    rate = n_tgt / T_end
    inits = [_theta_pack(0.5 * rate, 0.35, 1.5, 0.01, 1.13, 1.5, 1.7, 0.4),
             _theta_pack(0.8 * rate, 0.15, 2.0, 0.03, 1.30, 5.0, 2.2, 0.1),
             _theta_pack(0.3 * rate, 0.60, 1.0, 0.005, 1.08, 0.5, 1.5, 0.8)]
    best = None
    for th0 in inits:
        res = _fit_stage(D, th0, cap, 200)
        if best is None or res.fun < best.fun - 1e-9:
            best = res

    # --- converged EM stochastic declustering (<=MAX_EM iters or dLL < DLL_TOL) ---
    ll_traj = [-float(best.fun)]
    bw_traj = [bw0]
    n_iter = 0
    for it in range(MAX_EM):
        rho = _rho_background(best.x, D)
        bw = silverman_bw_km(lon_t, lat_t, rho, region)
        bg = BackgroundField.from_weighted_events(lon_t, lat_t, rho, region, bw_km=bw)
        D["u_tgt"] = bg.pdf_at_support_loo()
        prev = best.fun
        best = _fit_stage(D, best.x, cap, 150)
        n_iter = it + 1
        ll_traj.append(-float(best.fun))
        bw_traj.append(float(bw))
        if abs(prev - best.fun) < DLL_TOL:
            break

    rho = _rho_background(best.x, D)
    bw_final = silverman_bw_km(lon_t, lat_t, rho, region)
    bg = BackgroundField.from_weighted_events(lon_t, lat_t, rho, region, bw_km=bw_final)

    mu, k, alpha, c, p, d, q, gamma = _theta_unpack(best.x, D["beta"])
    params = EtasParams(mu_total=float(mu), background_xy=bg, k=float(k),
                        alpha=float(alpha), c=float(c), p=float(p), d=float(d),
                        q=float(q), gamma=float(gamma), mc=float(BASE_MC),
                        b=float(b_val), region=dict(region))
    params._fit_result = best
    info = {"n_tgt": n_tgt, "n_dropped": n_dropped, "b_val": b_val, "b_aki": b_aki,
            "n_em_iter": n_iter, "ll_trajectory": ll_traj, "bw_trajectory": bw_traj,
            "bw_final_km": bw_final, "sum_rho_background": float(rho.sum()),
            "background_fraction": float(rho.sum() / max(n_tgt, 1))}
    return params, info


def main():
    OUT.mkdir(exist_ok=True)
    cat = pd.read_csv(OUT / "catalog" / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    cat = cat[cat["datetime_utc"] < FIT_END].copy()

    params, info = fit_sv(cat, MODEL_BOX, FALLBACK_CAP)
    n_op = branching_ratio(params)
    n_op_mmax = branching_ratio(params, mmax=7.6)

    with open(PARAMS_PKL, "wb") as f:
        pickle.dump(params, f)

    # first-gen params for a side-by-side (published fit): read the report instead of refitting.
    fg = json.load(open(OUT / "etas" / "etas_fit_report.json"))
    report = {
        "variant": "sv_etas",
        "base_mc": BASE_MC, "operational_cap": FALLBACK_CAP,
        "fit_end": str(FIT_END.date()),
        "n_events_fit": int(info["n_tgt"]), "n_dropped_by_stai": int(info["n_dropped"]),
        "b_positive": info["b_val"], "b_aki": info["b_aki"],
        "em_iterations_run": info["n_em_iter"], "em_max": MAX_EM, "dll_tol": DLL_TOL,
        "ll_trajectory": info["ll_trajectory"], "bw_trajectory_km": info["bw_trajectory"],
        "bw_floor_km": BW_FLOOR_KM, "bw_final_km": info["bw_final_km"],
        "background_events_est": info["sum_rho_background"],
        "background_fraction": info["background_fraction"],
        "mu_total": float(params.mu_total), "k": float(params.k),
        "alpha": float(params.alpha), "c": float(params.c), "p": float(params.p),
        "d": float(params.d), "q": float(params.q), "gamma": float(params.gamma),
        "operational_branching_untrunc": float(n_op),
        "operational_branching_mmax7.6": float(n_op_mmax),
        "converged": bool(getattr(params, "_fit_result").success),
        "vs_firstgen": {
            "note": "sv-ETAS isolates a CONVERGED weighted-KDE background (<=10 EM "
                    "iters, 5 km-floor bandwidth) vs the published 2-iteration, 1.5 km-"
                    "floor background. The first-gen fit is ALREADY spatially variable; "
                    "the delta here is background convergence + bandwidth, not "
                    "constant->variable mu (design premise corrected).",
            "firstgen_mu_total": fg.get("mu_total"), "sv_mu_total": float(params.mu_total),
            "firstgen_alpha": fg.get("alpha"), "sv_alpha": float(params.alpha),
            "firstgen_p": fg.get("p"), "sv_p": float(params.p),
            "firstgen_branching_untrunc": fg.get("operational_branching_untrunc"),
            "sv_branching_untrunc": float(n_op),
        },
    }
    with open(REPORT, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
