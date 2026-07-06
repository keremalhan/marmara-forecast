"""Prospective monthly forecasting — the cheapest credibility there is.

Each run issues a 30-day forecast at the current catalog end, appends it to an
APPEND-ONLY, sha256-hashed log BEFORE the outcome is known, and scores any past
forecast whose 30-day window has now closed against what actually happened. Over
months this accumulates a genuine out-of-sample (prospective) track record —
unlike the pseudo-prospective backtests, these forecasts were logged with no
knowledge of the future.

Layout (results/prospective/):
  forecast_log.jsonl   append-only; one issued forecast per line + content hash
  scored_log.jsonl     append-only; one closed forecast's realized outcome per line
  track_record.md      running accumulation: predicted vs realized, Brier, reliability
  <catalog_end>/       full per-forecast outputs (grids .npz, maps, summary)

Idempotent: re-running on an unchanged catalog end does NOT log a duplicate forecast,
it only advances scoring. No protected files touched.

Run:  "<venv>/bin/python3" -m marmara.prospective run     # issue (if new) + score + track
      "<venv>/bin/python3" -m marmara.prospective score   # score + track only
"""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from marmara.paths import ROOT, RESULTS, DATA, MODELS, SEG_PATH, STRAIN_NPZ, KOERI_CSV  # noqa: E402,F401

from marmara import grid as G
from marmara.cascade import cascade_forecast
from marmara.metrics import lambda_to_p, p_to_lambda
from marmara.renewal import SEGMENTS, NOW_YEAR, conditional_prob

OUT = RESULTS
PRO = OUT / "prospective"
FLOG = PRO / "forecast_log.jsonl"
SLOG = PRO / "scored_log.jsonl"
DAY = pd.Timedelta(days=1)
EPS = 1e-9
K_LIVE = 8000
LEVELS = (5.0, 5.5, 6.0)


def _hash(d: dict) -> str:
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _hybrid_P(grid_row_feats, lam_sim, count_col):
    d = pickle.load(open(OUT / "models" / f"{count_col}_hybrid.pkl", "rb"))
    X = grid_row_feats.copy(); X["ln_lam_sim"] = np.log(lam_sim.ravel() + EPS)
    lam_ml = np.clip(d["reg"].predict(X), EPS, None)
    lam_h = np.clip(lam_sim.ravel(), EPS, None) ** (1 - d["w"]) * lam_ml ** d["w"]
    return d["iso"].transform(lambda_to_p(lam_h))


def issue_forecast():
    params = pickle.load(open(OUT / "etas_params.pkl", "rb"))
    rep = json.load(open(OUT / "etas_fit_report.json"))
    b_op, b_aki, b_pos = rep["operational_b_for_cascade"], rep["b_aki"], rep["b_positive"]
    cat = pd.read_csv(OUT / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    spec = G.MODEL_SPEC
    catalog_end = cat["datetime_utc"].max()
    t0 = catalog_end.normalize() + DAY            # forecast issued from the day after last event
    t0d = float(G._to_days(t0))
    target_end = t0 + 30 * DAY
    hist = cat[["datetime_utc", "longitude", "latitude", "mag_w"]]

    # cascade b-ensemble -> regional exceedance P + per-cell rates
    ens = {tag: cascade_forecast(params, hist, t0d, 30.0, spec.lon_c, spec.lat_c,
                                 K=K_LIVE, seed=1234, b=bb, per_sim_cap=50000)
           for tag, bb in (("aki", b_aki), ("op", b_op), ("pos", b_pos))}
    regional = {f"M{lvl}": {"central": ens["op"][f"Preg{lvl}"],
                            "range": [ens["pos"][f"Preg{lvl}"], ens["aki"][f"Preg{lvl}"]]}
                for lvl in LEVELS}
    # hybrid y35 / y45 per-cell probability maps (the issued classification products)
    EV = G.build_event_bundle(cat, 3.0); ctx = G.build_static_context()
    feats = G.features_at_window(EV, t0d, t0, ctx, params)
    Xf = pd.DataFrame({k: feats[k].ravel() for k in G.FEATURES})
    cop = ens["op"]
    P35 = _hybrid_P(Xf, cop["lam35"], "count35")
    P45 = _hybrid_P(Xf, cop["lam45"], "count45")
    lam35 = cop["lam35"].ravel()

    # renewal (30-day) per segment
    renew = {n: conditional_prob(s["mu_r"], NOW_YEAR - s["last"], 30 / 365.25)
             for n, s in SEGMENTS.items()}
    LO, LA = np.meshgrid(G.LON_C, G.LAT_C)
    top = pd.DataFrame({"lon": LO.ravel(), "lat": LA.ravel(), "P35": P35, "P45": P45}
                       ).sort_values("P35", ascending=False).head(10)

    fc = {
        "schema": "prospective_v1", "issued_utc": datetime.now(timezone.utc).isoformat(),
        "catalog_end": str(catalog_end), "t0": str(t0), "target_end": str(target_end),
        "horizon_days": 30, "b_ensemble": {"op": b_op, "aki": b_aki, "pos": b_pos},
        "regional_exceedance": regional,
        "combined_M6.8": float(1 - (1 - regional["M6.0"]["central"])
                               * (1 - (1 - np.prod([1 - v for v in renew.values()])))),
        "renewal_30d_M7": renew,
        "top_cells": top.round(4).to_dict("records"),
        "n_events_history": int(len(cat)),
        "K": K_LIVE, "scored": False,
    }
    # save per-cell grids for later per-cell scoring
    dst = PRO / t0.strftime("%Y-%m-%d"); dst.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst / "grids.npz", P35=P35, P45=P45, lam35=lam35,
                        lon=LO.ravel(), lat=LA.ravel())
    fc["grids"] = str((dst / "grids.npz").relative_to(OUT))
    # hash the PREDICTION content (stable fingerprint) — exclude issue-time metadata so
    # the same catalog reproduces the same hash and a logged forecast is verifiable.
    fc["content_hash"] = _hash({k: v for k, v in fc.items()
                                if k not in ("content_hash", "issued_utc")})
    return fc


def _read_jsonl(p):
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def log_forecast(fc):
    PRO.mkdir(parents=True, exist_ok=True)
    existing = _read_jsonl(FLOG)
    if any(e["t0"] == fc["t0"] for e in existing):
        return False                                # idempotent: already logged this t0
    with open(FLOG, "a") as f:
        f.write(json.dumps(fc, default=str) + "\n")
    return True


def score_closed():
    cat = pd.read_csv(OUT / "catalog.csv"); cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    catalog_end = cat["datetime_utc"].max()
    spec = G.MODEL_SPEC
    logged = _read_jsonl(FLOG)
    scored_t0 = {s["t0"] for s in _read_jsonl(SLOG)}
    newly = []
    for fc in logged:
        t0 = pd.Timestamp(fc["t0"]); tend = pd.Timestamp(fc["target_end"])
        if fc["t0"] in scored_t0 or tend > catalog_end:
            continue                                # not closed yet, or already scored
        win = cat[(cat["datetime_utc"] >= t0) & (cat["datetime_utc"] < tend)]
        realized = {f"M{lvl}": int((win["mag_w"] >= lvl).sum()) for lvl in LEVELS}
        occ = {f"M{lvl}": int(realized[f"M{lvl}"] > 0) for lvl in LEVELS}
        # per-cell y35/y45 realized (Brier vs the saved grids)
        gz = np.load(OUT / fc["grids"])
        ir, ic = G.cell_index_spec(win["longitude"].to_numpy(), win["latitude"].to_numpy(), spec)
        brier = {}
        for thr, key in ((3.5, "P35"), (4.5, "P45")):
            occg = np.zeros((spec.nlat, spec.nlon))
            w = win[win["mag_w"] >= thr]
            jr, jc = G.cell_index_spec(w["longitude"].to_numpy(), w["latitude"].to_numpy(), spec)
            on = jr >= 0
            occg[jr[on], jc[on]] = 1.0
            brier[key] = float(np.mean((gz[key] - occg.ravel()) ** 2))
        sc = {"t0": fc["t0"], "target_end": fc["target_end"], "scored_utc": datetime.now(timezone.utc).isoformat(),
              "content_hash": fc["content_hash"], "realized_counts": realized, "realized_occ": occ,
              "predicted_regional": {k: fc["regional_exceedance"][k]["central"] for k in fc["regional_exceedance"]},
              "brier_percell": brier}
        with open(SLOG, "a") as f:
            f.write(json.dumps(sc) + "\n")
        newly.append(sc)
    return newly


def track_record():
    scored = _read_jsonl(SLOG)
    lines = ["# Prospective track record (true out-of-sample)", "",
             f"Forecasts issued (logged): **{len(_read_jsonl(FLOG))}**; "
             f"windows closed & scored: **{len(scored)}**.",
             "Each forecast was hashed and logged before its outcome was known (see forecast_log.jsonl).", ""]
    if not scored:
        lines += ["_No 30-day windows have closed yet — the first score lands ~30 days after the "
                  "first issued forecast. Credibility accrues from here._"]
        (PRO / "track_record.md").write_text("\n".join(lines))
        return
    # regional exceedance: Brier + mean predicted vs observed frequency, per level
    lines += ["## Regional 30-day exceedance (predicted P vs realized)", "",
              "| level | n scored | mean predicted | observed freq | Brier |", "|---|---|---|---|---|"]
    for lvl in LEVELS:
        k = f"M{lvl}"
        p = np.array([s["predicted_regional"][k] for s in scored])
        o = np.array([s["realized_occ"][k] for s in scored], float)
        brier = float(np.mean((p - o) ** 2))
        lines.append(f"| M≥{lvl} | {len(scored)} | {p.mean()*100:.2f}% | {o.mean()*100:.2f}% | {brier:.4f} |")
    b35 = np.mean([s["brier_percell"]["P35"] for s in scored])
    b45 = np.mean([s["brier_percell"]["P45"] for s in scored])
    lines += ["", f"Per-cell Brier (mean over scored months): y35 {b35:.5f}, y45 {b45:.5f}.", "",
              "## Scored forecasts", "| t0 | target_end | realized M≥5/5.5/6 | pred P(M≥6) |", "|---|---|---|---|"]
    for s in scored:
        r = s["realized_counts"]
        lines.append(f"| {s['t0'][:10]} | {s['target_end'][:10]} | "
                     f"{r['M5.0']}/{r['M5.5']}/{r['M6.0']} | {s['predicted_regional']['M6.0']*100:.2f}% |")
    (PRO / "track_record.md").write_text("\n".join(lines))


def refresh_catalog():
    """Best-effort catalog refresh. The KOERI fetcher is a network scraper (blocked in
    sandbox; runs fine under launchd). We call the operator's refresh hook if present.
    Returns (advanced: bool, note: str). On any failure we proceed with the existing
    catalog and log the status — the run still scores closed forecasts."""
    hook = ROOT / "scripts" / "refresh_monthly.py"
    if not hook.exists():
        return (False, "no refresh hook (scripts/refresh_monthly.py) — using existing catalog")
    import subprocess
    try:
        r = subprocess.run([sys.executable, str(hook)], capture_output=True, timeout=1800, text=True)
        return (r.returncode == 0, (r.stdout or r.stderr or "")[-300:])
    except Exception as e:
        return (False, f"refresh failed: {e}")


def main():
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    PRO.mkdir(parents=True, exist_ok=True)
    status = {"run_utc": datetime.now(timezone.utc).isoformat()}
    if cmd == "run":
        advanced, note = refresh_catalog()
        status["catalog_refresh"] = {"advanced": advanced, "note": note}
        fc = issue_forecast()
        status["issued"] = log_forecast(fc)          # False if this catalog_end already logged
        status["catalog_end"] = fc["catalog_end"]; status["content_hash"] = fc["content_hash"]
    newly = score_closed()
    status["newly_scored"] = len(newly)
    track_record()
    with open(PRO / "last_run.json", "w") as f:
        json.dump(status, f, indent=2)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
