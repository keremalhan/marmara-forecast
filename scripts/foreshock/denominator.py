"""Run 7 — Foreshock false-alarm denominator.

Governed by docs/preregistration/v2_preregistration_amendment_4.md section B
(SHA-256 e103f573e38d12d336e0c9871a0897ae50ee52339e536b20612a8c2d727519eb), written and
hashed BEFORE this ran.

WHY. §4/§5 rest the "useful warning arrived only 36 minutes before rupture" claim on one
event: ten minutes after the lone ML 4.0 foreshock the Kumburgaz epicentral cell's 30-day
P(M>=6) gain jumped to 40x. That reports the one time the statistic fired before a mainshock.
It does not report how often the same statistic fires when NO mainshock follows. This run
supplies that denominator.

WHAT. For every mag_w >= 4.0 event in the model box, run the IDENTICAL Kumburgaz machinery
(marmara.m62_countdown.Case.forecast) at two strictly causal freezes -- t-1s and t+10min --
and record the 30-day P(M>=6) gain over the freeze's uniform-spread Poisson base. Count how
many escalate past 10x / 40x, and how many are followed by an actual M>=6 within 30 d and
25 km.

Writes results/round4/r7_foreshock_denominator.json. Reads only.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/foreshock/denominator.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from marmara import grid as G
from marmara.cascade import cascade_forecast
from marmara.paths import RESULTS

OUT = RESULTS / "round4"
OUT.mkdir(exist_ok=True)

# --- pre-registered constants (Amendment 4 section B) ---
TRIG_MAG = 4.0
FOLLOW_MAG = 6.0
FOLLOW_DAYS = 30.0
ASSOC_KM = 25.0
NBR_KM = 25.0
H_DAYS = 30.0
K_SIM = 6000
SEED = 42
PER_SIM_CAP = 50000
B_OP = 1.15
DM = 2.5                    # M6.0 - M3.5, the analytic GR step
POST_OFFSET = pd.Timedelta(minutes=10)
PRE_OFFSET = pd.Timedelta(seconds=1)
THRESHOLDS = (10.0, 40.0)
LADDER = [("+1h", pd.Timedelta(hours=1)), ("+6h", pd.Timedelta(hours=6)),
          ("+1d", pd.Timedelta(days=1)), ("+3d", pd.Timedelta(days=3)),
          ("+7d", pd.Timedelta(days=7)), ("+14d", pd.Timedelta(days=14)),
          ("+30d", pd.Timedelta(days=30))]
KM_PER_DEG = 111.0
DAY = pd.Timedelta(days=1)


def analytic_P(lam35, b, dm):
    """Per-cell P(M>=3.5+dm) via GR from the dense per-cell lam35 field. (m62_countdown fix#1)"""
    return 1.0 - np.exp(-np.asarray(lam35) * 10.0 ** (-b * dm))


def gain_at(cat, params, spec, freeze, ir, ic, nbr_mask):
    """Kumburgaz machinery, verbatim in substance (m62_countdown.Case.forecast).
    Returns (P6_cell, base6, gain_cell, gain_nbr_max)."""
    h = cat[cat["datetime_utc"] < freeze][["datetime_utc", "longitude", "latitude", "mag_w"]]
    assert len(h) == 0 or h["datetime_utc"].max() < freeze, "causality violated"
    c = cascade_forecast(params, h, float(G._to_days(freeze)), H_DAYS, spec.lon_c, spec.lat_c,
                         K=K_SIM, seed=SEED, b=B_OP, per_sim_cap=PER_SIM_CAP)
    lam35 = c["lam35"]
    past = h
    yrs = max((freeze - cat["datetime_utc"].min()).days / 365.25, 0.1)
    reg35_30d = float((past["mag_w"] >= 3.5).sum()) / yrs / 12.1667
    base6 = float(analytic_P(np.array([reg35_30d / spec.ncells]), B_OP, DM)[0])
    p6_cell = float(analytic_P(np.array([lam35[ir, ic]]), B_OP, DM)[0])
    p6_nbr = analytic_P(lam35.ravel()[nbr_mask], B_OP, DM)
    g_cell = (p6_cell / base6) if base6 > 0 else None
    g_nbr = (float(p6_nbr.max()) / base6) if base6 > 0 and p6_nbr.size else None
    return p6_cell, base6, g_cell, g_nbr


def main():
    t_all = time.time()
    params = G.load_params()
    spec = G.MODEL_SPEC
    cat = pd.read_csv(RESULTS / "catalog" / "catalog.csv")
    cat["datetime_utc"] = pd.to_datetime(cat["datetime_utc"])
    cat = cat.sort_values("datetime_utc").reset_index(drop=True)
    lo, hi = cat["datetime_utc"].min(), cat["datetime_utc"].max()

    # cell centres (flat), for the 25-km neighborhood mask
    LO, LA = np.meshgrid(spec.lon_c, spec.lat_c)
    cl, ca = LO.ravel(), LA.ravel()

    # ---- trigger set (pre-registered filters, each counted) ----
    inbox = ((cat["longitude"] >= spec.lon_c.min() - 0.05) & (cat["longitude"] <= spec.lon_c.max() + 0.05)
             & (cat["latitude"] >= spec.lat_c.min() - 0.05) & (cat["latitude"] <= spec.lat_c.max() + 0.05))
    trig_all = cat[(cat["mag_w"] >= TRIG_MAG) & inbox].reset_index(drop=True)
    burn = trig_all["datetime_utc"] >= lo + pd.Timedelta(days=365)
    obs = trig_all["datetime_utc"] + pd.Timedelta(days=FOLLOW_DAYS) <= hi
    trig = trig_all[burn & obs].reset_index(drop=True)
    counts = {"m4_in_box_all": int(len(trig_all)),
              "after_base_burnin_365d": int(burn.sum()),
              "after_followon_observable_30d": int(obs.sum()),
              "primary_denominator_both_filters": int(len(trig)),
              "subset_2024_onward": int((trig["datetime_utc"] >= pd.Timestamp("2024-01-01")).sum())}
    print(json.dumps(counts, indent=1), flush=True)

    big = cat[cat["mag_w"] >= FOLLOW_MAG].reset_index(drop=True)
    print(f"M>={FOLLOW_MAG} events in catalogue: {len(big)}", flush=True)

    rows = []
    for i, r in trig.iterrows():
        t = r["datetime_utc"]; elon = float(r["longitude"]); elat = float(r["latitude"])
        ir, ic = (int(x[0]) for x in G.cell_index_spec(np.array([elon]), np.array([elat]), spec))
        coslat = np.cos(np.radians(elat))
        dkm = np.sqrt(((cl - elon) * KM_PER_DEG * coslat) ** 2 + ((ca - elat) * KM_PER_DEG) ** 2)
        nbr = np.where(dkm <= NBR_KM)[0]

        p6_pre, base_pre, g_pre, gn_pre = gain_at(cat, params, spec, t - PRE_OFFSET, ir, ic, nbr)
        p6_post, base_post, g_post, gn_post = gain_at(cat, params, spec, t + POST_OFFSET, ir, ic, nbr)
        jump = (p6_post / p6_pre) if p6_pre > 0 else None

        # follow-on: M>=6 within 30 d AND within 25 km of the trigger
        fo = big[(big["datetime_utc"] > t) & (big["datetime_utc"] <= t + FOLLOW_DAYS * DAY)]
        hit = False; hit_info = None
        for _, e in fo.iterrows():
            d = np.sqrt(((float(e["longitude"]) - elon) * KM_PER_DEG * coslat) ** 2
                        + ((float(e["latitude"]) - elat) * KM_PER_DEG) ** 2)
            if d <= ASSOC_KM:
                hit = True
                hit_info = {"time": str(e["datetime_utc"]), "mag_w": float(e["mag_w"]),
                            "dist_km": round(float(d), 2),
                            "dt_min": round((e["datetime_utc"] - t).total_seconds() / 60.0, 1)}
                break

        rows.append({"i": int(i), "t": str(t), "mag_w": float(r["mag_w"]),
                     "lon": elon, "lat": elat, "ir": ir, "ic": ic, "n_nbr_cells": int(len(nbr)),
                     "P6_pre": p6_pre, "P6_post": p6_post,
                     "gain_pre": g_pre, "gain_post": g_post,
                     "gain_nbr_pre": gn_pre, "gain_nbr_post": gn_post,
                     "jump": jump, "hit": bool(hit), "hit_info": hit_info,
                     "is_2024plus": bool(t >= pd.Timestamp("2024-01-01"))})
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(trig)} triggers ({time.time()-t_all:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "r7_triggers.parquet", index=False)

    # ---- escalation accounting, both pre-registered definitions ----
    def account(sub, stat_col, label):
        out = {}
        n = len(sub)
        for th in THRESHOLDS:
            esc = sub[sub[stat_col] >= th]
            hits = int(esc["hit"].sum())
            out[f"ge_{int(th)}x"] = {
                "n_escalations": int(len(esc)),
                "n_hits": hits,
                "n_false_alarms": int(len(esc)) - hits,
                "false_alarm_rate": (round(1.0 - hits / len(esc), 6) if len(esc) else None),
                "escalation_rate_of_triggers": (round(len(esc) / n, 6) if n else None),
            }
        return {"statistic": label, "n_triggers": n, **out}

    def dist(v):
        v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
        if not v.size:
            return None
        q = [0, 5, 25, 50, 75, 90, 95, 99, 100]
        return {"n": int(v.size), "mean": round(float(v.mean()), 4),
                "percentiles": {f"p{p}": round(float(np.percentile(v, p)), 4) for p in q}}

    res = {
        "governed_by": {"amendment": "docs/preregistration/v2_preregistration_amendment_4.md",
                        "sha256": "e103f573e38d12d336e0c9871a0897ae50ee52339e536b20612a8c2d727519eb",
                        "section": "B"},
        "config": {"trigger_mag_w": TRIG_MAG, "follow_mag_w": FOLLOW_MAG,
                   "follow_days": FOLLOW_DAYS, "assoc_km": ASSOC_KM, "nbr_km": NBR_KM,
                   "horizon_days": H_DAYS, "K": K_SIM, "seed": SEED,
                   "per_sim_cap": PER_SIM_CAP, "b_op": B_OP,
                   "catalogue_span": [str(lo), str(hi)],
                   "machinery": "marmara.m62_countdown.Case.forecast (identical)"},
        "trigger_counts": counts,
        "n_M6_in_catalogue": int(len(big)),
        "M6_events": [{"time": str(e["datetime_utc"]), "mag_w": float(e["mag_w"]),
                       "lon": float(e["longitude"]), "lat": float(e["latitude"])}
                      for _, e in big.iterrows()],
        "distributions": {
            "gain_post_cell": dist(df["gain_post"]),
            "gain_pre_cell": dist(df["gain_pre"]),
            "jump_cell": dist(df["jump"]),
            "gain_post_nbr25km": dist(df["gain_nbr_post"]),
        },
        "accounting": {
            "primary_gain_post_cell": account(df, "gain_post", "absolute gain at t+10min (cell)"),
            "secondary_jump_cell": account(df, "jump", "jump = P6(t+10min)/P6(t-1s) (cell)"),
            "gain_post_nbr25km": account(df, "gain_nbr_post", "absolute gain at t+10min (max over 25km)"),
        },
        "subset_2024plus": {
            "primary_gain_post_cell": account(df[df["is_2024plus"]], "gain_post",
                                              "absolute gain at t+10min (cell), 2024+"),
        },
        "hits": [r for r in rows if r["hit"]],
        "n_hits_total": int(df["hit"].sum()),
    }

    # ---- alarm duration ladder for escalating triggers (primary definition, 10x) ----
    esc = df[df["gain_post"] >= 10.0]
    print(f"escalations (>=10x, primary): {len(esc)} of {len(df)}; running duration ladder ...",
          flush=True)
    dur = []
    for _, r in esc.iterrows():
        t = pd.Timestamp(r["t"]); elon, elat = r["lon"], r["lat"]
        ir, ic = int(r["ir"]), int(r["ic"])
        coslat = np.cos(np.radians(elat))
        dkm = np.sqrt(((cl - elon) * KM_PER_DEG * coslat) ** 2 + ((ca - elat) * KM_PER_DEG) ** 2)
        nbr = np.where(dkm <= NBR_KM)[0]
        lad = {}
        for lbl, off in LADDER:
            if t + off > hi:
                lad[lbl] = None
                continue
            _, _, g, _ = gain_at(cat, params, spec, t + off, ir, ic, nbr)
            lad[lbl] = g
        out = {"t": r["t"], "gain_post": r["gain_post"], "ladder": lad}
        for th in THRESHOLDS:
            last = None
            for lbl, _ in LADDER:
                if lad.get(lbl) is not None and lad[lbl] >= th:
                    last = lbl
                else:
                    break
            out[f"last_above_{int(th)}x"] = last
        dur.append(out)
        if len(dur) % 25 == 0:
            print(f"  duration {len(dur)}/{len(esc)} ({time.time()-t_all:.0f}s)", flush=True)

    def bracket_summary(key):
        order = [l for l, _ in LADDER]
        tally = {}
        for d in dur:
            k = d.get(key) or "<+1h"
            tally[k] = tally.get(k, 0) + 1
        return {k: tally.get(k, 0) for k in ["<+1h"] + order if tally.get(k, 0)}

    res["alarm_duration"] = {
        "definition": ("last forward-ladder point at which the cell gain remains >= threshold; "
                       "interval-censored -- the ladder is coarse by design"),
        "ladder": [l for l, _ in LADDER],
        "n_escalating_10x": int(len(esc)),
        "bracket_counts_10x": bracket_summary("last_above_10x"),
        "bracket_counts_40x": bracket_summary("last_above_40x"),
        "per_trigger": dur,
    }
    res["runtime_s"] = round(time.time() - t_all, 1)
    json.dump(res, open(OUT / "r7_foreshock_denominator.json", "w"), indent=2)

    a = res["accounting"]["primary_gain_post_cell"]
    print("\n=== PRIMARY (absolute gain at t+10min, trigger cell) ===")
    print(f"triggers: {a['n_triggers']}")
    for th in THRESHOLDS:
        k = a[f"ge_{int(th)}x"]
        print(f"  >={int(th):2d}x : {k['n_escalations']:4d} escalations, {k['n_hits']} hits, "
              f"{k['n_false_alarms']} false alarms, FAR {k['false_alarm_rate']}")
    print(f"total hits: {res['n_hits_total']}   M>=6 in catalogue: {res['n_M6_in_catalogue']}")
    print(f"runtime {res['runtime_s']}s -> results/round4/r7_foreshock_denominator.json")


if __name__ == "__main__":
    main()
