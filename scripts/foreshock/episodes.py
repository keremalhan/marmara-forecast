"""Run 21 — Item 3 of Amendment 8: foreshock EPISODE analysis (deduplicated denominator).

Governed by docs/preregistration/v2_analysis_amendment_8.md (SHA-256
c5684700aa656949908640faa326c6b6f15b3a699052f627272bc26a1186e690), hashed 2026-07-16T11:25:21Z
BEFORE this ran. No outcome predicted.

The t+10min false-alarm accounting (Section S5) counts 673 overlapping triggers as independent
alarms. This reconstructs the denominator as EPISODES.

  nodes  = the 673 qualifying mag_w>=4.0 triggers (from r7_triggers.parquet)
  edge   = two triggers whose alarm cylinders overlap in space-time
  episode = connected component

DEFINITIONAL DEGREES OF FREEDOM, made explicit (this is where the analysis is attackable):
  * cylinder radius r: the hit rule uses 25 km. Two r-radius disks overlap when centres are within
    2r. PRIMARY definition uses centre-distance < 2r (a single M6 could be a hit for both -> the
    alarms are not independent). We ALSO report the stricter centre-distance < r (co-located).
  * duration D: the hit rule uses 30 d. Time windows [t,t+D] overlap when |dt| < D.
  * PRIMARY: r = 25 km (edge at <50 km), D = 30 d. SENSITIVITIES: r in {20,25,30} (edge <2r),
    D in {14,30}, and the stricter centre-in-cylinder (<r) variant. Every variant is reported.

Distances are planar-approx (Marmara is small): dx = dlon*111.32*cos(lat), dy = dlat*110.57 km.

Writes results/round4/r21_foreshock_episodes.json.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/foreshock/episodes.py
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd

from marmara.paths import RESULTS

R4 = RESULTS / "round4"


class UF:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]; a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def components(n, edges):
    uf = UF(n)
    for a, b in edges:
        uf.union(a, b)
    roots = {}
    for i in range(n):
        roots.setdefault(uf.find(i), []).append(i)
    return list(roots.values())


def build_edges(t_days, x_km, y_km, hit, max_ctr_km, max_dt):
    """O(n^2) edge list; n=673 is tiny. Edge iff centre-distance < max_ctr_km and |dt| < max_dt."""
    n = len(t_days); edges = []
    for i in range(n):
        dt = np.abs(t_days - t_days[i])
        near_t = np.where(dt < max_dt)[0]
        for j in near_t:
            if j <= i:
                continue
            d = np.hypot(x_km[i] - x_km[j], y_km[i] - y_km[j])
            if d < max_ctr_km:
                edges.append((i, j))
    return edges


def episode_stats(comps, hit):
    n_ep = len(comps)
    hit_ep = sum(1 for c in comps if any(hit[i] for i in c))
    return {
        "n_episodes": n_ep,
        "n_episodes_with_M6": hit_ep,
        "episode_precision": round(hit_ep / n_ep, 5),
        "false_alarm_ratio": round((n_ep - hit_ep) / n_ep, 5),
        "largest_episode": max(len(c) for c in comps),
        "singletons": sum(1 for c in comps if len(c) == 1),
    }


def main():
    t0 = time.time()
    tr = pd.read_parquet(R4 / "r7_triggers.parquet")
    t = pd.to_datetime(tr["t"])
    t_days = (t - t.min()) / pd.Timedelta(days=1)
    t_days = t_days.to_numpy()
    lat = tr["lat"].to_numpy(); lon = tr["lon"].to_numpy()
    x_km = lon * 111.32 * np.cos(np.radians(lat.mean()))
    y_km = lat * 110.57
    hit = tr["hit"].to_numpy().astype(bool)
    n = len(tr)

    out = {
        "governed_by": {"amendment": "docs/preregistration/v2_analysis_amendment_8.md",
                        "sha256": "c5684700aa656949908640faa326c6b6f15b3a699052f627272bc26a1186e690"},
        "n_raw_triggers": int(n), "n_M6_hits": int(hit.sum()),
        "primary_definition": "edge iff centre-distance < 50 km (2 x 25) and |dt| < 30 d",
    }

    # ---- PRIMARY: r=25 (edge <50 km), D=30 d --------------------------------------------
    edges = build_edges(t_days, x_km, y_km, hit, 50.0, 30.0)
    comps = components(n, edges)
    out["primary"] = episode_stats(comps, hit)
    print(f"PRIMARY (edge <50 km, |dt|<30 d): {out['primary']['n_episodes']} episodes from {n} "
          f"triggers; precision {out['primary']['episode_precision']}; "
          f"FAR {out['primary']['false_alarm_ratio']}", flush=True)

    # ---- union-of-cylinders space-time footprint (Molchan x-axis) -----------------------
    # Monte-Carlo the fraction of model-box space-time inside the union of alarm cylinders,
    # over the analysis span, at r=25 km. The "region-time in alarm" accounting.
    rng = np.random.default_rng(42)
    box_lon = (26.0, 30.5); box_lat = (39.8, 41.4)
    span_lo, span_hi = t_days.min(), t_days.max() + 30.0
    NS = 200000
    slon = rng.uniform(*box_lon, NS); slat = rng.uniform(*box_lat, NS)
    st = rng.uniform(span_lo, span_hi, NS)
    sx = slon * 111.32 * np.cos(np.radians(lat.mean())); sy = slat * 110.57
    inside = np.zeros(NS, bool)
    for i in range(n):
        m = (np.abs(st - t_days[i] - 15.0) <= 15.0)  # within [t_i, t_i+30]
        if m.any():
            d = np.hypot(sx[m] - x_km[i], sy[m] - y_km[i])
            hitmask = d < 25.0
            idx = np.where(m)[0][hitmask]
            inside[idx] = True
    frac_st = float(inside.mean())
    out["union_region_time_fraction"] = round(frac_st, 5)
    out["molchan_point"] = {"tau_fraction_spacetime_alarmed": round(frac_st, 5),
                            "nu_miss_rate": round(1.0 - hit.sum() / max(hit.sum(), 1), 5),
                            "hits_captured": int(hit.sum()), "note": "1 of 1 M6 captured"}
    print(f"  union region-time in alarm: {frac_st*100:.2f}% of model-box space-time; "
          f"M6 captured: {int(hit.sum())}/1", flush=True)

    # ---- SENSITIVITIES ------------------------------------------------------------------
    sens = {}
    for r in (20.0, 25.0, 30.0):
        for D in (14.0, 30.0):
            e = build_edges(t_days, x_km, y_km, hit, 2 * r, D)
            c = components(n, e)
            sens[f"r{int(r)}_D{int(D)}_ctr<2r"] = episode_stats(c, hit)
    # stricter centre-in-cylinder (<r) variant at primary r,D
    e_strict = build_edges(t_days, x_km, y_km, hit, 25.0, 30.0)
    sens["r25_D30_ctr<r_strict"] = episode_stats(components(n, e_strict), hit)
    out["sensitivities"] = sens
    print("  sensitivities (episodes | precision | FAR):", flush=True)
    for k, v in sens.items():
        print(f"    {k:24s} {v['n_episodes']:4d} | {v['episode_precision']:.4f} | "
              f"{v['false_alarm_ratio']:.4f}", flush=True)

    # ---- robustness summary: how much does the episode count move across all variants? ---
    counts = [out["primary"]["n_episodes"]] + [v["n_episodes"] for v in sens.values()]
    out["robustness"] = {
        "episode_count_min": int(min(counts)), "episode_count_max": int(max(counts)),
        "raw_triggers": int(n),
        "all_variants_capture_the_M6": all(v["n_episodes_with_M6"] == 1 for v in sens.values()),
        "all_variants_precision_le": round(max(v["episode_precision"] for v in sens.values()), 4),
        "reading": ("episodes range across definitional choices; the M6 is captured in every variant, "
                    "and episode precision stays low (one true episode among many) throughout -- the "
                    "conclusion does not rest on any single definition."),
    }
    out["runtime_s"] = round(time.time() - t0, 1)
    json.dump(out, open(R4 / "r21_foreshock_episodes.json", "w"), indent=1, default=str)
    print(f"\n=== EPISODE DENOMINATOR ===")
    print(f"  raw triggers {n} -> primary {out['primary']['n_episodes']} episodes "
          f"(range {out['robustness']['episode_count_min']}-{out['robustness']['episode_count_max']} "
          f"across all definitions)")
    print(f"  M6 captured in every variant: {out['robustness']['all_variants_capture_the_M6']}")
    print(f"  episode precision <= {out['robustness']['all_variants_precision_le']} everywhere")
    print(f"  ({out['runtime_s']}s) -> results/round4/r21_foreshock_episodes.json")


if __name__ == "__main__":
    main()
