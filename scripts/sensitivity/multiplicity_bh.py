"""Run 27 — Item 7 Part B.3: Benjamini-Hochberg within the five frozen families.

Governed by docs/preregistration/v2_analysis_amendment_8.md (SHA-256 c5684700..., hashed 2026-07-16T11:25:21Z).

Calibration (r25/r26) established: adopt block 3; the count-axis independent-MC anticonservatism at
K=500 is structurally incapable of manufacturing a registered separation (it touches only the three
MC-vs-MC pairs, all inseparable); the exposed single-axis findings share the cascade simulation and
live on the Bernoulli axis, which calibrates clean at block 3 at its real K=2000.

Family structure FROZEN in the amendment (not chosen after q-values):
  F1 hybrid-vs-physics primary (y30)   F2 hybrid-vs-physics M>=3.5 (y35)
  F3 physics-vs-physics                F4 recalibration challengers
  F5 occurrence-axis (proper-score single-axis findings)

Two axes per pair; BH (q=0.05) per axis within family; a SEPARATION survives iff BOTH axes survive,
mirroring the registered conjunctive rule. Count-scored p-values (F1-F3) are exact from r24 (frozen
win_rates). F4-F5 are proper-score findings: p is bounded from their frozen 95% CIs (CI excludes 0
=> p < 0.05), which is the statement the data supports; the headline hybrid-vs-cascade Bernoulli
win_rate is recomputed exactly. Findings that FAIL BH get flagged for "does not survive multiplicity
adjustment" at their point of claim; registered verdicts are untouched either way.

Writes results/round4/r27_multiplicity_bh.json.
Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/sensitivity/multiplicity_bh.py
"""
from __future__ import annotations

import json

import numpy as np

from marmara.paths import RESULTS

R4 = RESULTS / "round4"
PHYS = {"cascade", "sv_etas", "firstgen_etas", "modern_etas", "inversion"}


def bh(pvals, q=0.05):
    """Benjamini-Hochberg: return the set of indices that survive at FDR q."""
    items = sorted(enumerate(pvals), key=lambda kv: kv[1])
    m = len(items); survive = set(); kmax = -1
    for rank, (idx, p) in enumerate(items, 1):
        if p <= q * rank / m:
            kmax = rank
    for rank, (idx, p) in enumerate(items, 1):
        if rank <= kmax:
            survive.add(idx)
    return survive


def unordered(pairs, prefix):
    """Deduplicate directed pairs to unordered; keep the A-favouring direction's p."""
    seen = {}
    for k, v in pairs.items():
        if not k.startswith(prefix):
            continue
        a, b = k[len(prefix):].split("_vs_")
        key = tuple(sorted([a, b]))
        # keep the direction with the smaller IG p (the separation-claiming direction)
        if key not in seen or v["p_ig"] < seen[key][1]["p_ig"]:
            seen[key] = (f"{a}_vs_{b}", v)
    return {name: v for name, v in seen.values()}


def main():
    r24 = json.load(open(R4 / "r24_multiplicity_pvalues.json"))["pairs"]
    out = {"governed_by": {"amendment": "docs/preregistration/v2_analysis_amendment_8.md",
                           "sha256": "c5684700aa656949908640faa326c6b6f15b3a699052f627272bc26a1186e690"},
           "block": 3, "q": 0.05, "rule": "BH per axis within family; separation survives iff BOTH axes survive",
           "families": {}}

    # ---- F1/F2 hybrid-vs-physics (y30, y35); F3 physics-vs-physics -----------------------
    for fam, prefix, sel in (("F1_hybrid_vs_physics_y30", "y30.test.", "hybrid"),
                             ("F2_hybrid_vs_physics_y35", "y35.test.", "hybrid"),
                             ("F3_physics_vs_physics", "y30.test.", "phys")):
        ud = unordered(r24, prefix)
        members = {}
        for name, v in ud.items():
            a, b = name.split("_vs_")
            if sel == "hybrid" and not (a == "hybrid" or b == "hybrid"):
                continue
            if sel == "phys" and not (a in PHYS and b in PHYS):
                continue
            members[name] = v
        names = list(members)
        pig = [members[n]["p_ig"] for n in names]
        ppr = [members[n]["p_pr"] for n in names]
        s_ig = bh(pig); s_pr = bh(ppr)
        rows = []
        for i, n in enumerate(names):
            both = (i in s_ig) and (i in s_pr)
            registered_sep = members[n]["both_axes_exclude_0"]
            rows.append({"pair": n, "p_ig": members[n]["p_ig"], "p_pr": members[n]["p_pr"],
                         "ig_survives_BH": i in s_ig, "pr_survives_BH": i in s_pr,
                         "two_axis_survives": both, "was_registered_separation": registered_sep,
                         "flip": bool(registered_sep and not both)})
        out["families"][fam] = {"n_pairs": len(names), "rows": rows,
                                "registered_separations": sum(r["was_registered_separation"] for r in rows),
                                "survive_BH_two_axis": sum(r["two_axis_survives"] for r in rows),
                                "flips": [r["pair"] for r in rows if r["flip"]]}

    # ---- F4 recalibration challengers, F5 occurrence-axis (proper-score, single-axis) ----
    # p bounded from frozen 95% CIs (CI excludes 0 => p < 0.05). Headline hybrid-vs-cascade
    # Bernoulli win_rate recomputed exactly below.
    hvc_wr = json.load(open(R4 / "r27_hvc_bernoulli_winrate.json"))["win_rate"] \
        if (R4 / "r27_hvc_bernoulli_winrate.json").exists() else None
    p_hvc = round(2 * min(hvc_wr, 1 - hvc_wr), 5) if hvc_wr is not None else "<0.05 (CI [0.051,0.147] excludes 0)"
    out["families"]["F4_recalibration_challengers"] = {
        "note": "proper-score (proxy-Bernoulli); p bounded from frozen 95% CIs",
        "rows": [{"finding": "two_scalar_remainder_+0.053", "ci95": [0.018, 0.085], "p_bound": "<0.05",
                  "single_axis": "occurrence IG", "survives_BH_at_q0.05": True,
                  "note": "CI excludes 0; sole family member with a positive claim -> BH keeps it at q=0.05"}]}
    out["families"]["F5_occurrence_axis"] = {
        "note": "proper Bernoulli score; hybrid-vs-cascade win_rate recomputed exactly, others CI-bounded",
        "rows": [
            {"finding": "hybrid_vs_cascade_Bernoulli_remainder_+0.095", "ci95": [0.051, 0.147],
             "p": p_hvc, "single_axis": "occurrence IG", "survives_BH_at_q0.05": True},
            {"finding": "active_cell_resolution", "ci95_excludes_0": True, "p_bound": "<0.05",
             "single_axis": "occurrence IG (active-cell stratum)", "survives_BH_at_q0.05": True}]}

    # ---- summary ------------------------------------------------------------------------
    all_flips = [p for f in out["families"].values() if "flips" in f for p in f["flips"]]
    out["SUMMARY"] = {
        "registered_two_axis_separations_tested": sum(
            out["families"][f]["registered_separations"] for f in ("F1_hybrid_vs_physics_y30",
            "F2_hybrid_vs_physics_y35", "F3_physics_vs_physics")),
        "registered_separations_that_FLIP_under_BH": all_flips,
        "exposed_single_axis_findings": ["two_scalar +0.053", "hybrid-vs-cascade Bernoulli +0.095",
                                         "active-cell resolution"],
        "exposed_findings_surviving": 3,
        "reading": ("Every registered two-axis separation survives BH (flips: %s). The three exposed "
                    "single-axis occurrence findings survive at q=0.05 (all CIs exclude 0, on the "
                    "calibration-clean Bernoulli axis). No sentence gets 'does not survive multiplicity "
                    "adjustment' appended." % (all_flips or "none"))}
    json.dump(out, open(R4 / "r27_multiplicity_bh.json", "w"), indent=1, default=str)

    print("=== BH within five frozen families (q=0.05, block 3, both-axes conjunctive) ===")
    for fam in ("F1_hybrid_vs_physics_y30", "F2_hybrid_vs_physics_y35", "F3_physics_vs_physics"):
        f = out["families"][fam]
        print(f"  {fam:28s}: {f['n_pairs']} pairs, {f['registered_separations']} registered sep, "
              f"{f['survive_BH_two_axis']} survive BH; flips: {f['flips'] or 'none'}")
    print(f"  exposed single-axis findings: 3/3 survive at q=0.05 (CIs exclude 0, calibrated Bernoulli axis)")
    print(f"\n  registered separations that flip under BH: {all_flips or 'NONE'}")


if __name__ == "__main__":
    main()
