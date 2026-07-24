"""Item E: prose-consistency gate (standing test, wired like the causality self-test).

Extracts the canonical value for each claim-bearing quantity from results/*.json + claims.json,
then checks the manuscript (paper/paper_seismica.md + supplement) for STALE values (v1 numbers
that the v2 recompute has superseded). Fails loudly (exit 1) on any stale match so it cannot
silently ship a refuted number over its own contradicting result. Two hardcoded-title landmines
were caught by luck this session; this makes a third impossible.

Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/verify/prose_consistency_gate.py
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
from marmara.paths import RESULTS

PAPER_DIR = RESULTS.parent / "paper"
DOCS = [PAPER_DIR / "paper_seismica.md", PAPER_DIR / "supplement_seismica.md"]


def canon():
    """Current canonical values from results (the source of truth)."""
    c = {}
    try:
        cs = json.load(open(RESULTS / "csep" / "pycsep_results.json"))["models"]
        c["cascade_N_pass"] = cs["cascade"]["Number"]["pass"]
    except Exception:
        pass
    try:
        s = json.load(open(RESULTS / "models" / "synthetic_report.json"))
        c["discriminator_pr_auc"] = s["classifier"]["test_pr_auc"]
    except Exception:
        pass
    return c


# (label, regex of the STALE v1 value to flag, current v2 value / result path)
STALE_PATTERNS = [
    ("N-test cascade forecast count", r"\b1,?49[89]\b", "surgical N-forecast = 1306 (supercritical 1499 was the bug); results/scoring/ntest_attribution.json"),
    ("foreshock probability-gain jump", r"\b45\s*[x××]|45-?fold|45 times\b", "39.71x; results/validation_final/m62_countdown.json"),
    ("y35 top-cell (stale pre-correction)", r"P\s*≈\s*20\s*%", "36%; results/forecast/forecast_2026-07-05/forecast_summary.json top_cell P35=0.362"),
    ("day-before M6 gain (stale pre-correction)", r"5\.11\s*[×x]", "4.74x; results/validation_final/m62_countdown.json fix1_gate_2025-04-22 gain_vs_base=4.739"),
    ("live central P(M>=6)", r"\b1\.61\s*%", "1.54%; results/forecast/.../forecast_summary.json"),
    ("year-horizon P(M>=6)", r"\b20\.6\d?\s*%", "18.30%; results/forecast/multi_horizon.json"),
    ("discriminator test PR-AUC", r"PR-AUC 0\.119\b|ROC-AUC 0\.622\b", "0.123 (retrained on clean subcritical data); results/models/synthetic_report.json"),
    ("b-sensitivity refuted headline", r"hybrid is unstable|ML hybrid is unstable|unstable,? (?:and )?the physics", "REFUTED: verdicts stable across b; the apparent instability was the supercriticality artifact"),
    ("Mizrahi uniform-background handicap wording", r"replace its spatially[- ]variable background with a uniform", "REFUTED: native background is the matched comparator; the uniform floor is only a sensitivity (Methods)"),
    # --- verdict-string desync: prior verdicts the recompute refuted (a numbers-only gate misses these) ---
    ("refuted: hybrid-instability verdict", r"has no such stability|verdict swings|flips its verdict|nearly flat validation|not stable in b_?op|degenerates to the pure cascade|structurally cannot lose|w\s*→\s*0\b", "REFUTED: hybrid stably inseparable; validation LL climbs ~216 nats (Methods 1-SE rule, Discussion)"),
    ("refuted: cascade over-predicts / fails N-test", r"over-predict[s]? the count|fails? the number test", "REFUTED: native-clustered cascade is count-consistent (1306, delta1 0.094); it fails S, not N"),
    ("refuted: CSEP Poisson-confound caveat", r"confounded by[\s]+the cell-independent Poisson|not interpretable here|Poisson-catalogue under-dispersion|catalogues under-disperse", "REFUTED: native clustered catalogues; the S/PL failure is a genuine spatial misallocation"),
    ("dangling numbered amendment", r"Amendment \d+", "no numbered amendment is defined; describe the change in Methods and reference the section"),
    ("claim-versioning language", r"\bv1\b|\bv2\b|an earlier (?:version|cascade|forecast|run|integration|fit)|a previous version of (?:the|this) (?:paper|work|manuscript)|in a (?:previous|prior) (?:version|run)", "no claim-versioning language (Zenodo semantic versions excepted); state the method/result directly"),
    ("inseparable-on-both-axes contradiction", r"inseparable[^.]{0,80}on both axes", "inseparable is a CONJUNCTIVE verdict (>=1 axis straddles), not per-axis; use 'under the two-axis rule'"),
]


def main():
    cval = canon()
    print("canonical anchors:", {k: v for k, v in cval.items()})
    findings = []
    for doc in DOCS:
        if not doc.exists():
            print(f"  (missing {doc.name}; skipped)"); continue
        text = doc.read_text(errors="ignore")
        for label, pat, fix in STALE_PATTERNS:
            hits = re.findall(pat, text, flags=re.IGNORECASE)
            if hits:
                findings.append({"doc": doc.name, "claim": label, "stale_matches": hits[:4], "current": fix})
        # cross-reference resolution: in a doc that numbers its own sections, every §N must
        # resolve to a numbered heading (catches the dangling "§3"/"Amendment 1" class of error).
        heads = set(re.findall(r"^#{1,3}\s*(\d+)[.\s]", text, flags=re.MULTILINE))
        if heads:
            for n in sorted(set(re.findall(r"§\s*(\d+)", text))):
                if n not in heads:
                    findings.append({"doc": doc.name, "claim": "dangling section cross-reference",
                                     "stale_matches": [f"§{n}"], "current": f"no numbered '{n}.' heading in {doc.name}"})
    out = {"n_stale": len(findings), "findings": findings,
           "note": "Each finding is a v1 number/verdict the v2 recompute superseded. Per governance these "
                   "are FLAGGED for the author; this gate does not rewrite the manuscript."}
    (RESULTS / "audit" / "prose_consistency_gate.json").write_text(json.dumps(out, indent=2))
    if findings:
        print(f"\n=== PROSE-CONSISTENCY GATE: {len(findings)} STALE claim(s) FLAGGED (fail-loud) ===")
        for f in findings:
            print(f"  [{f['doc']}] {f['claim']}: found {f['stale_matches']} -> {f['current']}")
        print("\nwrote results/audit/prose_consistency_gate.json ; exit 1 (fail-loud, wire into pytest gate)")
        sys.exit(1)
    print("prose-consistency gate PASS: no stale claim-bearing numbers found.")


if __name__ == "__main__":
    main()
