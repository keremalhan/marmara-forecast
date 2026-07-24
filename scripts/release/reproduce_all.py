"""reproduce-all — assert every number the manuscript rests on against its archived artifact.

WHY THIS EXISTS. Six stale-or-unsupported items reached, or nearly reached, print in this project,
and every one was caught the same way: re-run a procedure and compare it to its stored output.

  1. the GNSS placebo battery          (stale config)
  2. csep_eval's hardcoded b = 1.2     (stale literal)
  3. the Table S4 calibration battery  (v1.1.0 lineage)
  4. the b_op sweep itself             (superseded simulator)
  5. the corrected argmins 1.05/1.00   (asserted in an amendment, never measured -> r18)
  6. the deconfounded triple + test-era split (same -> r19; the test-era triple did not reproduce)

Re-run-and-compare is the check that finds this class. Automating it retires the class. This target
is the automation: it reads every claim the manuscript makes that is backed by an artifact, asserts
the artifact still says it, prints the manifest, and exits nonzero on ANY missing key, missing file,
or changed value. It does not re-simulate (that is hours); it verifies the archive is internally
consistent and that the manuscript's printed numbers still match it.

  --manifest      print the manifest and exit 0
  --mutate KEY    corrupt one checked value in memory and confirm the gate FAILS (self-test)

Run: PYTHONPATH=src MARMARA_ROOT=. <venv>/bin/python scripts/release/reproduce_all.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
R = ROOT / "results"
R4 = R / "round4"
PAPER = ROOT / "paper" / "paper_seismica.md"
SUPP = ROOT / "paper" / "supplement_seismica.md"


def load(p):
    f = Path(p)
    if not f.exists():
        return None
    try:
        return json.load(open(f))
    except Exception:
        return None


def dig(d, keys):
    """Walk an explicit list of keys. Dotted strings cannot be used: real keys here contain
    dots ('1.15', 'modern_3.0_to_3.5') and brackets ('subsample_[::3]'), so a dotted parser
    silently reports MISSING-KEY on keys that exist -- a gate that cries wolf is worthless."""
    cur = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        elif isinstance(cur, list) and isinstance(k, int) and k < len(cur):
            cur = cur[k]
        else:
            return None
    return cur


# (label, artifact, [key path], expected, tolerance)  -- expected None => key must merely exist
CHECKS = [
    # --- the b_op forensics (S10) --------------------------------------------------------
    ("S10.2 corrected argmin, [::3] subsample", "round4/r18_bop_argmin_measured.json",
     ["subsample_[::3]", "argmin_plain"], 1.05, 0),
    ("S10.2 corrected argmin, all 231 windows", "round4/r18_bop_argmin_measured.json",
     ["all_windows", "argmin_plain"], 1.00, 0),
    ("S10.2 corrected slope at the convention 1.15", "round4/r18_bop_argmin_measured.json",
     ["subsample_[::3]", "per_b", "1.15", "slope"], 1.1524, 1e-3),
    ("S10.3 deconfounded, modern 3.0->3.5", "round4/r19_deconfounded_b.json",
     ["deconfounded_estimates", "modern_3.0_to_3.5", "b"], 1.3655, 1e-3),
    ("S10.3 deconfounded, all pre-test 3.0->4.0", "round4/r19_deconfounded_b.json",
     ["deconfounded_estimates", "all_pretest_3.0_to_4.0", "b"], 1.4053, 1e-3),
    ("S10.3 deconfounded, catalogue 3.5->4.5", "round4/r19_deconfounded_b.json",
     ["deconfounded_estimates", "catalogue_3.5_to_4.5", "b"], 1.4177, 1e-3),
    ("S10.4 test era, pre-mainshock", "round4/r19_deconfounded_b.json",
     ["test_era_split", "pre_mainshock", "b_eff_3.0_to_3.5"], 1.2981, 1e-3),
    ("S10.4 test era, post-mainshock", "round4/r19_deconfounded_b.json",
     ["test_era_split", "post_mainshock", "b_eff_3.0_to_3.5"], 1.1374, 1e-3),
    ("S10.4 Md-conversion era slope", "round4/r19_deconfounded_b.json",
     ["era_effective_slopes", "md_conversion_era_2003_2012", "b_eff_3.0_to_3.5"], 0.8402, 1e-3),
    ("S10.4 modern-era slope", "round4/r19_deconfounded_b.json",
     ["era_effective_slopes", "modern_ml_era_2013_2021", "b_eff_3.0_to_3.5"], 1.3581, 1e-3),
    # --- the arms (S10.5, Table 3) -------------------------------------------------------
    ("S10.5 zero verdict changes at the arm endpoints", "round4/r15_table3_arms.json",
     ["prediction_held"], True, 0),
    ("S10.5 frozen set restored and verified", "round4/r15_table3_arms.json",
     ["restore_verified"], True, 0),
    ("S10.5 h(cascade) b-invariance spread", "round4/r16_arm_diagnostics.json",
     ["item2_h_flatness", "spread"], 0.0025, 1e-4),
    ("S10.5 Bernoulli arm b=1.00", "round4/r16_arm_diagnostics.json",
     ["arms", "1.00", "bernoulli_companion_K2000", "hybrid_vs_cascade_ig"], 0.1102, 1e-3),
    ("S10.5 Bernoulli arm b=1.15", "round4/r16_arm_diagnostics.json",
     ["arms", "1.15", "bernoulli_companion_K2000", "hybrid_vs_cascade_ig"], 0.1074, 1e-3),
    ("S10.5 Bernoulli arm b=1.40", "round4/r16_arm_diagnostics.json",
     ["arms", "1.40", "bernoulli_companion_K2000", "hybrid_vs_cascade_ig"], 0.1019, 1e-3),
    # --- live products (section 4/5) -----------------------------------------------------
    ("section 4 live P(M>=6) at the deconfounded law b=1.40", "round4/r17_live_b140.json",
     ["deconfounded law (1.40)", "P_M6"], 0.0030, 1e-4),
    # --- lineage -------------------------------------------------------------------------
    ("lineage: convention arm rebuilds the shipped grid bit-identically",
     "round4/claims_sensitivities.json", ["b_arm_lineage", "arms", "1.15", "bit_identical_to_shipped"], True, 0),
    ("lineage: b=1.00 arm is a genuinely different grid",
     "round4/claims_sensitivities.json", ["b_arm_lineage", "arms", "1.00", "bit_identical_to_shipped"], False, 0),
    # --- the frozen record ---------------------------------------------------------------
    ("pre-registration chain: recorded hashes verify (RUNPLAN excepted)",
     "audit/preregistration_chain.json", ["n_verified"], 8, 0),
    ("pre-registration chain: RUNPLAN discrepancy is documented",
     "audit/preregistration_chain.json", ["runplan_discrepancy", "actual"],
     "979b0b310f0aad6b74a0af93e4fac6206a28f43a4df529605de093cc9164773d", 0),
    ("registered b_op of record", "etas/etas_fit_report.json", ["operational_b_for_cascade"], 1.15, 0),
    ("stored sweep retained for forensics (Table S6)", "etas/etas_fit_report.json",
     ["b_calibration", "1.15", "slope"], 1.0131, 1e-3),
]

# manuscript strings that must still be present (or absent) after the rewrite
PROSE = [
    (PAPER, "b_op = 1.15 is an operational convention, not a\ncalibration", True,
     "section 3 states the convention framing"),
    (PAPER, "calibration-consistent", False, "the b_op-was-calibrated claim is gone from section 2"),
    (PAPER, "pre-specified pre-test calibration optimum", False,
     "Table 3's caption no longer calls 1.15 an optimum"),
    (PAPER, "operational calibration", False, "no 'operational calibration' phrasing survives"),
    (SUPP, "## Section S10.", True, "Section S10 exists"),
    (SUPP, "sequence-state-dependent", True, "S10.4 carries the approved wording"),
    (SUPP, "STAI-artifacted", False, "the withdrawn wording never entered the manuscript"),
    # --- back matter and captions: where round-8's defects actually sat. The body survived the
    # rewrite almost clean; the version note, Zenodo entry, supplementary list, data-availability
    # paragraph and captions did not. A gate that only checks the body checks the wrong pages.
    (PAPER, "current release 1.2.0", True, "version note is at 1.2.0"),
    (PAPER, "(Version 1.2.0) [Software]", True, "Zenodo reference is at 1.2.0"),
    (PAPER, "(Version 1.3.0) [Software]", False, "no stale 1.3.0 Zenodo version survives"),
    (PAPER, "the operational-b forensics (S10)", True, "supplementary list announces Section S10"),
    (PAPER, "release changelog (S7)", True, "supplementary list announces the merged S7"),
    (PAPER, "Tables S16–S17 (the operational-b sweep variants", True, "supplementary list announces Tables S16-S17"),
    (PAPER, "The three files are the registered count-scored file", False,
     "the mangled data-availability duplication is gone"),
    (PAPER, "extended-b sensitivities, exploratory by designation", True,
     "data availability states which claims file licenses the S10 arms"),
    (PAPER, "+0.101 to +0.109 at K = 2000", False, "the retracted estimator-mixed triple is gone"),
    (PAPER, "exact 95% upper bound on its precision", False, "the one-sided CP phrasing is gone"),
    (PAPER, "(Clopper\u2013Pearson)", True, "the CP interval is named"),
    (PAPER, "was calibrated on pre-test windows", False, "the last b_op story-fork is gone"),
    (PAPER, "four evaluation\ncampaigns", True, "the exposure ledger counts four campaigns"),
    (PAPER, "three evaluation\ncampaigns", False, "no stale campaign count survives"),
    (PAPER, "the ~0.4%\ntime-independent monthly base rate", False,
     "the gain endpoints no longer mix bases"),
    (SUPP, "which overlap threefold", False,
     "the false overlap claim is gone (step = horizon = 30 d: the windows tile)"),
    (SUPP, "contiguous non-overlapping 30-day tiles", True, "Table S6 states the tiling correctly"),
    (SUPP, "Exp is the matching expectation", True, "Table S4 defines Exp, not just Obs+"),
    # --- caught only by reading the rendered supplement, not by any grep ---
    # Gap closure (inconsistency #2): stale release-1.1.0 rare-product calibration values must not
    # reappear in the M>=4.5 Results paragraph; the config-of-record values (0.86 / 0.96) must be present.
    (PAPER, "0.75 at M≥5", False, "stale rare-product obs/exp (0.75) is gone from the M>=4.5 paragraph"),
    (PAPER, "0.84 at M≥5.5", False, "stale rare-product obs/exp (0.84) is gone"),
    (SUPP, "0.86 for P(M≥5)", True, "config-of-record rare-product obs/exp (0.86) is present (S15)"),
    (SUPP, "0.96 for P(M≥5.5)", True, "config-of-record rare-product obs/exp (0.96) is present (S15)"),
    (SUPP, "the sweep works only because the target sits", False,
     "S9.2's false 'the sweep works at Mc 3.0' claim is gone (it contradicted S9.2's own close)"),
    (SUPP, "The objective fails at both base Mc; only the mechanism differs", True,
     "S9.2 carries the both-Mc reconciliation"),
    (SUPP, "renewcommand", False, "no stray LaTeX leaked into the supplement source"),
    # The deconfounded triple CONVERGES (1.366/1.405/1.418) -- citing it as evidence of
    # heterogeneity is backwards, and a referee would say so. Caught by reading page 59.
    # The b(x,t) motivation must not cite era/threshold movement as physical evidence: S10.3 proves
    # the deconfounded triple CONVERGES and the raw movement is the conversion pile + rate error.
    # Doing so contradicts S10.3 two sections away -- the same shape as the S9.2 contradiction.
    (PAPER, "three deconfounded estimates spanning era, threshold and sequence state", False,
     "b(x,t) motivation does not cite the convergent deconfounded triple as heterogeneity evidence"),
    (PAPER, "effective b measured on this catalogue moves with era, with threshold and with sequence state", False,
     "b(x,t) motivation does not lump era/threshold movement in as physical evidence (contradicts S10.3)"),
    (PAPER, "era- and threshold-dependence is *not* part of that case", True,
     "b(x,t) motivation explicitly excludes era/threshold movement from the physical case"),
    (PAPER, "spatial variation is already\nmeasured rather than hypothesized", True,
     "b(x,t) physical case rests on the measured b_pos spatial field and sequence-state dependence"),
    # The live M6 gain factors and spans must come from the forecast-of-record range [0.13%, 3.74%]
    # (base 0.349%), NOT the r17 K=10000 recompute (3.83%/0.10% -> 11.0x/0.29x, span 38.7). The author's
    # cross-check found the two lineages split across sections -- exactly the stale-lineage class this
    # gate exists to assert, and these numbers were not previously in it.
    (PAPER, "ranges from 10.7× at b = 1.02 down to 0.37×", True,
     "S5 live-M6 gains use the forecast-of-record range, not the r17 recompute"),
    (PAPER, "11.0× at b = 1.02 down to 0.29×", False,
     "the stale-lineage M6 gains (11.0x/0.29x) are gone from S5"),
    (SUPP, "a b-ensemble span of a factor 28.8", True,
     "S6 b-ensemble span matches the record (28.8 = 3.74/0.13), not the stale 38.7"),
    (SUPP, "factor 38.7", False, "the stale 38.7 span is gone from the supplement"),
    (SUPP, "38.7× span", False, "the stale 38.7 span (dwarfs sentence) is gone from the supplement"),
    (SUPP, "factor of 26–102 across horizons", True,
     "S6 dominates-factor matches the main text (26-102), not the stale 28-102"),
    (SUPP, "28–102", False, "the stale 28-102 dominates-range is gone from the supplement"),
    # Item 2: the subsample slope (1.152) pairs with the subsample argmin (1.05), never the full-set argmin.
    (PAPER, "optimum moves to 1.05 on the registered subsample", True,
     "S3 pairs the subsample slope 1.152 with the subsample argmin 1.05, not the full-set argmin"),
    (PAPER, "1.152 and its optimum moves to 1.00.", False,
     "the S3 Table-S6 column conflation (subsample slope, full-set argmin) is gone"),
    # Item 4: 0.428 was correct from unrounded scalars but unreproducible from the printed 0.46/0.40.
    (PAPER, "rate-weighted mean, 0.428", False,
     "the false-precision 0.428 (unreproducible from printed s_bg/s_trig) is gone"),
    # Item 5: the caption promises outcomes, not four quantiles the text does not print.
    (PAPER, "its test outcomes are given in the text", True,
     "Figure 3 caption does not over-promise hybrid quantiles"),
    (PAPER, "its quantiles are given in the text", False,
     "the over-promising 'quantiles' caption wording is gone"),
    # Item 6: the 231/26/258/262 window accounting is reconciled where 258 is anchored.
    (PAPER, "one straddling the 2024-01-01 boundary that the b_op sweep excludes", True,
     "the 231 + straddler + 26 = 258 (and +4 = 262 stored) window accounting is stated"),
]


def main():
    argv = sys.argv[1:]
    mutate = None
    if "--mutate" in argv:
        mutate = argv[argv.index("--mutate") + 1]
    manifest_only = "--manifest" in argv

    rows, failures = [], []
    for label, art, key, exp, tol in CHECKS:
        d = load(R / art)
        if d is None:
            rows.append(("MISSING-FILE", label, art, None, exp))
            failures.append(f"{label}: artifact {art} missing or unreadable")
            continue
        got = dig(d, key)
        if mutate and mutate in label:
            got = (not got) if isinstance(got, bool) else (
                (got + 1.0) if isinstance(got, (int, float)) else "MUTATED")
        if got is None:
            rows.append(("MISSING-KEY", label, art, None, exp))
            failures.append(f"{label}: key {key} absent from {art}")
            continue
        if exp is None:
            rows.append(("PRESENT", label, art, got, "(exists)"))
            continue
        ok = (got == exp) if (isinstance(exp, bool) or tol == 0) else (
            isinstance(got, (int, float)) and abs(got - exp) <= tol)
        rows.append(("OK" if ok else "CHANGED", label, art, got, exp))
        if not ok:
            failures.append(f"{label}: {art}:{key} = {got!r}, manuscript says {exp!r}")

    for path, needle, want, why in PROSE:
        if not path.exists():
            failures.append(f"prose: {path.name} missing"); rows.append(("MISSING-FILE", why, path.name, None, want)); continue
        # Whitespace-insensitive: needles that span a line break ("the sweep works\nonly because")
        # evaded three literal greps in this project. Collapse runs of whitespace on both sides.
        hay = re.sub(r"\s+", " ", path.read_text())
        present = re.sub(r"\s+", " ", needle) in hay
        if mutate and mutate in why:
            present = not present
        ok = (present == want)
        rows.append(("OK" if ok else "PROSE-DRIFT", why, path.name, present, want))
        if not ok:
            failures.append(f"prose: {why} -- '{needle[:40]}' present={present}, expected {want}")

    w = max(len(r[1]) for r in rows) + 2
    print(f"{'status':<13} {'claim':<{w}} {'artifact':<38} {'value':>12}  expected")
    print("-" * (13 + w + 38 + 26))
    for st, label, art, got, exp in rows:
        g = f"{got:.4f}" if isinstance(got, float) else str(got)
        print(f"{st:<13} {label:<{w}} {str(art):<38} {g:>12}  {exp}")
    print("-" * (13 + w + 38 + 26))
    n_ok = sum(1 for r in rows if r[0] in ("OK", "PRESENT"))
    print(f"{n_ok}/{len(rows)} checks pass  |  {len(CHECKS)} artifact assertions, {len(PROSE)} prose assertions")

    # freeze-manifest check: every Amendment-8 prose-bound number still matches its source artifact
    import subprocess
    mf = subprocess.run([sys.executable, str(ROOT / "scripts/release/freeze_manifest.py"), "--check"],
                        capture_output=True, text=True,
                        env={"PYTHONPATH": "src", "MARMARA_ROOT": ".", "PATH": "/usr/bin:/bin"},
                        cwd=str(ROOT))
    mf_ok = mf.returncode == 0
    print(f"freeze-manifest: {mf.stdout.strip().splitlines()[-1] if mf.stdout.strip() else 'no output'}")
    if not mf_ok:
        failures.append("freeze-manifest drift: " + mf.stdout.strip().replace("\n", "; "))

    if manifest_only:
        return 0
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(f"  ! {f}")
        print(f"\nreproduce-all: FAIL ({len(failures)} failure(s))")
        return 1
    print("\nreproduce-all: PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
