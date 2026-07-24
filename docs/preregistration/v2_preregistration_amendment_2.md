# v2 pre-registration — Amendment 2 (2026-07-14)

Governs alongside the base protocol (`docs/v2_preregistration.md`, SHA-256
`377ed43e3f63f175eed1928d0f4b69b28b3caafcf0307c9c01e16bbfd9f9105c`) and Amendment 1
(2026-07-13). This is an **append**, not a rewrite. It records two corrections to prior claims
surfaced by an independent adversarial pre-submission review, so that neither a withdrawn
pre-registered item nor a corrected factual claim vanishes silently from a protocol whose
integrity rests on the append-only hash chain.

## Trigger

An independent adversarial review (distinct from the audit machinery that produced v1.2.0)
flagged (a) the §4 "completeness-corrected count (1,428), N-test quantile 0.022" as un-reproducible
and internally inconsistent with the §2 completeness argument, and (b) the manuscript's §2
timezone origin story as inconsistent with the shipped pipeline.

## Amendment

**A. The pre-registered completeness-corrected-count sensitivity is WITHDRAWN.** Base protocol §4
("Completeness / observed count") registered, as a *footnoted sensitivity* (not the primary
observation), a GR-extrapolation completeness correction anchored on the [3.45, 4.5) band giving a
deficit ~45 and a corrected count ~1,428 with a b-uncertainty band. It is withdrawn for two
reasons:

1. **Un-reproducible.** Neither the value 1,428 nor its N-test quantile 0.022 is emitted by any
   script or `results/*` artifact; both appeared only in manuscript prose. A claim whose armor is
   machine-checkability cannot ship a number no artifact produces.
2. **Internally inconsistent.** A GR-roll-off completeness *correction* presupposes sub-Mc
   incompleteness that §2 denies for the ML-dominated 2024+ test period (modern ML-population
   completeness `mag_w` ≈ 2.72, below the M≥3.0 target). The correction and the completeness
   argument cannot both stand; the completeness argument is retained.

The **primary observation is unchanged**: raw on-grid M≥3.0 = **1,383**, with **M≥3.5 the
completeness-unambiguous anchor** (base protocol §4). The CSEP count verdict is now reported
against 1,383 only, with the clustered-dispersion caveat (the forecast mean sits ~6% below the
observed total; the clustered N-test admits 1,383 within its spread, where an analytic-Poisson
N-test would under-disperse and spuriously reject).

**B. The §2 timezone account is corrected (data-description correction, recorded for the chain).**
v1.1.0 described the KOERI bulletin as Türkiye local time requiring a three-hour offset to UTC.
Verification against USGS/NEIC authoritative UTC for 21 well-recorded pre-2016 anchors in both
daylight-saving regimes (4 winter, 17 summer; max |residual| 2.0 s / 2.7 s;
`scripts/verify/tz_anchors.py` → `results/tz_anchor_crosscheck.json`) shows the reviewed zeqdb
catalogue is **true UTC year-round**; `data/build_catalog.py` applies no offset to it. The
local→UTC conversion (daylight-saving-aware `zoneinfo('Europe/Istanbul')`) applies **only** to the
preliminary 2026 monthly feed (features-only, excluded from every metric). §2 is rewritten to this
verified account; the manifest records the anchor cross-check.

## Recorded justification

- Item A: `results/csep/pycsep_results.json` and `results/csep/*summary.md` score against 1,383
  only; no corrected-count artifact exists. §2's `mag_w` completeness argument (Mc_ML ≈ 2.72)
  contradicts the premise of a sub-Mc correction over the test period.
- Item B: `results/tz_anchor_crosscheck.json` (21 anchors, both DST regimes) and
  `data/build_catalog.py:42` (zeqdb parsed as UTC, no offset).

## Unchanged

The claim rule, bootstrap machinery (B = 2000, seed 42, mean block 3), the surgical-k cascade, the
1-SE gate, the primary M≥3.0 observation (1,383), the M≥3.5 completeness anchor, Amendment 1's
native-background `modern_etas` comparator, and all other base-protocol items are unchanged.
