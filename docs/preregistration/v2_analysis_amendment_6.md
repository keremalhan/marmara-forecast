# v2 pre-registration — Amendment 6 (2026-07-16)

## What this is

A **prospectively timestamped post-review analysis amendment**, written and hashed **before** the
analyses it specifies are run, and archived with Amendments 1–5. Same claim and same limits as
Amendment 5: it does not restore an untouched test set, and the exploratory exposure disclosed in §3
and §6 stands. This is the fourth campaign against that test set, and §3's evaluation-history
paragraph will say so.

Governs alongside the base protocol (`docs/v2_preregistration.md`, SHA-256
`377ed43e3f63f175eed1928d0f4b69b28b3caafcf0307c9c01e16bbfd9f9105c`) and Amendments 1–5. Append-only.

## Trigger: a fourth stale-lineage artifact, found by re-run-and-compare

Re-running the shipped `calibrate_b` procedure and diffing against its stored output established:

* The stored b_op sweep (`etas_fit_report.b_calibration`, Table S6) reproduces **exactly** under
  `preserve_branching=False` (pred 1340.8 / 1222.5, slopes 0.9221 / 1.0131 at b = 1.10 / 1.15). It
  was computed with the **supercritical** simulator and never re-run after the v1.2.0 branching fix.
  The fix modified `cascade.py` and `grid_hybrid.py`; `calibrate_b.py` was not touched, so nothing
  forced a re-run, and `grid_hybrid` read the stale value through `load_b_op()`.
* Control: at b = 1.542 ≈ `params.b` the rescale is a near-no-op (factor 0.9997) and the two
  simulators agree to 9.1e-05 in slope. The divergence appears only where the rescale fires.
* **Scope is bounded**: the shipped `grid_hybrid.parquet` **is** `preserve_branching=True`
  (bit-exact against a pb=True re-simulation on windows 100 and 248). The simulator was never
  supercritical downstream. **Only the b_op value is stale.**
* Re-running the procedure at the corrected simulator selects **b_op = 1.05** on the registered
  `[::3]` subsample and **b_op = 1.00** on all 231 pre-test windows.

## Why we do not simply rebuild at 1.00

Two facts, established before this amendment was written, make 1.00 suspect rather than clean.

1. **1.00 is below every magnitude-distribution estimate of b** (Aki 1.019, b-positive 1.542), so a
   central forecast at b_op = 1.00 would sit at or outside the boundary of the b-ensemble that is
   supposed to bracket it. Redefining the ensemble to contain the value the calibration just
   produced would be adjusting a gate to fit a result, which this project has refused elsewhere and
   refuses here.
2. **The calibration objective is contaminated by the catalogue defect §2 already documents.** The
   pre-test catalogue's own effective 3.0→3.5 slope is **0.990**; the corrected full-window sweep
   returns **1.00**. It is reproducing the calibration era's slope, not discovering a property of the
   process. The era is not homogeneous: the Md-conversion era (2003–2012) has an effective slope of
   **0.840** and a M≥3.5 rate of **344.7/yr**, against **1.358** and **115.5/yr** for the modern
   ML-picked era (2013–2021) and **1.167** for the test era. The mixture piles ~5,000 Md events near
   `mag_w` 3.45–3.65, straddling the 3.5 threshold the objective counts.

   The two bugs partially cancelled: `preserve_branching=False` inflated predicted counts ~14% while
   `[::3]` under-realized ~11%, landing the stale sweep at 1.15 — near the test era's 1.167. The
   corrected-but-contaminated sweep moves *away* from the era the products actually run in.

**Note on attribution.** This is rate non-stationarity plus a magnitude-conversion artifact. It is
**not** the within-cell multiplicity deficit: multiplicity is the arrangement of events inside cells,
the calibration objective is Σλ, and the two are orthogonal. An earlier draft of this reasoning
conflated them; the conflation is withdrawn.

**Note on precision versus accuracy.** The slope's seed variability is sd ≈ 0.002 (three seeds, full
windows), while the argmin moves a full grid step between window subsets ([::3] → full: 1.05 → 1.00).
The sweep is a precise measurement of a contaminated quantity. Seed separation is not evidence of
correctness and will not be reported as such.

## The two diagnostics (specified before running)

**D1 — era-split sweep.** Corrected simulator, full windows, restricted to **modern-era pre-test
windows (t0 ≥ 2013-01-01)**, full candidate list, seeds {5000, 6000, 7000}.
*Prediction on record: the argmin returns to ≈1.12–1.15.* If it does, contamination is proven.

**D2 — threshold-split sweep.** The **M≥4.0-count** calibration that §6 already names in print as the
repair, on full pre-test windows, corrected simulator, seeds {5000, 6000, 7000}. The 3.45 pile sits
far below 4.0, and within-cell multiplicity at M≥4.0 is ≈1, which also removes the count/occupancy
ambiguity.
*Prediction on record: ≈1.3.* Anything ≫ 1.00 proves threshold-dependence.

## Decision rule (fixed now, before the answers)

* **If the modern-era argmin (D1) ≥ b_Aki = 1.0186**, it becomes the operational calibration of
  record, and the pipeline is rebuilt at that value.
* **Otherwise**, the pipeline is rebuilt at the corrected full-window argmin (1.00), and the
  sub-Aki value is written up as a genuine count-versus-magnitude-law tension, attributed to rate
  non-stationarity — not to multiplicity.
* Either way, **every sweep variant is reported** in the revised Table S6: stale (supercritical,
  `[::3]`), corrected-full-window, era-split, and threshold-split, with their argmins side by side.

**We own the procedure change.** Restricting the calibration era is a change to a registered
procedure made after seeing a result, and we say so in those words. Three mitigations, stated rather
than implied: the trigger was a bug fix, not a disliked number; the justification is *verbatim* the
argument §2 already makes for using Mc = 3.0 against the max-curvature 3.65 — the modern ML-picked
population dominates the test period, and a calibration should follow the same population its targets
do; and every variant is reported, so the reader can apply a different rule and see what it gives.

## Pre-flight before any rebuild (binding)

Before six hours of cascade re-simulation, re-score at the candidate b through the existing b-arm
machinery: (i) the cascade's **CSEP M-test**, and (ii) **y35 test obs/exp**. Rationale: drawing
magnitudes at a b below the test era's empirical near-threshold slope (1.167) makes the simulated
magnitude distribution too heavy-tailed, so the M-test pass (γ = 0.25) is at genuine risk, and y35
test obs/exp is expected to swing from ~1.03 toward over-prediction. **Calibrating on contaminated
pre-test data and then failing the clean test era — discovered after the rebuild — is the worst
available outcome, and this pre-flight exists to prevent it.** If the pre-flight fails at the
decision rule's chosen b, execution stops and the result is reported before anything is rebuilt.

## Rebuild specification (whatever b is chosen)

* w re-selected by the registered 1-SE parsimony rule on validation; no other hyperparameter moves.
* Seed policy: per-window seed 5000+i for calibration, 1000+k for the grid, as shipped; the seed
  triple {5000, 6000, 7000} is used only to report the sweep's seed variability, never to select.
* **Table 3's arms extend downward to {1.00, 1.05, argmin, 1.15}** regardless of outcome. The
  model-comparison verdicts are paired at common b and already stable across 1.10–1.20; showing
  stability across the whole contested range insulates the thesis from the calibration decision.
* Unconditional reporting; no gates.
* Release **1.3.0**. §3's evaluation history, §6, the S7 changelog, Table S8, both claims files and
  the Zenodo version all update.

## Blast radius, stated so it cannot be mistaken

This is a **calibration repair to the magnitude law**, and its blast radius is the **rare-event
products** (y45, P(M≥5/5.5/6), the live products, Table S5) and the honesty of the magnitude law.
It is **not** a retraction of the ML-versus-ETAS result: those verdicts are paired comparisons at a
common b_op, already stable across the swept range, and will be shown stable across the extended
range. Every section written under this amendment must keep that distinction visible.

## Durable fix

A `reproduce-all` target that regenerates every `results/` artifact from shipped code and diffs
against the archive. Four stale-lineage items (the GNSS battery, `csep_eval`'s hardcoded b, the
Table S4 battery, and now the b_op sweep) were each found by re-running a procedure and comparing to
its stored output. Re-run-and-compare is the check that finds this class; automating it retires the
class.
