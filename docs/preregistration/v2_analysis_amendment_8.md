# v2 pre-registration — Amendment 8 (2026-07-16) — release 1.4.0 (umbrella)

## What this is, and the honesty ledger inside it

A **prospectively timestamped post-review analysis amendment**, written and hashed **before** the
analyses it specifies are run, archived with Amendments 1–7. It does not restore an untouched test
set; the exploratory exposure disclosed in §3 stands. This is the fifth campaign against that test
set, and §3's evaluation-history paragraph will say so. Every analysis below is exploratory by
designation, adjudicates no registered verdict, and is reported **unconditionally** — every outcome,
whether it strengthens, weakens or reverses the reading, with no pass/fail gate.

Governs alongside `docs/v2_preregistration.md` (SHA-256
`377ed43e3f63f175eed1928d0f4b69b28b3caafcf0307c9c01e16bbfd9f9105c`) and Amendments 1–7.

**The honesty ledger.** Not everything below is prospective, and the document says which is which.
*Established before this was written* (reported as results, not lodged as predictions): the
deconfounded triple point estimates (Amendment 7 §2; `r19_deconfounded_b.json`), the algebraic
reduction of the calibration objective to a count estimator, and the exact binomial-interval formula
that reduction implies. *Genuinely prospective* (specified here, unrun): the binned-MLE and
b-positive headline estimators of item 1, every bootstrap, the P(M≥6) reference band, and the
analyses of items 3–7 and 9. Labelling settled arithmetic a "prediction" would be theatre.

---

## Item 1 — the magnitude law above the pile, with an interval; and the reduction

**Framing, fixed now so it cannot drift.** The crossing-point formula
`b = log₁₀(N₁/N₂)/ΔM` is **not a finding**: it is the classical two-threshold Gutenberg–Richter
count estimator (Utsu 1965-style), and it will be cited as classical, never presented as derived.
What *is* a finding, and stated as one, is the **reduction**: the corrected simulator-based b_op
calibration objective, once normalized by the b-independent M≥3.0 slope, cancels the cascade
identically and collapses to that classical count estimator. The b_op sweep therefore contained no
information beyond the catalogue's own counts — which is why it is unidentified at base Mc = 3.5 (the
rescale pins the counted quantity) and contaminated at base Mc = 3.0 (the conversion pile). This is
the same genre as the a(h) count-scoring identity (elementary math; the value is the application),
and it is framed the same way: *algebra, not evidence; the finding is what it reveals about the
procedure.*

**Headline estimators (prospective).** The defensible magnitude law above the Md-conversion pile,
estimated two independent ways on the population `mag_w ≥ 4.0` (safely above the ~3.45–3.65 pile):

1. **Binning-corrected MLE** (Aki 1965 / Utsu, with the 0.1 bin correction):
   `b = log₁₀(e) / (mean(M) − (Mc − ΔM/2))`, `Mc = 4.0`, `ΔM = 0.1`. Computed on the whole
   catalogue and on the modern era (`t ≥ 2013-01-01`).
2. **b-positive** (van der Elst 2021), the transient-robust estimator already in the pipeline
   (`marmara.catalog.b_positive`), over a **pre-specified δ sweep** `δ ∈ {0.1, 0.2, 0.3, 0.4, 0.5}`
   on the same `mag_w ≥ 4.0` population, whole and modern.
3. **Bootstrap CI per estimator**: event-resampling percentile CI (B = 2000, seed 42) for each.

**Consistency check (mixed).** The deconfounded triple (established point estimates; Amendment 7 §2)
now carries two intervals each:
* **Exact binomial**, from the reduction. Since `N₂` is a nested subset of `N₁`,
  `N₂ | N₁ ~ Binomial(N₁, 10^{−bΔM})`, giving `SE(b) = SE(p)/(p·ln10·ΔM)` in closed form:
  modern 3.0→3.5 (5,515/1,145) **1.366 ± 0.023**; all pre-test 3.0→4.0 (12,866/506)
  **1.405 ± 0.019**; whole catalogue 3.5→4.5 (5,024/192) **1.418 ± 0.031**.
* **Window-block bootstrap** over the same window-ids (B = 2000, seed 42), as the honest check on
  the binomial's iid-GR assumption (which is roughly right for *b* because clustering perturbs rates,
  not magnitude draws). **Report the wider of the two intervals.**

**Two caveats, fixed now.** (i) The three deconfounded estimates are **not averaged** — nested
samples, not independent draws; they are quoted as mutually consistent (they agree at ~1–2σ) or, if
they do not, the disagreement is reported. (ii) If the binned MLE and b-positive **disagree
materially** above the pile, that is a **finding, not a nuisance**: it is reported, the ensemble
stays primary, and neither δ nor the threshold is tuned toward agreement — tuning would recreate the
b_op selection pathology this release exists to retire.

**Propagation (prospective).** Take the headline b and its interval and propagate the interval
endpoints through the live cascade at t0 = 2026-07-05 to obtain a **reference band** for the 30-day
regional P(M≥6). This band is a labelled reference sitting inside the full b-ensemble range; it is
**not** a second central. The ensemble range remains the issued forecast.

## Item 3 — foreshock episode analysis (deduplicated denominator)

The t + 10 min false-alarm accounting (§4, Section S5) counts overlapping triggers as independent
alarms. Reconstruct the denominator as episodes.

* **Graph**: nodes = qualifying `mag_w ≥ 4.0` triggers in the model box; an edge joins two triggers
  whose 30-day / 25-km alarm cylinders overlap in space-time; **episodes = connected components**.
* **Report** (all unconditional, no predicted outcome lodged): raw trigger count; episode count;
  **episode precision** (episodes containing the one M≥6 / total episodes); **false-alarm ratio**
  (renamed from "rate" everywhere — it is a ratio, not a per-unit-time rate); capture rate; the
  **union-of-cylinders region-time in alarm** as the honest space-time footprint; the Molchan-diagram
  point; and **2–3 pre-specified sensitivities**: radius ∈ {20, 25, 30} km and duration ∈ {14, 30} d.
* No outcome is predicted; the episode count and precision are whatever they are.

## Item 4 — prospective forecast log, scored as issued

One main-text table from the append-only hashed prospective log. **As-issued scoring only.**

* Each entry is included only if: its hash timestamp **precedes** its t0; its configuration is
  recorded; and its 30-day target lies **fully inside the reviewed catalogue** (not the preliminary
  tail; `REVIEWED_END`).
* An entry issued under the superseded (pre-v1.2.0) configuration is scored **exactly as issued**,
  with a footnote naming the config. **No entry is ever rescored under the corrected simulator** —
  re-scoring a prospective issue retroactively is precisely the move this project forbids.
* Report per-entry: t0, issued config, issued P, realized outcome, as-issued score.

## Item 5 — post-freeze confirmation windows (review-lag gated)

Section §2 states the recent tail is preliminary. KOERI's reviewed bulletin lags months, so of the
four candidate windows (t0 = 2026-03-12 / 04-11 / 05-11 / 06-10) only those whose **30-day target
lies entirely within the reviewed catalogue as of the 2026-07-11 fetch** are eligible.

* Determine eligibility from `REVIEWED_END` first; **include only the clean windows**, however few
  (likely one or two). Do not force four.
* For each eligible window report per-window obs/exp and sign, explicitly labelled
  **retrospective, non-adjudicating** (these windows postdate the frozen test split; they confirm,
  they do not re-adjudicate any verdict).

## Item 6 — leave-Kumburgaz-out Bernoulli, self-contained

* **Exclusion rule (calendar, one definition, no alternatives)**: drop every window whose 30-day
  target **intersects 2025-04-23 … 2025-07-23** (the Kumburgaz sequence quarter).
* Compute the hybrid-vs-cascade edge as **per-window Bernoulli ΔIG directly from the stored K = 5000
  occupancies** (cheap; no re-simulation), on the surviving windows. Report the block-bootstrap CI
  (B = 2000, seed 42) and, alongside it, the **proper-score sign count** (windows with positive vs
  negative Bernoulli ΔIG). The entire small-n defence stays within the one proper score — the
  count-scored 25/26 figure is **not** borrowed.

## Item 7 — multiplicity, on the existing replicate arrays

* **Deduplicate** the pairwise comparisons to the **84 unique ordered pairs** actually adjudicated;
  group into **five families** (hybrid-vs-physics primary, hybrid-vs-physics M≥3.5, physics-vs-physics,
  recalibration challengers, occurrence-axis).
* **p-values by CI inversion** on the **existing B = 2000, seed-42 block-bootstrap replicate arrays**
  — the smallest α at which the percentile interval excludes zero. **Reuse the seed-42 replicates;
  invent no new bootstrap.**
* **Benjamini–Hochberg within family.** One supplement table, labelled **post-review sensitivity**.
* Report failures: if the +0.053 two-scalar remainder or the active-cell split does **not** survive
  BH, it is printed as not surviving.

## Item 9 — optional: Mc = 3.5 arm at the deconfounded b

Run **only if time allows**, and reported regardless of outcome. Rebuild the Mc = 3.5 arm at an
**imposed b = 1.38–1.40** (the deconfounded law from item 1), with the 1-SE w-reselection rule
**frozen as specified here**, no other hyperparameter moved. This tests whether the Mc = 3.5 arm's
verdicts — shown structurally b-tracking in Section S9.2 — settle when b is fixed at the crustal law
rather than imposed arbitrarily.

## Mechanics

Release **1.4.0**. Every new prose number added to `scripts/round4/reproduce_all.py`. S7c changelog;
Table S8 addendum; both claims files; Zenodo 1.4.0; PDF read front to back; submit. If the set
threatens to spiral, items 1–4 plus the wording pass ship, and items 5–7 and 9 carry to the
response-to-reviewers round as pre-specified analyses.
