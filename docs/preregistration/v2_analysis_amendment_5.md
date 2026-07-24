# v2 pre-registration — Amendment 5 (2026-07-15)

## What this document is, and what it is not

This is a **prospectively timestamped post-review analysis amendment**. It is written, dated and
hashed **before executing** the analyses it specifies, and it is archived alongside the earlier
amendments. That is the whole of its claim.

It is **not** a pre-registration in the strong sense, and it does not restore an untouched test set.
The analyses below are prompted by review of results already seen. The exploratory exposure of the
test period disclosed in §3 and §6 stands undiminished; nothing here repairs it, and no reader should
treat these items as confirmatory. What the timestamp buys is narrower and worth exactly what it is:
the questions, the procedures, and the reporting commitment were fixed before the answers were known,
so the outcomes cannot have been selected after the fact.

Governs alongside the base protocol (`docs/v2_preregistration.md`, SHA-256
`377ed43e3f63f175eed1928d0f4b69b28b3caafcf0307c9c01e16bbfd9f9105c`) and Amendments 1–4. Append-only.

## Adjudicator of record (unchanged)

The count-scored conjunctive rule fixed before the final evaluation remains the adjudicator of
record; `results/claims.json` holds the verdicts. Nothing here re-adjudicates it.

## Reporting commitment (binding, and the point of the document)

**Every item below is reported unconditionally**: unchanged, weakened, strengthened, or reversed.
**No item carries a pass/fail gate, and none may acquire one.** A prior version of this plan wrote
"gate: every verdict unchanged," which defined success as confirmation of the conclusion already
drawn. That is outcome-dependent adjudication — the precise practice this paper exists to criticize —
and it is struck. The correct posture for a robustness analysis is that it has no preferred outcome:
if item A reverses a Table 2 verdict, the reversal is the finding and the manuscript changes to say
so.

Two workflow consequences follow. First, if any verdict in **A** or **F** differs from the shipped
tables, execution stops and the result is interpreted before any text is written. Second, **no item
is declared harmless in advance**; B, C and D are each permitted to change sentences in the
manuscript.

---

## A — Mc = 3.5 full-pipeline arm (y35 only)

**Question.** Do the Table 2 (M≥3.5) verdicts persist when *every* stage — features, labels, ETAS
fits, b_op calibration, ML training, and w-selection — is conducted above a stricter completeness
threshold, rather than only the labels being raised (Table S12)?

**Procedure.** Rebuild the grid at base Mc = 3.5. Refit the in-house first-generation ETAS and the
cascade at that threshold; include sv-ETAS if it is cheap to refit, and the independent inversion
optionally — if either is omitted, the omission is disclosed rather than passed over. Re-run the b_op
count-calibration sweep on the pre-test windows only. Retrain the booster on the new grid. Re-select
the blend weight w by the pre-registered 1-SE rule on validation. Run the truncated-catalogue
self-test on the new grid. Evaluate y35 over the 26 test windows on both axes, for all
hybrid-versus-physics and physics-versus-physics pairs.

**Scope, to be stated in the manuscript verbatim in substance.** This arm tests the M≥3.5 conclusion
under strict completeness. It does not and cannot validate the M≥3.0 primary target, which has no
existence at this floor. An unchanged Table 2 is therefore evidence about M≥3.5 and about nothing
else.

## B — Test-only calibration table

**Question.** How do the calibration products read when computed only over data that never informed
b_op?

**Procedure.** Recompute every Table S4 product over the 26 test windows alone (t0 ≥ 2024-01-22),
excluding every window that entered the b_op calibration. Report alongside the existing 2022→2026
battery rather than in place of it, whatever it shows.

## C — Branching-ratio profile

**Question.** What does the likelihood say about the branching ratio n, whose MLE sits at the 0.95
cap (an active bound, hence no interior curvature and no Hessian interval — §6, Table S11)?

**Procedure.** Constrained refits with all nuisance parameters re-optimized at fixed
n ∈ {0.80, 0.85, 0.88, 0.90, 0.92, 0.94}; the n = 0.95 and n = 0.999 fits already exist. Report
2[ℓ(n̂) − ℓ(n)] at each point.

**Naming rule, fixed now.** If the grid supports it, state a **one-sided lower bound** using the
boundary-adjusted likelihood-ratio criterion, noting the 50:50 mixture caveat that applies when the
optimum lies on an active bound. If any constrained refit fails to converge, or the criterion is
crossed between two coarse grid points rather than at one, the whole item is named a **constrained
branching-ratio sensitivity** and **no interval is stated**. We do not interpolate a bound out of a
six-point grid.

Propagate the n = 0.90 fit through the 30/90/365-day regional P(M≥6) as **one downside scenario**,
labelled as such, not as an uncertainty band.

## D — Alarm-duration ladder, correct denominators

**Question.** How long does a ≥40× alarm persist, among the triggers that actually cross ≥40×?

**Defect being corrected.** The shipped ladder computed ≥40× persistence over the 621 triggers that
escalate at ≥10×. Most of those never reach 40× at all, so they are recorded as "falling below 40×
within the hour" when they were never above it — an artifact of the wrong denominator, and the source
of the "two-thirds decay within an hour" sentence.

**Procedure.** Recompute the ≥40× ladder over the **217 triggers exceeding 40× at t + 10 min only**.
Leave the 621-denominator false-alarm and precision rows untouched: the two denominators answer
different questions — precision among all ≥10× alarms, versus persistence conditional on crossing
≥40× — and both remain in Table S10 with **the denominator named in every sentence that quotes a
number from it**. Confirm follow-up availability: the 30-day observability filter should make every
ladder rung ≤ +30 d observable for every trigger; state this, and state that durations are
interval-censored between rungs.

## E — Precision intervals

Clopper–Pearson 95% intervals for the precision 1/621 (trigger cell) and 1/627 (25-km
neighbourhood).

## F — Grouped-temporal early-stopping sensitivity (y30 and y35)

**Question.** Does the ML stage's tree count, and do the test verdicts, change when the early-stopping
split respects time rather than being drawn at random?

**Defect being corrected.** `early_stopping='auto'` selects the tree count on scikit-learn's internal
*random* 10% of training rows (Table S9). This is within-training and cannot leak the test period,
but a random split of temporally autocorrelated rows places near-duplicate neighbours on both sides,
so the stopping decision is made under optimistic conditions. A prior version of this plan asserted
that the bias "can only be toward more trees, and the 1-SE rule then shrinks w." That was an
intuition presented as a theorem: T1-5 does not establish that direction, and no result in this
repository does. The claim is withdrawn. We measure the direction instead of asserting it.

**Procedure.** Disable the internal random stop. Select the tree count on a temporal within-train
tail: fit on windows with t0 ≤ 2020-12-31, evaluate candidate n_trees ∈ {10, 25, 50, 100, 147, 200,
400} on the 2021 windows, and take the best. Refit on the full training window at the selected count.
Re-select w by the 1-SE rule on the true validation split. Evaluate the test verdicts.

**Report** the selected tree count, the selected w, and hybrid-versus-cascade and
hybrid-versus-first-generation on both axes — unconditionally, in whichever direction they move.
