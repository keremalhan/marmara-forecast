# v2 pre-registration — Amendment 3 (2026-07-15)

Governs alongside the base protocol (`docs/v2_preregistration.md`, SHA-256
`377ed43e3f63f175eed1928d0f4b69b28b3caafcf0307c9c01e16bbfd9f9105c`) and Amendments 1–2. Append-only.

## Adjudicator of record (unchanged)
The **pre-registered count-scored conjunctive rule remains the adjudicator of record**; the
machine-readable `results/claims.json` verdicts are the claims. Nothing here re-adjudicates the
registered rule. Re-scoring under a proper score chosen after seeing results would be score-shopping;
we therefore report BOTH, with the registered count-scored rule primary, and add a named-proper-score
analysis that explains what the registered IG axis was measuring and which verdicts are score-robust.

## Trigger
The round-3 battery showed the occurrence-Poisson information gain is inflated for count-calibrated
clustered forecasts by a closed-form ≈ h−1−ln h per model (h = Σλ/N_pos), so the registered IG axis
mixes a scoring artifact with skill. A proper binary score (Bernoulli/log-score; Serafini, Naylor,
Lindgren, Werner & Main, 2022) is needed to separate them.

## Amendment — the named-proper-score analysis (pre-specified before running)
1. **(i) Native occurrence forecasts.** Each model is scored at the forecast it actually states
   (CSEP philosophy): the simulator's native Monte-Carlo occupancy P̂(N≥1) from K=500 clustered
   catalogues for the clustered forecasters (cascade, sv-ETAS, and the hybrid via per-cell thinning
   of the cascade catalogues by λ_hybrid/λ_cascade), and 1−e^(−λ) for the first-generation
   intensities (first-generation ETAS, the independent inversion, Poisson, smoothed). Bernoulli
   log-score. The per-model-scalar occurrence version (1−e^(−sλ), s val-fit) is retained as a
   DIAGNOSTIC that isolates the non-global part of any residual.
2. **(ii) Shrinkage sensitivity.** With K=500, a realised cell at p̂=0 makes the log-score −∞; the
   headline verdicts must be shown insensitive across add-one, λ-blend, and two blend weights; any
   fragile verdict is re-estimated at higher K for the occupancy only.
3. **(iii) Test-set occupancy totals**, all models (the round-3 461/807/448 split was validation;
   labelled as such, test version produced).
4. **(iv) Full re-emit.** `results/claims_bernoulli.json`: all 168 pairwise Bernoulli+native
   verdicts × b_op {1.10, 1.15, 1.20}, plus the 28 negative-control pairs re-scored the same way,
   so the machine-readable object covers every Bernoulli verdict the text references. Deterministic
   given (i).
5. **Controls before any Bernoulli interval enters prose:** 28 negative controls under Bernoulli;
   b_op sweep for every quoted pair; single pre-registered seed/B (no knife-edge reruns).
6. **Units:** occurrence-IG is reported per positive, count-IG per event; the three-scoring table
   must not invite cross-column magnitude comparison.
7. Optional: the test scoring of the w=0.6 selection (proper-score re-selection), reported if used.

## Recorded expectation (pre-run, for honesty about direction)
Because the cascade's native occupancy over-forecasts occurrence (~1.75× on validation), native
scoring makes the cascade PAY for that over-forecast while the ML's trim does not, so the
hybrid-vs-cascade Bernoulli edge may be LARGER under native scoring than the scalar-recalibrated
residual (+0.099) — a real, non-global, quiet-cell-concentrated calibration improvement — OR it may
stay ~+0.1 if the trim is mostly global. Ranking verdicts (PR-AUC) are score-invariant and unchanged.

## Unchanged
Registered count-scored rule and all base-protocol + Amendment 1–2 items. The claims-of-record file
is unchanged; `claims_bernoulli.json` is an additional, clearly-secondary object.
