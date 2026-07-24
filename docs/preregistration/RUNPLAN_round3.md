# RUNPLAN — round 3 experiment battery (2026-07-14, pre-registered before running)

This is the **third** test-touch of the study (the paper discloses two prior campaigns). To keep
it protocol rather than a new peek: every run below is listed with its decision rule *before*
execution; this file is hashed and the hash recorded via the dated-amendment mechanism; all
tuning/triage is on **validation**; **test is scored once**, in a single batch, only for
validation survivors. No manuscript edit is made until the batch completes and results are
reviewed.

Isolation rule: every run writes to `results/round3/` (or prints), NEVER overwriting canonical
`predictions_*.parquet`, `claims.json`, `etas_params.pkl`, or `evaluation.*`.

## Promotion gate (verbatim; applied to any positive result)
A candidate is promoted only if it (a) passes the two-axis rule (both IG and PR-AUC 95% block-
bootstrap intervals exclude zero in its favour) on the SINGLE pre-registered test batch, (b) holds
across b_op ∈ {1.10, 1.15, 1.20}, and (c) survives the appropriate negative control — seed-pair
refits at minimum, the full placebo battery if a data channel is involved.

## Decision matrix (fixed before results)
- scalar or isotonic recovers the +0.29 edge → headline sharpens to "the edge is one scalar / a
  monotone map."
- rank-loss or tuned booster separates AND survives the gate → headline reverses (better paper).
- argmax-w separates → report both selection rules + a rule-sensitivity paragraph.
- hybrid passes CSEP S → new positive claim (ML fixes the spatial defect).
- everything null → paper stands; each null becomes one sentence.

## Tier 0 — runs the paper needs regardless
- **T0-1 y45 reconciliation.** Score canonical model-box cascade and the wide-box-trained hybrid on
  the IDENTICAL model-box test set; print both PR-AUCs and IG. Decision: report the exact eval-set/
  construction difference that yields cascade 0.069 (canonical) vs 0.008 (wide-box path).
- **T0-2 hybrid through CSEP (S/PL).** Build hybrid clustered catalogues by per-cell ratio-thinning
  the cascade native catalogues (λ_hybrid/λ_cascade), push through the same pyCSEP path. Decision:
  report N/M/S/PL; pass S → new positive claim, fail S → decomposition confirmed by a consistency test.
- **T0-3 fit windows.** Verify (done, from code) all three fits use FIT_END=2021-12-31 (train-only).
  Decision: one-sentence disclosure (fact, not a refit).
- **T0-4 block-length sensitivity.** Re-run the paired bootstrap at mean block ∈ {2,3,5,8}; emit a
  verdict-stability table for the primary y30 pairs. Decision: report insensitivity (or not).
- **T0-5 negative controls.** Extend the seed-pair battery from 6 toward 20–30 if a cascade re-sim
  per seed is cheap. Decision: report empirical false-positive rate.
- **T0-6 decomposition module.** `mechanism_ig_split.py` is committed; write the formula down
  (IG = placement Σ y·Δlnλ + count −ΣΔλ) for Methods/S1.

## Tier 1 — ML's best shot, validation-first (triage on val; test once for survivors)
- **T1-1 scalar challenger.** Fit global s (MLE Npos/Σλ) and a 2-scalar {s_low, s_high} split by
  λ_sim median, on val Poisson-LL; score rescaled cascade on test. Recovers +0.29? → edge is 1–2 numbers.
- **T1-2 isotonic recalibration.** Fit monotone g(λ_sim) on val; score g(cascade) on test vs the
  hybrid. Matches hybrid? → "re-calibration" proven constructively.
- **T1-3 argmax-w.** Score hybrid at the rejected argmax (w=0.9 y30, 0.6 y35) under the two-axis rule.
- **T1-4 drop monotonicity.** Retrain booster with monotonic_cst=None; compare val LL, score test.
- **T1-5 tune booster.** Validation search over depth/iter/lr/l2 (~50 trials); score best on test.
- **T1-6 offset formulation.** Refit Poisson booster with ln(λ_sim) as OFFSET, not feature.
- **T1-7 pure-ML.** Drop the cascade coupling ln(λ_sim), w=1; the 19 features (which INCLUDE etas_rate = first-generation ETAS) are retained: does the booster rediscover ETAS-level skill without the cascade prior?
- **T1-8 rank objective.** Train a ranking loss (if lightgbm available; else documented skip);
  evaluate PR-AUC + placement term — the direct test of "no ranking skill."
- **T1-9 regime split.** Hybrid vs cascade separately on active (recent M≥4.5 within 25 km) vs quiet cells.
- **T1-10 diagnostics.** (a) per-window ΔIG across the 26 test windows; (b) run the count/placement
  decomposition on VALIDATION (was the 216-nat climb all count-term too?).

## Unchanged / inherited
Shared scoring (`metrics.py`), block bootstrap (B=2000, seed 42, `bootstrap.py`), b_op sweep,
placebo battery, splits (train<2022, val 2022–23, test 2024+), base Mc=3.0. Base pre-registration
and Amendments 1–2 unchanged.
