# v2 pre-registration — Amendment 3, addendum (2026-07-15)

Appends to Amendment 3 (SHA-256 `0812a8dbdb7d9296b569325b7a98857245b8bf44604591c17b92a25760081314`).
Three additional pre-specified steps before any native-score number enters prose:

1. **MC Jensen-bias debias.** The native log-score uses p̂ from K catalogues; ln p̂ is downward-biased
   by ≈ (1−p)/(2Kp), asymmetric because clustered models are scored on noisy MC occupancy while
   first-generation intensities are scored on exact 1−e^(−λ). Recompute native scores at K ≈ 5000 on
   the 26 test windows and extrapolate the per-cell bias in 1/K to K→∞; report the debiased native IGs.
   Pre-stated prediction: physics-pair native IG residuals (measured − closed-form calibration
   differential) → ~0 after debias — i.e. among physics models the properly-scored likelihood axis is
   only calibration geometry.
2. **Thinning tie-break.** The hybrid native occupancy replaces capped thinning (ratio ≤ 1) with
   thinning-plus-Poisson-top-up: where λ_hyb > λ_cascade, superpose independent events at the excess
   rate. Pre-stated: if the occupancy-PR dPR collapses under top-up, the +0.011 was a construction
   artifact (capped thinning froze active cells, pushing negatives down for free); ranking claims rest
   on the construction-free INTENSITY PR axis regardless.
3. **Ranking axis of record = intensity PR-AUC** (invariant under monotone transforms; the
   pre-registered rule's axis). Occupancy-PR is a diagnostic only. Hybrid-vs-cascade native verdict =
   IG(Bernoulli, debiased) × intensity-PR.
4. Report the w=0.6 (Bernoulli-objective) selection's test scoring if it is used in prose.

Units fixed: occurrence-IG per positive, count-IG per event (factor m ≈ 2.34); one canonical chain.
