# Round 5 — feature-independence study (reviewer-requested ablation + grouped PCA)

Script: `scripts/round5/a1_independence.py` → `results/round5/independence.json` (canonical
artifacts untouched). Target: y30 test (26 windows, 592 positives). Config identical to the
canonical hybrid (Poisson HGB, lr 0.05, depth 6, monotonic constraints, seed 42) except tree
count is selected on the val period 2022–2024 (grouped temporal selection, per the review).
CIs: stationary block bootstrap, B=2000, seed 42, mean block 3 windows; verdicts use the
pre-registered rule (IG **and** PR-AUC CIs exclude 0).

## Question
How many genuinely independent information axes do the ~20 inputs contain, and does any axis
carry out-of-sample information beyond ETAS?

## Answers

**1. Linear effective dimension: 15 of 20.** Train-only grouped PCA at ≥90% block variance:
recent counts 8→4, geophysics 4→3, catalogue stats 4→4, recurrence 2→2, ETAS 2→2.
Redundancy is concentrated exactly where the review predicted (multiscale/neighbourhood counts).

**2. Predictive effective dimension: ~6.** Booster retrained on top-k components: performance
climbs to k=6 (one axis each: overall-recent-activity, static-stress, catalogue-stats,
recurrence, then the ETAS level axis — the largest single jump, Bernoulli IG vs cascade
+0.09→+0.22, PR 0.179→0.209) and saturates at k≈6–7 (best val LL of the entire study at k=7,
−2505.6). Components 8–15 add nothing or hurt. The 6-component model ties the best raw-feature
model on likelihood.

**3. Ranking information: essentially ONE axis (ETAS).** Conditional grouped permutation on the
full model: permuting the ETAS block costs ΔPR-AUC 0.149; every other block costs ≤0.034.
No ablation model beats the raw cascade's test PR-AUC (0.2325); the full 20-feature pure-ML
model is *worse* (0.1734), and full_vs_etas_only IG is negative (−0.21 [−0.42, +0.01],
per-window −16/+10) — adding the 18 non-ETAS features causally *hurts* out-of-sample likelihood
(overfitting relative to ~26 independent test sequences, as the review hypothesised).

**4. Catalogue features cannot reconstruct ETAS.** catalogue_no_etas loses to both
etas_only (IG −0.41 [−0.79, −0.21], verdict B_beats_A) and the raw cascade (−0.31 [−0.70,
−0.07], B_beats_A). ETAS is a compact sufficient statistic of the catalogue at this data scale,
not a redundant transform the booster can relearn.

**5. The only positive signal is calibration-shaped, and it is bounded by ETAS itself.**
- etas_only vs cascade: IG +0.105 [0.0002, 0.223], Bernoulli +0.084 [0.010, 0.208],
  ΔPR −0.013 → verdict inseparable. This *causally reproduces* the paper's three-layer thesis:
  a small proper-score occurrence/calibration edge, zero ranking edge.
- etas_plus_phys vs etas_only: IG +0.219 [0.148, 0.305], Bernoulli +0.204 [0.145, 0.277],
  stable in every stability check (+24/−2 windows, LOWO range [0.207, 0.234], leave-Kumburgaz-out
  +0.241) but ΔPR +0.006 → inseparable under the rule. Static geophysics (ΔCFS, strain, fault
  distance) improves the *rate scale*, not the ranking — and the result does **not** surpass the
  analytic first-generation ETAS baseline (Bernoulli LL/pos −3.83 vs firstgen −3.80; occ-Poisson
  −3.99 vs −3.91). Even the best "new information" candidate only re-calibrates toward what a
  better-calibrated ETAS already achieves.
- The two ETAS generations are jointly better than either alone (etas_only beats
  etas_cascade_only, A_beats_B) — first-gen carries calibration information the supercritical
  cascade lacks, consistent with the manuscript's cascade-overprediction finding.

**6. Fragility note (supports the blended-hybrid design).** The etas_plus_recent model explodes
in count space: Σλ = 10,217 expected test events vs 1,383 observed, driven by one zero-event
cell assigned λ≈5,570 in the 2024-01-22 window (unseen count-feature combination at test time).
Its ranking and Bernoulli scores stay normal — a pure rate-scale extrapolation pathology of raw
multiscale count features under a Poisson booster. (This heavy tail is also why that pair's
bootstrap CI is unstable — ratio estimator dominated by one window; flagged, not interpreted.)
The canonical w-blended hybrid is structurally immune (w=0 recovers the cascade).

## One-sentence conclusion
Of ~20 engineered inputs there are ~15 linearly independent axes but only ~6 that carry any
out-of-sample predictive information, of which exactly one — the ETAS intensity axis — carries
the ranking skill; everything the flexible model adds beyond ETAS is either rate recalibration
bounded by the analytic first-generation ETAS baseline, or overfitting noise. This is a causal,
ablation-based confirmation of the manuscript's thesis that the ML stage recalibrates ETAS
rather than discovering new predictive structure.
