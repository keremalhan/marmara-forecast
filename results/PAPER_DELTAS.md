# PAPER_DELTAS — manuscript revisions required by the v2 analysis

Every comparative sentence must now quote `results/claims.json` (machine-derived
block-bootstrap verdicts), never a raw point difference. Supporting artifacts:
`results/bootstrap_ci.{json,md}` (95% CIs, B=2000, stationary block bootstrap over
window-ids), `results/evaluation.{json,md}` (8 predictors, power-labelled),
`results/gnss_v2_decision.json`, `results/transfer_naf.md`, `results/csep/`.

**Verdict rule (pre-registered, in code):** model A "beats" B only if the 95% CI of
the paired difference excludes 0 in A's favour for BOTH information gain AND PR-AUC;
otherwise "inseparable".

---

## 1. Abstract — "the cascade ranks best (PR-AUC 0.130 vs 0.126)"

**DELETE.** The 0.130-vs-0.126 gap is within noise.

**REPLACE WITH:** "On the operational M≥3.5 target the cascade, the ML-hybrid and all
ETAS variants (first-generation, our converged sv-ETAS, and an independent Mizrahi
inversion) are statistically **inseparable** (95% block-bootstrap CIs — e.g. cascade
vs first-gen ETAS: ΔPR-AUC CI [−0.007, +0.025] straddles 0), while **all decisively
beat the non-clustering baselines** (cascade vs Poisson: ΔPR-AUC [+0.045, +0.156],
ΔIG [+0.44, +1.38] nats/event). On the **powered M≥3.0 target** (≈3.5× the positives)
the physics-based cascade/ETAS models **decisively beat the ML hybrid** (hybrid vs
cascade: ΔPR-AUC CI [−0.106, −0.056], ΔIG [−1.46, −0.37]) — more positives did not
let the ML overtake the physics."

## 2. "The hybrid cannot lose to the cascade by construction"

**APPEND the essential qualifier:** "...**on the validation set** (the blend weight w
is chosen on validation). Out of sample the guarantee does not hold: on M≥3.5 test
the hybrid trailed the cascade on PR-AUC (0.117 vs 0.130) while leading on IG
(ΔIG CI [+0.04, +0.47]) — net inseparable; and on the powered M≥3.0 test the cascade
**beats** the hybrid on both metrics (B_beats_A)."

## 3. "The spatial problem is solved"

**REPLACE WITH:** "Spatial skill is mature — consistent with the smoothed-seismicity
literature, all clustering models achieve similar ROC-AUC/Molchan. The where/when
decomposition shows the **residual difficulty is temporal, not spatial**: models
separate only where forecasts must place probability in *time* (active sequences),
which is why the powered M≥3.0 target resolves differences the M≥3.5 target cannot."

## 4. "GNSS adds no measurable skill (null result)"

**⚠ CORRECTED BY PHASE-V AUDIT — C3 is VOID. The earlier "genuine signal" draft was
overstated and MUST NOT appear.** (See results/verify/gnss_verdict.md.)

**REPLACE WITH the audited rigorous null:** "We rebuilt the GNSS channel properly
(`gnss_v2`: per-component E/N trajectory model with secular + known steps +
annual/semiannual terms, common-mode filtering, IDW over ≤60 km stations, and a
Delaunay strain-rate invariant) and gated it identically. Its truncated self-test is
bit-for-bit causal at 13 cutoffs incl. an adversarial mid-month. **It nevertheless
yields no statistically-supported skill.** The source-screening test IG (+0.098) has a
95% block-bootstrap CI of **[−0.016, +0.207] (includes 0)**, and it does **not
collapse under time-shuffle or circular-shift placebos** (placebo IG means +0.119 /
+0.088 ≈ the real value) — the hallmark of noise, not time-aligned deformation. The
apparent operational gain on the powered M≥3.0 target likewise lies **inside** its
time-shuffle placebo null (real ΔPR-AUC +0.033 vs null [−0.003, +0.063]; real ΔIG
+1.18 vs null [−0.12, +1.25]) and is therefore spurious. It is not a
station-availability artifact (coverage-only IG ≈ 0). **This strengthens the null to
feature-engineering-robust and independently vindicates the v1 GNSS null** — onshore
GNSS resolves the offshore Main Marmara Fault too weakly to add forecast skill. We
still disclose the v1 scalar's design flaws (`hypot(E,N)` before detrending, no step
correction, no seasonal, nearest-station only), but the corrected channel confirms,
rather than overturns, the null. `hybrid_gnss` is reported only as a documented
negative; the operational model uses `hybrid` (no GNSS)."

## 5. Baseline section — retire "ETAS" as a monolith

**REPLACE the single "ETAS baseline" with the named generations:** (a) first-generation
expected-counts ETAS (`firstgen_etas`); (b) the cascade (full branching Monte-Carlo);
(c) **sv-ETAS** (`sv_etas`) — our own fit with EM stochastic declustering run to
convergence (≤10 iters) and a 5 km-floor bandwidth. **Report honestly that sv-ETAS ≈
first-gen**: the published fit's weighted-KDE background was already converged at 2
iterations (μ 0.683 vs 0.703, α 1.389 vs 1.386, p 1.067 vs 1.066), so converging it
changes essentially nothing — a robustness result, not new skill. (d) an
**independent third-party inversion** (`modern_etas`, lmizrahi/etas 3.0.0) on the
same catalogue slice — a tapered-Omori fit (p=0.81, q=0.89, b=1.76) structurally
different from ours yet **competitive** (highest ROC-AUC 0.894 and Molchan 0.785 of
all predictors on M≥3.5): the in-house ETAS is **not a strawman**.

## 6. Targets — add y30 (primary, powered); demote y45 (unpowered)

**ADD** the M≥3.0 table as the **primary** powered comparison (592 test positives,
≈3.5× M≥3.5), where ML-vs-ETAS separation is statistically resolvable (10/28 bootstrap
pairs resolved). **DEMOTE** M≥4.5 to descriptive/unpowered: 22 test positives, 25/28
pairs inseparable, CIs enormous — **state "unpowered; no ranking claims"** and
**remove the w=0.8 headline** (chosen on 13 validation positives — a noise-fit, not a
result).

## 7. ADD a CSEP section (`results/csep/`)

"Within a CSEP-style framework (catalog-based N/M/S/PL tests, Savran et al. 2020) the
cascade and sv_etas forecasts are **number- and magnitude-consistent** with observed
M≥3.0 seismicity over the test period (N-test ≈1450 forecast vs 1383 observed, pass;
M-test pass); the independent Mizrahi first-generation forecast **under-predicts** the
count (783, fail) and its steeper b fails the M-test. (Spatial/PL tests are confounded
by Poisson-catalogue cell-independence vs real within-cell clustering — documented;
native-clustered S-testing is the noted refinement.) pyCSEP itself was installed but
does not import on this exFAT volume — the identical published tests were implemented
in-house."

## 8. Synthetic discriminator — state the circularity ceiling

**ADD:** "The synthetic discriminator is trained on ETAS simulations, so **by
construction it cannot detect beyond-ETAS physics**; the M≥5.5 timing null is
therefore partly design-bounded, not purely empirical. Label the Sındırgı-vs-Kumburgaz
comparison **illustrative (n=2)** rather than a skill claim, unless it is extended to
≥20 scored historical sequences."

---

## One-line summary for the abstract's contribution sentence

Replace any "we beat ETAS" framing with: **"A leakage-audited ML forecaster matches —
but does not beat — first-generation, cascade, converged, and independent third-party
ETAS on the powered M≥3.0 target (all inseparable among the physics models; the ML
hybrid in fact trails them), all decisively above non-clustering baselines and CSEP
number/magnitude-consistent; and a properly re-engineered GNSS channel yields no
statistically-supported skill (apparent gains fail time-shuffle/circular-shift
placebos and the IG CI includes 0), confirming rather than overturning the GNSS
null."**
