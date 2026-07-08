# M2 — Rewritten sections (drop-in prose for `paper/paper_seismica.md`)

Each block is publication-ready and marked with the v1 anchor it replaces or follows.
Numbers match `manuscript_v2/01_claims_map.md` (⇐ `claims.json`) exactly.

---

## ▸ REPLACE the Abstract (§ before "## Abstract" body)

The Sea of Marmara hosts a seismic gap beneath a metropolitan region of about 18 million
people, and the 23 April 2025 Mw 6.2 Kumburgaz earthquake renewed interest in short-term
forecasting there. We present a leakage-audited forecasting benchmark on a strictly
causal, homogenized KOERI catalogue (31,329 model-box events, 2003–2026), built around a
machine-checkable causality gate — a truncated-catalogue self-test that recomputes every
feature from data available at the forecast time — and evaluated under a **pre-registered
statistical rule**: one model beats another only if a stationary block bootstrap
(B=2000, resampling whole 30-day window-ids) puts the 95% confidence interval of the
paired difference away from zero for **both** information gain and precision–recall AUC.
Against this rule we compare an ETAS×ML hybrid to four physics forecasters — a
first-generation ETAS, a conditional cascade Monte-Carlo model, a spatially-variable-μ
ETAS fit by EM declustering, and an independently-implemented Mizrahi ETAS. On the primary
powered target (M≥3.0 in 30 days) **the physics models decisively outperform the machine
learning**: every ETAS variant beats the hybrid on both axes (e.g. cascade PR-AUC 0.229
vs 0.146; information gain −0.88 to −1.46 nats/event against the hybrid), while the four
physics models are mutually inseparable. On the M≥3.5 target the hybrid and all ETAS
variants become statistically indistinguishable, and every clustering model beats the
non-clustering baselines. A GNSS deformation channel that appeared to lift the hybrid
(bootstrap CI excluding zero) is shown by a placebo battery to be **spurious** — a
time-shuffled GNSS series reproduces the same "gain" — and is reported as a null,
illustrating that CI significance is necessary but not sufficient without label-shuffle
controls. Genuine pyCSEP consistency tests confirm the cascade and sv-ETAS forecasts are
number- and magnitude-consistent with observed seismicity while the first-generation
Mizrahi model under-counts. A conditional large-event discriminator ranks the escalating
2025 Sındırgı sequence far above the decaying Kumburgaz sequence, and an
information-arrival analysis shows the Mw 6.2 was spatially localizable all year (top ~1%
cell) but timing was foreshock-bounded — the lone M4.5 foreshock raised the 30-day M≥6
probability gain to 42× only ten minutes beforehand. All negative results are reported;
the live 30-day regional M≥6 probability is about one percent.

**Keywords:** earthquake forecasting; ETAS; machine learning; data leakage; block
bootstrap; CSEP; Sea of Marmara; operational earthquake forecasting; foreshocks;
information gain

---

## ▸ ADD to Methods (§3) — new subsection "3.x Statistical inference: pre-registered block bootstrap"

Model comparison is governed by a rule fixed before the test set was scored. For each
ordered pair of forecasts and each target we resample the 26 test windows with a
Politis–Romano stationary block bootstrap (mean block length 3 windows, B=2000 replicates,
seed 42), keeping all 1219 cells of a window together in every draw. Windows tile at
30-day steps; although the targets are non-overlapping they are temporally dependent
through shared feature history and aftershock clustering, so an i.i.d. row-level bootstrap
would badly understate the variance — in this study the block CI is roughly seven times
wider than the naïve row CI (`results/verify/v3_bootstrap_audit.json`). We compute the
paired difference of information gain (Poisson log-likelihood per event, nats) and of
precision–recall AUC, take the 2.5/97.5 percentiles as the 95% CI, and declare that
**A beats B only if both intervals exclude zero in A's favour**; otherwise the pair is
*inseparable*. All 168 pairwise verdicts are emitted to a machine-readable
`claims.json`, and no ranking is stated in this paper that `claims.json` does not
license. We designate M≥3.0/30-day as the primary powered target (592 test positives),
M≥3.5 as a powered secondary (167 positives), and M≥4.5 as unpowered (no ranking claims).

## ▸ ADD to Methods (§3) — "3.x Physics comparators: sv-ETAS and an independent Mizrahi ETAS"

To avoid comparing against a strawman ETAS we add two modern physics baselines to the
first-generation model and the cascade. The **spatially-variable-μ ETAS (sv-ETAS)**
re-estimates the background field by expectation–maximization declustering: each event is
probabilistically assigned to background or triggered, and the background rate μ(x,y) is a
Gaussian-kernel density (Silverman bandwidth, 5 km floor) of the background-weighted
epicentres, iterated to convergence. The **Mizrahi ETAS** is fit with the independent
`lmizrahi/etas` package (v3.0.0; tapered Omori, inverted parameters) in an isolated
environment and its first-generation intensity is integrated on our grid. We find that
sv-ETAS is statistically indistinguishable from the first-generation model (Δ information
gain and ΔPR-AUC CIs overlap zero at both y30 and y35). This is not EM degeneracy: the
first-generation background is *already* strongly spatially variable (spatial coefficient
of variation of μ(x,y) = 1.27), the EM converged in 8 iterations with a stable
log-likelihood and a sane background fraction of 0.34, so the two backgrounds coincide
because the simpler model was already spatially adaptive (C5).

## ▸ ADD to Methods (§3) — "3.x GNSS deformation channel and the placebo battery"

We engineered a GNSS channel (v2) from 22 Marmara continuous-GPS stations (Nevada
Geodetic Laboratory IGS20 solutions): per-component trajectory models (secular +
annual/semiannual + antenna/earthquake steps) fit **only on epochs earlier than 365 days
before each forecast window**, a common-mode filter, inverse-distance interpolation, and a
Delaunay strain-rate field; every feature passes the truncated-catalogue causality gate
bit-for-bit. Because a deformation feature can enter the ML through spurious structure, we
pre-registered a placebo battery: the channel must beat time-shuffled, circularly-shifted
(≥2 yr), and spatial-support-only surrogates of itself. The augmented hybrid's apparent
lift did **not** clear this bar (Results §4.x); we therefore treat GNSS as a null and
exclude it from the headline ranking. Independent EPOS/MIDAS velocities cross-validate the
secular fits: after removing the expected ~24 mm/yr IGS20-vs-Eurasia frame offset, the
gnss-v2 secular residuals are below 1 mm/yr at all four co-located stations, so the null
is not an artifact of mis-fit secular motion.

## ▸ REPLACE the CSEP paragraph in Methods (§3) / Results (§4)

Consistency with observed seismicity is assessed with the catalog-based CSEP tests of
Savran et al. (2020) — number (N), magnitude (M), spatial (S) and pseudo-likelihood (PL) —
computed with **genuine pyCSEP 0.8.0**. (pyCSEP does not import on the exFAT working
volume, whose copy-mode install corrupts the bundled `matplotlibrc`; it runs cleanly in a
fresh APFS environment.) Each forecast supplies 1000 stochastic catalogues built by
Poisson-sampling its per-cell M≥3.0 rate over the test period and drawing
Gutenberg–Richter magnitudes; the same statistics are also implemented in-house as a
cross-check, and the pyCSEP and in-house N/M verdicts agree for all three models.

---

## ▸ REPLACE Results (§4) opening + primary-target subsection

**4.1 Primary target (M≥3.0/30 days): physics beats machine learning.** On the primary
powered target the four physics forecasters form an inseparable top cluster — cascade
PR-AUC 0.229, sv-ETAS 0.228, first-generation 0.223, Mizrahi 0.206 — well above the ML
hybrid (0.146) and the smoothed/Poisson baselines (0.125/0.124). Under the pre-registered
rule every physics model beats the hybrid on both axes: the hybrid trails the cascade by
0.88 nats/event of information gain (95% CI [−1.46, −0.37]) and 0.084 PR-AUC
([−0.106, −0.057]), and trails the first-generation ETAS by 1.46 nats ([−2.06, −0.92]) and
0.077 PR-AUC. The hybrid does not even cleanly clear the non-clustering baselines here
(worse information gain, marginally better PR-AUC → inseparable). First-generation ETAS is
the cleanest baseline-beater, exceeding both smoothed and Poisson on both axes
(information gain +0.66 and +0.62 nats). The added events at M≥3.0 give the test the power
to resolve what M≥3.5 could not: with more positives, the ML hybrid's disadvantage against
well-specified physics becomes statistically decisive.

**4.2 Secondary target (M≥3.5): machine learning matches physics.** At M≥3.5 the hybrid
and every ETAS variant are mutually inseparable (all paired CIs span zero on at least one
axis), while all clustering models beat the non-clustering baselines — the hybrid exceeds
the smoothed baseline by 0.51 nats ([+0.32, +0.77]) and Poisson by 1.14 nats
([+0.57, +1.78]). The v1 headline that the cascade "ranks best" at M≥3.5 (PR-AUC 0.130 vs
0.126) survives as a point estimate but is **not** a significant ranking: the cascade,
sv-ETAS, first-generation and hybrid are one statistical family at this magnitude.

**4.3 The GNSS channel is a null.** The GNSS-augmented hybrid appears to beat the plain
hybrid at y30 with both bootstrap intervals excluding zero (Δ information gain −1.18
[−1.93, −0.44], ΔPR-AUC −0.033 [−0.066, −0.011] in the hybrid's disfavour). The placebo
battery dissolves this: a time-shuffled GNSS series yields a mean information gain of
+0.119 and a circular-shift +0.088, straddling the "real" +0.098, whose own bootstrap CI
[−0.016, +0.207] includes zero; the operational y30 placebo places the real ΔPR-AUC
(+0.033) squarely inside the null band [−0.003, +0.063] and the real ΔIG (+1.18) inside
[−0.12, +1.25]. Nearly 60% of the apparent gain comes from a single window. We conclude
the channel carries no resolvable deformation signal and report it as a null — a concrete
illustration that a confidence interval excluding zero is necessary but not sufficient
evidence of a real effect.

**4.4 CSEP consistency (genuine pyCSEP).** Over the test period (1383 observed M≥3.0
events) the cascade is number-consistent (forecast mean 1457) and magnitude-consistent
(M-test quantile 0.067), as is sv-ETAS (1446; 0.076). The independent Mizrahi
first-generation model under-predicts the count (783; N-test fail) and fails the M-test —
its lack of secondary triggering costs it the aftershock population. The spatial and
pseudo-likelihood tests reject every model because the Poisson stochastic catalogues are
cell-independent and under-disperse relative to real clustered seismicity (e.g. cascade PL
observed 1718 vs simulated mean 165); this is a property of the catalogue approximation,
not evidence against the spatial forecast, and a native-clustered-catalogue S-test is
noted as future work. The pyCSEP and in-house N/M verdicts agree for all three models.
