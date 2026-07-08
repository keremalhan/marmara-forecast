# M1 — Claims map (claim → verdict → evidence)

Every quantitative claim the v2/v3 manuscript may make, its machine verdict, and the
exact artifact that backs it. **All numbers here are copied from `results/claims.json`
(the pre-registered bootstrap referee) and `results/bootstrap_ci.md`; the manuscript
must not state a comparative ranking that this table does not license.**

**Pre-registered decision rule** (`claims.json.rule`): *A beats B iff the 95%
percentile bootstrap CI of the paired difference excludes 0 in A's favour for BOTH
information gain (nats/event) AND PR-AUC; otherwise the pair is inseparable.*
Bootstrap: Politis–Romano stationary block bootstrap over ordered window-ids (all 1219
cells per window kept together), B=2000, mean block 3.0, seed 42.

**Target power tiers** (`train.POWER`): **y30 (M≥3.0) = primary, powered**;
y35 (M≥3.5) = powered secondary (the original v1 acceptance target); y45 (M≥4.5) =
unpowered — no ranking claims. `claims.json` sets `primary:true` only on **y30/test**.
⚠️ `results/bootstrap_ci.md` still carries a stale "y35/test PRIMARY — acceptance
target" header from v1; the operative primary is y30/test. Numbers are unaffected.

Test split: `n=31694`, y30 positives 592, y35 positives 167, 26 windows,
2024-01-22 … 2026-03-12.

---

## Named claims C1–C6 (from REVIEW_PACKET Section V)

| # | Claim | Verdict | Evidence |
|---|---|---|---|
| **C1** | At y35 (M≥3.5) the ML hybrid, cascade, and all ETAS variants are statistically **inseparable**; every clustering model beats the non-clustering baselines. | **VERIFIED** | `bootstrap_ci.md` y35/test; V3 (machinery), V6 (verdicts faithful) |
| **C2** | At y30 (M≥3.0, powered) the physics models (cascade / sv-ETAS / modern-ETAS / first-gen) **decisively beat** the ML hybrid — both CIs exclude 0. | **VERIFIED** | `claims.json` y30/test primary rows (all `B_beats_A`) |
| **C3** | The GNSS-v2 channel carries genuine deformation signal resolvable at y30. | **VOID** | placebo battery non-collapse + IG CI includes 0 + operational placebo spurious → `results/verify/gnss_verdict.md` |
| **C4** | The independent Mizrahi (`modern_etas`) forecaster is competitive, not a strawman. | **VERIFIED** | y35/test ROC-AUC 0.8945 (highest); enters the inseparable y35 cluster |
| **C5** | sv-ETAS ≈ first-gen because the first-gen background was already spatially variable and converged. | **VERIFIED** | V4: first-gen μ(x,y) spatial CoV 1.27; EM 8 iters; bg fraction 0.34 |
| **C6** | CSEP: cascade & sv-ETAS number+magnitude-consistent; Mizrahi first-gen under-predicts. | **VERIFIED (real pyCSEP)** | genuine pyCSEP 0.8.0 N/M agree with in-house for all 3 models — `results/csep_v3/` |

---

## The primary ranking (y30 / test) — verbatim from `claims.json`

Full-split PR-AUC / ROC-AUC (higher better):

| model | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| cascade | **0.2294** | 0.8734 | 0.01717 |
| sv_etas | 0.2276 | 0.8758 | 0.01716 |
| firstgen_etas | 0.2230 | **0.8800** | 0.01638 |
| modern_etas | 0.2057 | 0.8780 | 0.01673 |
| hybrid_gnss | 0.1791 | 0.8760 | 0.01695 |
| hybrid | 0.1458 | 0.8720 | 0.01962 |
| smoothed | 0.1247 | 0.8688 | 0.01917 |
| poisson | 0.1239 | 0.8558 | 0.01761 |

Pairwise verdicts (Δ = A − B; * = CI excludes 0):

| pair | ΔIG [95% CI] | ΔPR-AUC [95% CI] | verdict |
|---|---|---|---|
| hybrid vs cascade | −0.884 [−1.465, −0.370]* | −0.0835 [−0.1062, −0.0565]* | **cascade beats hybrid** |
| hybrid vs sv_etas | −0.879 [−1.511, −0.307]* | −0.0817 [−0.1086, −0.0521]* | **sv_etas beats hybrid** |
| hybrid vs modern_etas | −1.100 [−1.757, −0.557]* | −0.0598 [−0.0893, −0.0263]* | **modern_etas beats hybrid** |
| hybrid vs firstgen_etas | −1.460 [−2.055, −0.916]* | −0.0771 [−0.1059, −0.0444]* | **firstgen_etas beats hybrid** |
| cascade vs sv_etas | +0.005 [−0.109, +0.118] | +0.0018 [−0.0070, +0.0099] | inseparable |
| cascade vs firstgen_etas | −0.576 [−0.704, −0.468]* | +0.0064 [−0.0108, +0.0263] | inseparable |
| sv_etas vs firstgen_etas | −0.581 [−0.685, −0.478]* | +0.0046 [−0.0059, +0.0195] | inseparable |
| modern_etas vs firstgen_etas | −0.360 [−0.572, −0.135]* | −0.0173 [−0.0292, −0.0059]* | **firstgen beats modern** |
| firstgen_etas vs smoothed | +0.657 [+0.505, +0.840]* | +0.0982 [+0.0671, +0.1292]* | **firstgen beats smoothed** |
| firstgen_etas vs poisson | +0.624 [+0.504, +0.750]* | +0.0991 [+0.0584, +0.1374]* | **firstgen beats poisson** |
| hybrid vs smoothed | −0.802 [−1.385, −0.234]* | +0.0211 [+0.0036, +0.0471]* | inseparable (mixed sign) |
| hybrid vs poisson | −0.836 [−1.471, −0.251]* | +0.0220 [+0.0052, +0.0458]* | inseparable (mixed sign) |

**Reading:** the four physics models form an inseparable top cluster that beats the ML
hybrid on both axes; first-gen ETAS is the cleanest baseline-beater. The hybrid cannot
even cleanly clear the smoothed/Poisson baselines at y30 (IG worse, PR better → mixed →
inseparable). This is C2.

---

## The GNSS trap (why C3 is VOID despite a "significant" CI)

`claims.json` **hybrid vs hybrid_gnss / y30 test**: `B_beats_A`,
ΔIG −1.177 [−1.932, −0.442]\*, ΔPR −0.0332 [−0.0661, −0.0106]\* — i.e. the bootstrap CI
says the GNSS-augmented hybrid beats the plain hybrid on both axes. **This CI is real but
the signal it measures is spurious.** The pre-registered placebo battery (Phase 2/V2)
shows the same lift survives when the GNSS series is time-shuffled or circularly shifted
≥2 yr:

- placebo IG mean **+0.119 / +0.088** vs real **+0.098** (placebos do NOT collapse);
- IG bootstrap CI on the real gain **+0.098, [−0.016, +0.207] — includes 0**;
- y30 operational placebo: real ΔPR **+0.033 inside null [−0.003, +0.063]**;
  real ΔIG **+1.18 inside null [−0.12, +1.25]** → **SPURIOUS**;
- 59.6% of the y30 gain concentrates in a single window (2024-04, 16 events).

**Lesson for the paper:** a bootstrap CI that excludes 0 is *necessary but not
sufficient*; a placebo/label-shuffle control is required to separate signal from
structured noise. The GNSS channel is reported as a **null** (vindicating the v1 GNSS
null), and hybrid_gnss is excluded from the headline ranking.

---

## Secondary target (y35 / test) — the "ML matches physics" result

Full-split PR-AUC: cascade 0.1300, sv_etas 0.1276, firstgen 0.1260, hybrid_gnss 0.1189,
hybrid 0.1174, modern 0.1103, smoothed 0.0440, poisson 0.0323.

- hybrid vs cascade / sv_etas / modern / firstgen: **all inseparable** (C1).
- hybrid vs smoothed +0.507 [+0.317,+0.774]\* IG, +0.073\* PR → **hybrid beats smoothed**.
- hybrid vs poisson +1.136 [+0.571,+1.776]\* IG, +0.085\* PR → **hybrid beats poisson**.
- ETAS family internally inseparable; modern vs firstgen `B_beats_A` (first-gen edges Mizrahi).

---

## Phase D (starving channels) and Phase C (CSEP) — one-line each

- **dense sub-Mc3 catalogue**: 36,098 AFAD events → 10,610 after homogenize/dedup/blast/Mc;
  leakage-pass, truncated bit-for-bit; **val IG −0.030, test IG +0.130 [0.045, 0.229]** →
  **NOT promoted** (fails val>+0.02 ∧ test>0; val/test sign disagreement).
- **repeating earthquakes**: catalogue absent (paywalled supplements) → reported *absent, not null*.
- **EPOS/MIDAS velocities**: cross-validation only; after removing the ~24 mm/yr IGS20-vs-Eurasia
  frame offset, gnss_v2 secular residuals **< 1 mm/yr** at 4 matched stations → trajectory model sound.
- **CSEP (genuine pyCSEP 0.8.0)**: cascade N pass (obs 1383 vs fcast 1457) + M pass (γ 0.067);
  sv_etas N pass (1446) + M pass (γ 0.076); modern_etas N fail (783) + M fail. S/PL reject all
  (Poisson-catalogue under-dispersion confound). pyCSEP N/M **agree with in-house for all 3 models**.

---

## Prospective (do not recompute — hash-chained, protected)

Live 30-day regional M≥6 probability ≈ **1%** (`results/prospective/`, hash-chain not touched).
