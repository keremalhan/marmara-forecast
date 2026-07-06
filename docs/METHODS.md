# Methods — the Marmara forecasting system

This is the methods guide for the `marmara` package: what each stage computes, the
design choices and their justifications, and the honest results. It is the single,
unified description of the system (the leakage-prevention design is in `docs/AUDIT.md`).

Every number below is reproduced from an artifact under `results/`; the file is named
in parentheses. All features are **strictly causal**: the forecast at window start
`t0` uses only events with time `< t0`.

## 0. Study region and grid

- **Model box:** 25.6–30.9° E, 39.6–41.9° N — the Main Marmara Fault system and the
  northern strands of the North Anatolian Fault, where the KOERI network is densest.
- **Wide box** (rare-event training only): 25.0–31.5° E, 39.0–42.5° N.
- **Grid:** 0.1° cells; **30-day** forecast windows stepped through time. The model
  box is 1,219 cells; the evaluation uses 261 windows.

## 1. Catalogue (`src/marmara/catalog.py` → `results/catalog_report.json`)

Homogenized, deduplicated, blast-screened KOERI catalogue, 2003-01-04 → 2026-07-06.

- 103,410 raw events → 103,388 after wide-box dedup; **31,329** in the model box
  (`catalog_report.json`).
- **Magnitude homogenization** to proxy-Mw (`mag_w`) via Kadirioğlu & Kartal (2016);
  the conversions are applied (and flagged) slightly below their stated validity
  ranges for a small tail — documented once, used consistently.
- **Completeness Mc = 3.65** (max-curvature mode 3.45 + 0.2). This is *inflated* by
  ~15k Md events piling near `mag_w` 3.45; the dominant modern ML population completes
  near `mag_w` 2.72. Because targets are on `mag_w`, **y35 = 3.5** stays above ML-
  population completeness and **y45 = 4.5** is far above Mc. Mc is used identically for
  feature counting and for every baseline (fair). See the `mc_caveat` field.
- Model-box counts (`counts_model_box`): **≥3.5: 5,021; ≥4.5: 192; ≥5.5: 12;
  ≥6.0: 1** (2025-04-23 Mw 6.2, Kumburgaz). The single real M6 is why M6 numbers come
  from Gutenberg–Richter (GR) extrapolation + synthetic validation, never from a
  classifier trained on one positive.
- **b-values** (ensemble, because the mixed catalogue makes b the dominant M6
  uncertainty): Aki MLE `b_aki ≈ 1.02`, b-positive (van der Elst 2021) `b_pos ≈ 1.54`,
  calibration-consistent operational `b_op = 1.2` (real-window regression slope ≈ 1.04).
  The `{1.02, 1.54}` pair brackets M5/M6 extrapolations.

## 2. ETAS (`src/marmara/etas_model.py`, `src/marmara/etas_fit.py`)

An in-house ETAS (Ogata, 1988): MLE fit, `expected_counts` (first-generation), and a
branching simulator. The likelihood machinery is validated by 5/5 unit tests
(`tests/test_etas.py`: synthetic-parameter recovery, causal filtering, Omori decay, GR
consistency of the simulator).

**STAI refit (honest null).** We applied a short-term aftershock-incompleteness
correction (Helmstetter et al. 2006 `Mc(t)`), base Mc = 3.0, and raised the branching
cap 0.95 → 0.999 to test whether the MLE would leave the cap. **STAI dropped 0
events** — at base Mc = 3.0 the post-M5.5 incompleteness window is sub-hour, so the
correction is a near-no-op here (it needs a *low* base Mc). Raising the cap did **not**
un-pin the fit: the MLE went straight to the new cap (n = 0.999, the documented
degenerate near-critical mode). We therefore **retain the 0.95 cap** for stable
simulation, applied identically to the feature ETAS and the ETAS baseline (fair). This
null is reported, not hidden (`results/etas_fit_report.json`).

## 3. Feature grid and the leakage self-test (`src/marmara/grid.py`, `grid_hybrid.py`)

19 causal features per (cell, window): multi-scale event counts (30/90/365 d, in-cell
and 3-/5-cell neighbourhoods), time since last M≥3.5 / M≥4.5 within 25 km, local
b-positive, rate ratios, mean depth, distance to nearest fault, and time-causal
Coulomb stress change ΔCFS (Okada 1992; King, Stein & Lin 1994) with per-segment
receivers. The hybrid grid adds `ln(λ_sim)` from the cascade.

**Leakage self-test (a methods contribution, hard gate —
`tests/test_grid_leakage.py`).** 26 (cell, t0) rows are recomputed from a catalogue
**truncated to `< t0`** and must reproduce the stored grid **exactly** (0.0
deviation); no feature may correlate `> 0.999` with a target. This makes the absence
of look-ahead machine-checkable and, unlike a target-permutation check, catches
look-ahead that is not target-correlated (`results/leakage_ok.json`).

## 4. Cascade Monte-Carlo forecaster (`src/marmara/cascade.py`)

The single biggest modelling improvement over first-generation ETAS. For `[t0, t0+H)`
we forward-simulate new background + residual-Omori offspring of **all** history +
**full recursive cascades**, vectorized across K sims (K = 500 backtest, 10,000 live).
Per-cell rare-magnitude rates are computed **analytically** from the dense λ(M≥3.5)
field via GR, `λ_cell(M≥X) = λ_cell(M≥3.5)·10^(−b·(X−3.5))`, which removes the
Monte-Carlo `λ6 = 0` artifact. **Anisotropy** (elliptical 3:1 kernel for M≥5.5
offspring, major axis from the parent's first-72 h aftershock PCA or the nearest fault
strike) is the operational default.

**Gate (4 checks, `tests/test_cascade.py` → `results/cascade_ok.json`):**
(a) future events change nothing (causal); (b) reliability slope on synthetic
catalogues within [0.8, 1.2] (**0.949**; the slope is a variance-sensitive estimator,
so the test uses K = 400 sims to suppress Monte-Carlo bias — it does not relax the
band); (c) the day-after-M6 rate is **1.79×** the first-generation `expected_counts`
(first-gen under-counts active sequences — exactly the weakness the cascade fixes);
(d) anisotropy elongates M≥5.5 offspring along-strike **2.77×** (vs 1.06 isotropic).

## 5. Hybrid forecaster and baselines (`src/marmara/train.py`, `baselines.py`, `evaluate.py`)

Hybrid rate `λ = λ_sim^(1−w)·λ_ML^w`, where `λ_ML` is a Poisson gradient-boosted model
on 30-day counts and the weight `w` is chosen on the **validation** per-event
log-likelihood. By construction the hybrid cannot lose to the cascade. Four baselines:
plain Poisson climatology, fault-proximity, smoothed seismicity, and the ETAS
simulation. Splits by window start: **train** ≤ 2021-12-31, **val** 2022–2023,
**test** 2024-01-01 → t0+30 d ≤ 2026-03-31 (target fully inside reviewed data; the
preliminary tail is used only by the live forecast).

**Information gain (IG)** is nats per observed event vs a baseline (positive ⇒ better),
scored through one shared function for the model and every baseline (Rhoades et al.
2011).

### y35 (M≥3.5), test — 167 positives (`results/evaluation.json`)

| predictor | PR-AUC | ROC-AUC | Brier |
|---|---|---|---|
| cascade (ETAS-sim) | **0.130** | 0.877 | 0.00498 |
| first-generation ETAS | 0.126 | **0.894** | 0.00497 |
| hybrid (cascade × ML, w = 0.7) | 0.117 | 0.884 | **0.00495** |
| smoothed seismicity | 0.044 | 0.886 | 0.00544 |
| Poisson climatology | 0.032 | 0.848 | 0.00532 |
| fault-proximity | 0.010 | 0.687 | 0.00526 |

IG(hybrid − baseline): vs Poisson **+1.14**, vs fault-proximity **+0.95**, vs smoothed
**+0.51**, vs cascade **+0.25**, vs first-generation ETAS **−0.38**.

**Honest verdict.** The **cascade beats first-generation ETAS on ranking** (0.130 vs
0.126) — and by ~1.8× inside active sequences (§4) — but the cascade, hybrid, and
first-gen ETAS are within noise of one another, and ETAS keeps the best ROC. The ML
hybrid edges the cascade in likelihood (IG +0.25) but not in ranking; **it does not
beat a properly-fit ETAS** (IG −0.38). This matches the literature that neural models
rarely beat ETAS, and is reported as-is. (A plain ML classifier without the cascade
prior clears the naive baselines — IG vs Poisson +1.07, fault +0.88, smoothed +0.66 —
but also loses to ETAS, IG −0.46; `results/evaluation_baseline.json`.)

### y45 (M≥4.5) — the overfit and its fix

On the model box (only **22 test positives**) the hybrid **overfits**: PR-AUC 0.033 <
cascade 0.066 < first-gen 0.067, and the hybrid Brier (0.00121) is *worse* than the
cascade's (0.00068) — the large IG-vs-cascade there (+1.30) is an artifact
(`evaluation.json`). The **wide-box remedy** (`src/marmara/widebox_y45.py` →
`results/widebox_y45_report.json`) trains on the wide box (593,775 rows, **201**
M≥4.5 training positives vs ~104) and evaluates on the model box only: the chosen
weight drops to **w = 0.1** (lean on the cascade, stop overfitting), the Brier improves
to 0.00078, PR-AUC 0.022 vs cascade 0.008, and **IG(hybrid vs cascade) = +0.735** — IG
and PR-AUC now agree. The eval set is still 22 positives (noisy), but the model is
correctly regularized. **Production y45 = the wide-box hybrid; model-box y45 is
diagnostic-only.**

## 6. Conditional large-event discriminator (`src/marmara/synthetic.py`)

A "bigger-one-ahead" classifier trained on synthetic sequences, because the real
catalogue has too few large events to train on directly. From **250** base ETAS sims +
BPT-timed characteristic mainshocks (M ~ U(6.8, 7.6) on the four segments, each with a
full aftershock cascade): **31,712** snapshots (**4.95%** positive) taken 1/3/7 d after
every sim M≥4.5; the label is a sim event ≥ (largest-so-far − 0.3) within 30 d.
Seismicity-only features (b-positive drop, sequence count, largest magnitude, time
since start, distance to nearest segment). **Test PR-AUC 0.180, ROC 0.738** (~3.6× the
base rate; `results/synthetic_report.json`).

Applied to the **real 2025 sequences** with no retraining, the classifier ranks the
dangerous sequence far higher: the escalating **Sındırgı doublet scores 0.277 (33rd
percentile)** vs the decaying **Kumburgaz sequence at ~0.01 (1st–2nd percentile)**.

**Synthetic M5.5 null.** Even with ETAS ground truth, a HistGB on seismicity-only
features gives only marginal M5.5+/90-day predictability over climatology (test PR-AUC
~0.03, IG vs sim-Poisson ≈ 0). Large-event timing is close to a Poisson process
modulated by clustering — the expected, honest result.

## 7. FTLS alert and renewal priors (`src/marmara/sequence_mode.py`, `renewal.py`)

- **Foreshock Traffic-Light System** (Gulia & Wiemer 2019): after the 2025-04-23
  Mw 6.2 the relative b-value dropped (b_seq/b_ref = 0.65 → RED, elevated
  large-aftershock hazard). FTLS is an **alert layer, not a rate multiplier**; the
  Gulia–Wiemer low-coverage pseudo-prospective caveat is printed in the report.
- **BPT renewal** (Matthews et al. 2002; Parsons 2004 anchor; α = 0.5,
  `results/renewal_report.json`): 30-year conditional P(M~7) — İzmit **0.1%** (ruptured
  1999), Ganos 8.1%, Princes Islands 13.7%, **Central Marmara / Kumburgaz 21.8%**
  (~260 yr elapsed). Combined Princes + Central 30-yr = **32.5%**, within the Parsons
  (2004) 30–50% ballpark (sanity anchor passes). Renewal is a large-event layer only;
  it is **never** blended into the y35/y45 products.

## 8. Live forecast and horizon products (`src/marmara/forecast.py`, `forecast_horizons.py`)

At t0 = 2026-07-05, next 30 days (`results/forecast/forecast_2026-07-05/`):

- Regional 30-day **P(M≥6) = 0.13%–4.56%** (b-ensemble), **central b_op = 1.12%** —
  the wide range honestly reflects the mixed-catalogue b uncertainty. For comparison,
  a plain-Poisson base rate is the same fraction-of-a-percent per month. **No claim of
  an imminent large event.**
- Highest 30-day M≥3.5 cell: **Marmara Denizi (28.35 E, 40.85 N), P ≈ 43%** — the 2025
  Mw 6.2 aftershock zone on the Central Marmara segment.
- **Quarter / year products with backtest scaling** (`multi_horizon.json`): four annual
  backtests give predicted 51.4 vs realized 62 M≥3.5 counts, reliability slope 1.21
  (within [0.7, 1.3]), so the global scaling factor is 1.0 (no correction needed).
  Central-`b_op`: quarter (90 d) P(M≥6) 3.9%, year (365 d) P(M≥6) 15.7%.

## 9. Prospective protocol (`src/marmara/prospective.py`, `scripts/`)

A monthly job issues a 30-day forecast at the catalogue end, **appends it to a
sha256-hashed, append-only log before the outcome is known**, and scores past
forecasts once their window closes (`results/prospective/`). This is genuine
out-of-sample credibility — it cannot be hindsight-gamed like a backtest. The catalogue
is refreshed monthly from the KOERI preliminary XML feed
(`scripts/refresh_monthly.py`).

## 10. Extension data sources (honest nulls, `src/marmara/sources/`, `source_ig_test.py`)

A harness measures the marginal IG of a new data source over the hybrid, with the same
leakage gate. Results (`results/source_ig_*.json`):

- **GNSS static-strain coupling:** tested (a `gnss_rate_change` column derived from the
  interseismic strain field), leakage-clean (max |corr| 0.012), IG val −0.014 / test
  −0.063 → **no measurable gain**. The static field is time-invariant by design; a
  time-varying coupling signal would need station time series.
- **Dense (ML-repicked) micro-catalogue** and **repeating-earthquake creep**: the
  external data are not bundled, so these are documented as the highest-value future
  levers rather than tested here.

## Information-arrival analysis

A general procedure (`src/marmara/m62_countdown.py`): for a target event, freeze the
system at a schedule of strictly causal lead times, recompute all products from
pre-freeze data only, and track the target cell's spatial percentile, its probability
gain over a uniform baseline, and the discriminator score as functions of lead time.
This separates persistent spatial hazard (flat percentile) from transient temporal
information (a jump in gain when an informative event arrives). Applied to the 2025
events (`results/validation_final/m62_countdown.{md,png}`): the 23 April
Mw 6.2 fault cell was persistently the **top ~1% seismicity cell all year** (99th
percentile, gain 4–7×), but the specific event was **near-unforecastable in time** — the
cell was quiescent (0–2 M≥2 per 30 d), and only the lone M4.5 foreshock lifted the
30-day P(M≥6) gain to **42×** (and the discriminator to 0.76), ten minutes before the
mainshock. Run across the three 2025 events, the method places each on the same
**where-solved / when-foreshock-bounded** axis.
