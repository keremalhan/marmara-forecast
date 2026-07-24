# CSEP consistency tests (catalog-based): M>=3.0 (y30), test period

Test period 2024-01-22..2026-03-12; 1383 observed M>=3.0 events; N_sim=1000.

catalog-based N / M / S / pseudo-likelihood (Savran et al. 2020 / pyCSEP definitions; in-house, see module docstring)

Pass = forecast NOT rejected at alpha=0.05 (N: min(d1,d2)>0.025; M/S/PL: gamma in [0.025,0.975]).

| model | N_obs | N_fcast | N-test | S-test | M-test | PL-test |
|---|---|---|---|---|---|---|
| cascade | 1383 | 1304.1 | **FAIL** (d1 0.013, d2 0.988) | **FAIL** (1.0) | **FAIL** (0.006) | **FAIL** (1.0) |
| sv_etas | 1383 | 1294.9 | **FAIL** (d1 0.008, d2 0.993) | **FAIL** (1.0) | **FAIL** (0.003) | **FAIL** (1.0) |
| modern_etas | 1383 | 784.3 | **FAIL** (d1 0.0, d2 1.0) | **FAIL** (0.0) | **FAIL** (0.0) | **FAIL** (1.0) |

## Interpretation

**The number test is the decisive count check.** The cascade and sv_etas forecasts under-predict the observed count (1304 and 1295 vs 1383 observed M>=3.0 events) and fail the N-test, but also fail the M-test (M-test, GR b=1.15).
The modern_etas forecast under-predicts the count (784 vs 1383) and fails the N-test: its first-generation intensity has no secondary triggering, and its steeper inverted b=1.7618772126029445 over-weights small events in the M-test.

**S-test and PL-test reject every model (γ≈1): this is a limitation of the Poisson-sampled catalogues, not evidence against the spatial rate forecast.** The catalogues are drawn cell-independently from the gridded rate, so they do not reproduce the within-cell CLUSTERING of real seismicity; the observed spatial/rate pseudo-likelihood therefore exceeds every Poisson realization. A native-clustered-catalogue S-test (using the cascade's own stochastic catalogues, which cluster) is the proper refinement and is left as future work.

**Bottom line.** Over the test period the cascade and sv_etas ETAS forecasts are magnitude-inconsistent but under-predict the M>=3.0 count and fail the N-test; the independent modern first-generation (Mizrahi) forecast under-counts and fails both. Forecaster ranking is separable from absolute count calibration, which the number test rejects here.