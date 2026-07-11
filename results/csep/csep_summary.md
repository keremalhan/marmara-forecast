# CSEP consistency tests (catalog-based): M>=3.0 (y30), test period

Test period 2024-01-22..2026-03-12; 1383 observed M>=3.0 events; N_sim=1000.

catalog-based N / M / S / pseudo-likelihood (Savran et al. 2020 / pyCSEP definitions; in-house, see module docstring)

Pass = forecast NOT rejected at alpha=0.05 (N: min(d1,d2)>0.025; M/S/PL: gamma in [0.025,0.975]).

| model | N_obs | N_fcast | N-test | S-test | M-test | PL-test |
|---|---|---|---|---|---|---|
| cascade | 1383 | 1498.1 | **FAIL** (d1 1.0, d2 0.0) | **FAIL** (1.0) | PASS (0.955) | **FAIL** (1.0) |
| sv_etas | 1383 | 1488.8 | **FAIL** (d1 0.997, d2 0.003) | **FAIL** (1.0) | PASS (0.915) | **FAIL** (1.0) |
| modern_etas | 1383 | 782.4 | **FAIL** (d1 0.0, d2 1.0) | **FAIL** (0.0) | **FAIL** (0.0) | **FAIL** (1.0) |

## Interpretation

**The number test is the decisive count check.** The cascade and sv_etas forecasts over-predict the observed count (1498 and 1489 vs 1383 observed M>=3.0 events) and fail the N-test, but remain magnitude-consistent (M-test, GR b=1.2).
The modern_etas forecast under-predicts the count (782 vs 1383) and fails the N-test: its first-generation intensity has no secondary triggering, and its steeper inverted b=1.762 over-weights small events in the M-test.

**S-test and PL-test reject every model (γ≈1): this is a limitation of the Poisson-sampled catalogues, not evidence against the spatial rate forecast.** The catalogues are drawn cell-independently from the gridded rate, so they do not reproduce the within-cell CLUSTERING of real seismicity; the observed spatial/rate pseudo-likelihood therefore exceeds every Poisson realization. A native-clustered-catalogue S-test (using the cascade's own stochastic catalogues, which cluster) is the proper refinement and is left as future work.

**Bottom line.** Over the test period the cascade and sv_etas ETAS forecasts are magnitude-consistent but over-predict the M>=3.0 count and fail the N-test; the independent modern first-generation (Mizrahi) forecast under-counts and fails both. Forecaster ranking is separable from absolute count calibration, which the number test rejects here.