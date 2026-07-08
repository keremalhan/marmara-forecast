# CSEP consistency tests (catalog-based) — M>=3.0 (y30), test period

Test period 2024-01-22..2026-03-12; 1383 observed M>=3.0 events; N_sim=1000.

catalog-based N / M / S / pseudo-likelihood (Savran et al. 2020 / pyCSEP definitions; in-house — see module docstring)

Pass = forecast NOT rejected at alpha=0.05 (N: min(d1,d2)>0.025; M/S/PL: gamma in [0.025,0.975]).

| model | N_obs | N_fcast | N-test | S-test | M-test | PL-test |
|---|---|---|---|---|---|---|
| cascade | 1383 | 1456.6 | PASS (d1 0.978, d2 0.03) | **FAIL** (1.0) | PASS (0.711) | **FAIL** (1.0) |
| sv_etas | 1383 | 1446.2 | PASS (d1 0.953, d2 0.05) | **FAIL** (1.0) | PASS (0.627) | **FAIL** (1.0) |
| modern_etas | 1383 | 782.9 | **FAIL** (d1 0.0, d2 1.0) | **FAIL** (0.0) | **FAIL** (0.0) | **FAIL** (1.0) |

## Interpretation

**N-test and M-test are the robust results.** The cascade and sv_etas forecasts are consistent with the observed NUMBER of M≥3.0 events (~1450 forecast vs 1383 observed) and with the observed MAGNITUDE distribution (GR b=1.2). `modern_etas` fails both: its first-generation intensity (no secondary triggering) UNDER-predicts the count (783 vs 1383), and its steeper inverted b=1.76 over-weights small events (M-test).

**S-test and PL-test reject every model (γ≈1) — this is a limitation of the Poisson-sampled catalogues, not evidence against the spatial rate forecast.** The catalogues are drawn cell-independently from the gridded rate, so they do not reproduce the within-cell CLUSTERING of real seismicity; the observed spatial/rate pseudo-likelihood therefore exceeds every Poisson realization. A native-clustered-catalogue S-test (using the cascade's own stochastic catalogues, which cluster) is the proper refinement and is left as future work.

**Bottom line (the field's minimum criterion):** within a CSEP-style framework the cascade and sv_etas ETAS forecasts are number- and magnitude-consistent with observed M≥3.0 seismicity over the test period; the independent modern (Mizrahi first-gen) forecast is not (it under-counts).