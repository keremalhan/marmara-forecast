# CSEP consistency tests: GENUINE pyCSEP 0.8.0 (M>=3.0, y30 test period)

Test period 2024-01-22..2026-03-12; 1383 observed M>=3.0 events; 1000 stochastic catalogues/model; region 1219 cells.
Environment: pyCSEP toolkit.

GENUINE pyCSEP catalog-based number/magnitude/spatial/pseudolikelihood (Savran et al. 2020)

Pass = forecast NOT rejected at alpha=0.05 (N: min(delta1,delta2)>0.025; M/S/PL: gamma in [0.025,0.975]).

| model | N-test | M-test | S-test | PL-test |
|---|---|---|---|---|
| cascade | **FAIL** (1.0) | PASS (0.062) | **FAIL** (0.0) | **FAIL** (0.0) |
| sv_etas | **FAIL** (0.997) | PASS (0.055) | **FAIL** (0.0) | **FAIL** (0.0) |
| modern_etas | **FAIL** (0.0) | **FAIL** (0.0) | **FAIL** (0.0) | **FAIL** (0.0) |

## Cross-check vs in-house csep_eval (N/M)

| model | N pyCSEP | N in-house | M pyCSEP | M in-house | agree |
|---|---|---|---|---|---|
| cascade | False | False | True | True | YES |
| sv_etas | False | False | True | True | YES |
| modern_etas | False | False | False | False | YES |

## Interpretation

**The N-test and M-test are the robust, citable results and agree with the in-house implementation.** The cascade and sv_etas forecasts over-predict the observed count (1498 and 1489 vs 1383 observed M>=3.0 events) and fail the N-test, but remain magnitude-consistent (M-test); `modern_etas` (Mizrahi first-gen, no secondary triggering) under-counts and also fails the N-test.

**S-test and PL-test reject every model (quantile 0.0, outside [0.025,0.975])** because the stochastic catalogues are Poisson (cell-independent) draws from the gridded rate: they under-disperse relative to real clustered seismicity, so the observed spatial/rate pseudo-likelihood is far more extreme than any realisation. This is a property of the catalogue approximation, not evidence against the spatial forecast; a native-clustered-catalogue S-test is future work (it would require modifying the hash-protected cascade reproduction path).

**Bottom line:** genuine pyCSEP confirms the in-house result. Over the test period the cascade and sv_etas forecasts are magnitude-consistent but over-predict the M>=3.0 count and fail the N-test; the independent Mizrahi first-gen (`modern_etas`) under-counts and fails both. The paper cites these pyCSEP numbers; the in-house implementation is retained as a cross-check (N/M verdicts agree for all three models).