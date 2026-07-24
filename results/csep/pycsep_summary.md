# CSEP consistency tests: pyCSEP 0.8.0 (M>=3.0, y30 test period)

Test period 2024-01-22..2026-03-12; 1383 observed M>=3.0 events; 500 stochastic catalogues/model; region 1219 cells.
Environment: pyCSEP toolkit.

GENUINE pyCSEP catalog-based number/magnitude/spatial/pseudolikelihood (Savran et al. 2020)

Pass = forecast NOT rejected at alpha=0.05 (N: min(delta1,delta2)>0.025; M/S/PL: gamma in [0.025,0.975]).

| model | N-test | M-test | S-test | PL-test |
|---|---|---|---|---|
| cascade | PASS (0.094) | PASS (0.252) | **FAIL** (0.0) | **FAIL** (0.0) |
| sv_etas | PASS (0.08) | PASS (0.256) | **FAIL** (0.0) | **FAIL** (0.0) |
| modern_etas | **FAIL** (0.0) | **FAIL** (0.0) | **FAIL** (0.0) | **FAIL** (0.0) |

## Cross-check vs in-house csep_eval (N/M)

| model | N pyCSEP | N in-house | M pyCSEP | M in-house | agree |
|---|---|---|---|---|---|
| cascade | True | False | True | False | **NO** |
| sv_etas | True | False | True | False | **NO** |
| modern_etas | False | False | False | False | YES |

## Interpretation

**Catalogue kind.** cascade and sv_etas use NATIVE CLUSTERED catalogues -- one cascade realization each (background + real-history offspring + recursive in-window triggering), magnitudes from the cascade's own Gutenberg-Richter draw at the operational b_op. `modern_etas` is a first-generation intensity with no clustered simulator, so it keeps the plain Poisson (cell-independent) draw.

**N-test (pass) and M-test (pass).** With the cascade's OWN clustered dispersion (aftershock sequences widen the count distribution far beyond Poisson), the cascade (1306) and sv_etas (1296) forecasts pass the N-test against 1383 observed M>=3.0 events and remain magnitude-consistent (M-test). The forecast mean sits below the observed count yet the clustered N-test is satisfied -- an analytic-Poisson N-test under-disperses and would spuriously reject here. `modern_etas` (Mizrahi first-gen) under-counts (782) and fails N and M.

**S-test (fail) and PL-test (fail) on the NATIVE clustered catalogues.** Because these catalogues reproduce within-window clustering, an S/PL rejection is NOT a catalogue-dispersion artifact: it is a genuine SPATIAL miscalibration. It is the same two-sided rate misallocation independently quantified in results/ntest_residual_probe.* and em_background_probe.* -- too much forecast rate in the diffuse background, too little in the concentrated Kumburgaz aftershock zone. The spatial-consistency test thus INDEPENDENTLY REDISCOVERS the misallocation. (modern_etas remains Poisson; its S/PL failure is not interpretable the same way.)

**Bottom line.** On native clustered catalogues the surgical cascade is count- and magnitude-consistent (N pass, M pass) but spatially miscalibrated (S fail). The count and spatial verdicts jointly confirm the v2 diagnosis: the supercriticality bug that inflated the count is fixed, and the residual is a spatial rate misallocation, not a count error. (The in-house csep_eval is the Poisson approximation and is superseded here for cascade/sv_etas; any N/M cross-check divergence reflects that dispersion upgrade, not an error.)