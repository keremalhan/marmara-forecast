# M5 — Limitations and future work (drop-in for §6)

Replaces/extends the v1 "Limitations and future work". Ordered by how much each bounds
the paper's claims.

1. **One region, one test period.** All results are for the Sea of Marmara model box over
   a single 26-window test period (2024–2026). The catalogue and fault geometry are
   Marmara-only, so generalization to the wider North Anatolian Fault is **untested**, not
   demonstrated (`results/transfer_naf.md`). The powered-target conclusion "physics beats
   ML" is a statement about this benchmark, not a universal claim.

2. **Bounded test-set exposure (disclosed).** The hybrid model family scored the test set
   three times over the study; all three were identical-configuration deterministic
   reproductions and all hyperparameters were selected on the validation split only
   (`results/test_touch_log.json`, `results/verify/v5_test_touch.json`). We disclose this
   rather than claim a single-touch ideal; no test-set information entered model selection.

3. **CSEP spatial/PL tests are confounded.** The catalog-based S and PL tests reject every
   forecast because the stochastic catalogues are Poisson (cell-independent) and
   under-disperse relative to clustered seismicity; only the N and M tests are
   interpretable here. A native-clustered-catalogue S-test — drawing the cascade's own
   branching catalogues with event locations — is the correct refinement and is left as
   future work (it would require exposing per-event output from the reproduction-locked
   cascade path).

4. **The GNSS null is conditional on this representation.** We show that *this* engineered
   GNSS channel (22-station IGS20 trajectory + strain-rate features) carries no
   placebo-robust signal over this period. A different deformation representation, a denser
   network, or a longer window could in principle differ; what the placebo battery
   establishes is that the apparent bootstrap-significant gain we observed is
   indistinguishable from structured noise, so a positive claim was not warranted.

5. **sv-ETAS ≈ first-gen is catalogue-specific.** The two backgrounds coincide because the
   first-generation μ(x,y) was already strongly spatially variable (CoV 1.27) in this
   catalogue. In regions where the first-generation background is near-uniform, EM
   re-estimation could matter; the equivalence should not be read as general.

6. **Dense-catalogue non-stationarity.** The sub-Mc3 AFAD channel showed a positive test
   information gain (+0.130 [0.045, 0.229]) but a negative validation gain (−0.030),
   consistent with completeness non-stationarity across the validation→test boundary; it
   failed the pre-registered promotion rule. A per-year-Mc-normalized feature or a longer
   validation window could revisit it.

7. **Completeness and homogenization.** Results depend on the KOERI magnitude
   homogenization and the assumed magnitude of completeness; the dense channel's per-year
   Mc estimation and the daytime-blast filter are heuristics.

8. **Large events are descriptive only.** M≥4.5 is unpowered (too few test positives); no
   ranking claim is made there, and the large-event discriminator results are presented as
   illustrative case studies, not calibrated skill.

9. **Circularity ceiling of the discriminator.** The large-event discriminator is trained on
   ETAS simulations, so by construction it cannot detect beyond-ETAS precursory physics; its
   Sındırgı-vs-Kumburgaz separation reflects clustering statistics, not a physical precursor.

10. **Repeaters absent, not null.** A repeating-earthquake channel was designed but its
    catalogue is paywalled and could not be fetched; it is reported as *absent* (no test run),
    distinct from the GNSS *null* (tested and dissolved).
