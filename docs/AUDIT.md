# Leakage prevention

Data leakage, the flow of information from the evaluation target back into the features
or the model-selection procedure, is the single most common way that machine-learning
studies of seismicity report skill that does not survive honest forward validation.
Because rare large events are the quantity of interest, even a small leak is enough to
manufacture apparent M6 skill that is really base-rate rescaling. This benchmark is
therefore designed so that the common leakage modes **cannot occur**, and so that the
absence of look-ahead is *machine-checkable* rather than asserted (see
`src/marmara/tests/test_grid_leakage.py` and `docs/METHODS.md`). This document lists the
leakage modes and the design choice that rules out each one; the same list doubles as a
checklist for anyone evaluating an ML forecasting model.

## Leakage modes and the design that prevents them

1. **Class-balancing by row duplication.** Duplicating positive rows before a
   train/test split places copies of test positives into training. → Positives are
   weighted with a `sample_weight` column; **no row is ever duplicated**, so a positive
   cannot appear on both sides of a split.

2. **Threshold tuned on the test set.** Selecting the decision threshold (or any
   hyper-parameter) on the test data leaks the answer into the report. → The threshold
   and all hyper-parameters are chosen on a **validation slice**. The final configuration's
   test evaluation is deterministic; test-set scores were regenerated more than once across
   the study, but only as identical-configuration reproductions (no test-set tuning), as
   disclosed in the manuscript's limitations.

3. **Ablation that is not forward-chained.** Measuring a feature's importance over the
   whole record lets future information influence the estimate. → All ablation and
   evaluation is **forward-chained** (past-only): the model at window `t0` sees only
   events with time `< t0`.

4. **Hyper-parameter search over the test era, and target-encoding features.** Tuning a
   kernel half-life on all years (including the test era), or including a magnitude
   feature that partly encodes the target, both leak. → Kernel/half-life
   hyper-parameters are tuned on **pre-test data only**, and target-encoding features are
   excluded.

5. **A leakage test that does not test for leakage.** A "permutation test" that never
   actually permutes the target cannot detect leakage, and can miss look-ahead that is
   not target-correlated. → We use a **truncated-catalogue self-test**: features at
   `(cell, t0)` are recomputed from a catalogue truncated to `< t0` and must reproduce
   the stored grid **bit-for-bit**, and no feature may correlate `> 0.999` with a target.
   This checks causality directly rather than through a proxy.

6. **An orchestrator with placeholder steps.** If the "end-to-end" run does not actually
   execute the real stages, the reported numbers are not reproducible. → The reproduction
   script (`run_all.sh`) runs the real, ordered stages with the four correctness gates as
   hard stops.

7. **Silent model overwrite in cross-validation.** Because `estimator.fit()` returns
   `self`, a CV loop that reuses one estimator object can silently refit and persist a
   *test-fitted* model. → Estimators are `clone()`d, so the persisted model is only ever
   the validation-selected one.

## Causal data processing

Three properties of the catalogue itself would otherwise leak or bias the features, and
are enforced during catalogue construction (`src/marmara/catalog.py`; provenance in
`data/fetch_manifest.json`):

- **True UTC timestamps.** Source bulletin timestamps are in Türkiye local time; they
  are converted to UTC and verified empirically against a known anchor event. An
  uncorrected offset would corrupt every time-since-last-event and short-window feature.
- **Typed, homogenized magnitudes.** Magnitudes of different types (ML, Md, Mw, Ms) are
  typed and converted to a proxy-Mw (`mag_w`) via Kadirioğlu & Kartal (2016), so that a
  single consistent magnitude scale defines both features and targets.
- **Blast screening.** Anthropogenic quarry blasts are removed, so the background rate
  and feature counts reflect tectonic seismicity only.

## The consequence of enforcing causality

The honest cost of preventing leakage is that the headline numbers are **smaller, and in
places negative**, the machine-learning model does not beat a properly-fit ETAS on the
rarer targets, and several extension data sources add no measurable gain. Those results
are reported as-is throughout `docs/METHODS.md` and the results artifacts. A
benchmark you cannot lose is not a benchmark; the value here is that every number
survives a causality gate any reader can run.
