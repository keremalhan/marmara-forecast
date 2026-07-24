# VERDICT: forecast everything, no overfit/underfit

1. AGGREGATE calibration: ALL products calibrated in the mean (obs/exp within Poisson noise of 1: y35 1.019, y45 1.257, M5 0.855, M5.5 0.958, M6 1.208).
2. SHAPE (10-bin): y35 calibrated (slope 0.951); y45 calibrated (slope 0.91; wide-box y45 w=0.1 is the calibrated variant).
3. Rarity-limited shape (reported via aggregate, NOT tuned): P(M>=5.0) 30d (cascade), P(M>=5.5) 30d (cascade), P(M>=6.0) 30d (cascade).
4. M6.2 spatial (30d, 04-22): epicenter percentile by lam6 0 (by seismicity 99), gain6 0.0×, top-decile-lam6=False.
5. M6.2 short-term: post-foreshock 24h regional expected-M6 1.0e-03 = 1.3333333333333333× quiet, FTLS UNKNOWN (n_seq=3), but absolute probability stays small.
6. M6.2 gate ALL-PASS: False.
7. Physical finding: M6/30d/per-cell is below forecast resolution and had NO base rate (0 M6 in the box pre-event); the only real precursor was the 36-min M4 foreshock, and 10 min after it there were too few events for FTLS/classifier. That absent information IS the finding.
8. Underfit check (y35 test): cascade IG vs baselines all>0 = True; hybrid = False.
9. Supported claims: 30-day spatial hazard + sequence-response (FTLS/cascade) are calibrated and skillful; M≥6 short-term is an ELEVATION signal, not a probability of certainty.
10. NOT supported: sharp deterministic prediction of a specific large event ahead of its immediate foreshock; reported as measured, not tuned.